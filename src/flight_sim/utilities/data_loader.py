"""Functions for loading aerodynamic CSV data."""

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Literal, get_args

import numpy as np
import pandas as pd  # type: ignore

from flight_sim.units import Scalar, UnitChecked, Vector
from flight_sim.utilities.dcm import missile_to_body


@dataclass(frozen=True)
class AeroCoefficients:
    """Body-frame aerodynamic coefficients at one flight condition."""

    # Force components along the body X, Y and Z axes
    cx: float = field(metadata={"column": "Cx"})
    cy: float = field(metadata={"column": "Cy"})
    cz: float = field(metadata={"column": "Cz"})

    # Moment components about the body X, Y and Z axes
    cmx: float = field(metadata={"column": "CMx"})
    cmy: float = field(metadata={"column": "CMy"})
    cmz: float = field(metadata={"column": "CMz"})


# CSV columns in AeroCoefficients field order
_AERO_COLUMNS: tuple[str, ...] = tuple(
    coefficient.metadata["column"] for coefficient in fields(AeroCoefficients)
)

# Axes a table's components are along
AeroFrame = Literal["body", "missile"]

# How a table reaches past its grid: "linear" extends the edge cell, "clamp"
# holds the edge value
Extrapolation = Literal["linear", "clamp"]


@dataclass(frozen=True)
class AeroTable(UnitChecked):
    """Trilinear lookup of aerodynamic coefficients over Mach, alpha_tot and phi_a.

    The frames and angles are defined in ``flight_sim.utilities.dcm``, with the
    angles in degrees; phi is wrapped into [0, 360). Outside the grid, each
    coefficient follows ``extrapolation`` along each axis it is off.
    """

    mach_axis: np.ndarray  # Strictly increasing, at least two values
    alpha_axis: np.ndarray  # Degrees, strictly increasing, at least two
    phi_axis: np.ndarray  # Degrees, strictly increasing, at least two

    # Shape (len(mach_axis), len(alpha_axis), len(phi_axis), 6), last axis in
    # AeroCoefficients field order
    values: np.ndarray

    # Forces are normalized by the reference area, moments also by the length
    reference_area: Annotated[Scalar, "m**2"]
    reference_length: Annotated[Scalar, "m"]

    # Point the moments are taken about, from the nose tip in body axes
    reference_point: Annotated[Vector, "m"]

    # Missile-frame components are rotated into body axes after interpolation
    frame: AeroFrame

    extrapolation: Extrapolation

    # Reference values in SI units, cached on construction
    reference_area_m2: float = field(init=False, repr=False, compare=False)
    reference_length_m: float = field(init=False, repr=False, compare=False)
    reference_point_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the frame and extrapolation, and cache the reference values.

        Raises:
            ValueError: If the frame or extrapolation is not one of its options.
        """
        super().__post_init__()
        if self.frame not in get_args(AeroFrame):
            raise ValueError(f"Unknown frame {self.frame!r}; use {get_args(AeroFrame)}")
        if self.extrapolation not in get_args(Extrapolation):
            raise ValueError(
                f"Unknown extrapolation {self.extrapolation!r};"
                f" use {get_args(Extrapolation)}"
            )
        object.__setattr__(
            self, "reference_area_m2", float(self.reference_area.m_as("m**2"))
        )
        object.__setattr__(
            self, "reference_length_m", float(self.reference_length.m_as("m"))
        )
        object.__setattr__(self, "reference_point_m", self.reference_point.m_as("m"))

    def __call__(self, mach: float, alpha: float, phi: float) -> AeroCoefficients:
        """Interpolate every coefficient at one flight condition.

        Args:
            mach (float): Mach number.
            alpha (float): Total angle of attack in degrees.
            phi (float): Aerodynamic roll angle in degrees.

        Returns:
            AeroCoefficients: The interpolated coefficients in body axes.
        """
        phi = phi % 360.0
        clamp = self.extrapolation == "clamp"
        i, t_mach = _cell(self.mach_axis, mach, clamp)
        j, t_alpha = _cell(self.alpha_axis, alpha, clamp)
        k, t_phi = _cell(self.phi_axis, phi, clamp)
        cell = self.values[i : i + 2, j : j + 2, k : k + 2]
        # Collapse the cell along phi, then alpha, then Mach
        at_phi = cell[:, :, 0] + t_phi * (cell[:, :, 1] - cell[:, :, 0])
        at_alpha = at_phi[:, 0] + t_alpha * (at_phi[:, 1] - at_phi[:, 0])
        coefficients = at_alpha[0] + t_mach * (at_alpha[1] - at_alpha[0])
        if self.frame == "missile":
            to_body = missile_to_body(math.radians(phi))
            coefficients = np.concatenate(
                (to_body @ coefficients[:3], to_body @ coefficients[3:])
            )
        return AeroCoefficients(*coefficients.tolist())


def _cell(axis: np.ndarray, point: float, clamp: bool) -> tuple[int, float]:
    """Return the grid cell holding a point and the fraction of the way across it.

    Points beyond either end of the axis use the first or last cell, with the
    fraction held within [0, 1] when ``clamp`` is set.
    """
    index = min(
        max(int(np.searchsorted(axis, point, side="right")) - 1, 0), len(axis) - 2
    )
    fraction = (point - axis[index]) / (axis[index + 1] - axis[index])
    return index, min(max(fraction, 0.0), 1.0) if clamp else fraction


def aero_table_from_csv(
    filepath: str | Path,
    *,
    reference_area: Scalar,
    reference_length: Scalar,
    reference_point: Vector,
    frame: AeroFrame,
    extrapolation: Extrapolation,
) -> AeroTable:
    """Read a flight sim CSV and build one lookup covering every coefficient.

    Args:
        filepath (str | Path): CSV with "Mach", "Alpha" and "Phi" columns plus one
            column per ``AeroCoefficients`` field, with components along the
            axes of ``frame``.
        reference_area (Scalar): Area the coefficients are normalized by.
        reference_length (Scalar): Length the moments are also normalized by.
        reference_point (Vector): Point the moments are taken about, from the
            nose tip in body axes.
        frame (AeroFrame): Axes of the CSV's components.
        extrapolation (Extrapolation): How the table reaches past its grid.

    Returns:
        AeroTable: Lookup over the CSV's grid.
    """
    data_frame = pd.read_csv(filepath)

    mach_axis = np.sort(data_frame["Mach"].unique())
    alpha_axis = np.sort(data_frame["Alpha"].unique())
    phi_axis = np.sort(data_frame["Phi"].unique())

    values = np.stack(
        [
            data_frame.pivot_table(
                values=column, index="Mach", columns=["Alpha", "Phi"]
            )
            .to_numpy()
            .reshape(len(mach_axis), len(alpha_axis), len(phi_axis))
            for column in _AERO_COLUMNS
        ],
        axis=-1,
    )
    return AeroTable(
        mach_axis,
        alpha_axis,
        phi_axis,
        values,
        reference_area=reference_area,
        reference_length=reference_length,
        reference_point=reference_point,
        frame=frame,
        extrapolation=extrapolation,
    )
