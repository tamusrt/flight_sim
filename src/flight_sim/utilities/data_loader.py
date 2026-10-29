"""Functions for loading aerodynamic CSV data."""

import math
import warnings
from dataclasses import dataclass, field
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
    cx: float
    cy: float
    cz: float

    # Moment components about the body X, Y and Z axes, about the reference point
    cmx: float
    cmy: float
    cmz: float


# CSV columns, in the order of the table's last axis
_AERO_COLUMNS: tuple[str, ...] = ("Cx", "Cy", "Cz", "CMx", "xCP")

# Largest jump in xCP, in reference lengths, from the first nonzero alpha row
# to the zero alpha row before it is flagged
_ZERO_ALPHA_XCP_TOLERANCE = 1.0

# Axes a table's components are along
AeroFrame = Literal["body", "missile"]

# How a table reaches past its grid: "linear" extends the edge cell, "clamp"
# holds the edge value
Extrapolation = Literal["linear", "clamp"]


@dataclass(frozen=True)
class AeroTable(UnitChecked):
    """Trilinear lookup of aerodynamic coefficients over Mach, alpha_tot and phi_a.

    The force acts at the center of pressure, xCP reference lengths along body
    X from the reference point, which gives the pitching and yawing moments.
    The rolling moment is tabulated. At zero alpha, where the force vanishes,
    xCP is the limit of the CP as alpha goes to zero.

    The frames and angles are defined in ``flight_sim.utilities.dcm``, with the
    angles in degrees; phi is wrapped into [0, 360). Outside the grid, each
    coefficient follows ``extrapolation`` along each axis it is off.
    """

    mach_axis: np.ndarray  # Strictly increasing, at least two values
    alpha_axis: np.ndarray  # Degrees, strictly increasing, at least two
    phi_axis: np.ndarray  # Degrees, strictly increasing, at least two

    # Shape (len(mach_axis), len(alpha_axis), len(phi_axis), 5), last axis
    # Cx, Cy, Cz, CMx, xCP
    values: np.ndarray

    # Forces are normalized by the reference area, moments also by the
    # length, which xCP is measured in
    reference_area: Annotated[Scalar, "m**2"]
    reference_length: Annotated[Scalar, "m"]

    # Point xCP is measured from, from the nose tip in body axes
    reference_point: Annotated[Vector, "m"]

    # Missile-frame components are rotated into body axes after interpolation
    frame: AeroFrame

    extrapolation: Extrapolation

    # Reference values in SI units, cached on construction
    reference_area_m2: float = field(init=False, repr=False, compare=False)
    reference_length_m: float = field(init=False, repr=False, compare=False)
    reference_point_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the options and values, and cache the reference values in SI units.

        Raises:
            ValueError: If the frame or extrapolation is not one of its
                options, the values do not match the axes, or the values are not finite.
        """
        super().__post_init__()
        if self.frame not in get_args(AeroFrame):
            raise ValueError(f"Unknown frame {self.frame!r}; use {get_args(AeroFrame)}")
        if self.extrapolation not in get_args(Extrapolation):
            raise ValueError(
                f"Unknown extrapolation {self.extrapolation!r};"
                f" use {get_args(Extrapolation)}"
            )
        shape = (len(self.mach_axis), len(self.alpha_axis), len(self.phi_axis))
        if self.values.shape != (*shape, len(_AERO_COLUMNS)):
            raise ValueError(
                f"Aero table values must have shape {(*shape, len(_AERO_COLUMNS))}"
            )
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Aero table values must be finite")
        if self.alpha_axis[0] == 0.0:
            jump = np.abs(self.values[:, 0, :, 4] - self.values[:, 1, :, 4])
            if np.any(jump > _ZERO_ALPHA_XCP_TOLERANCE):
                warnings.warn(
                    f"Zero alpha xCP differs from the next alpha row by up to "
                    f"{jump.max():.3g} reference lengths; it should be the CP's "
                    f"limit as alpha goes to zero",
                    stacklevel=2,
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
        cx, cy, cz, cmx, xcp = self._interpolate(mach, alpha, phi)
        if self.frame == "missile":
            # Rolling about X leaves the axial force and rolling moment alone
            cx, cy, cz = missile_to_body(math.radians(phi)) @ np.array([cx, cy, cz])
        # Moment of the force acting at the CP
        return AeroCoefficients(
            float(cx),
            float(cy),
            float(cz),
            float(cmx),
            float(-xcp * cz),
            float(xcp * cy),
        )

    def _interpolate(self, mach: float, alpha: float, phi: float) -> np.ndarray:
        """Return the table's columns at a point, with phi already wrapped."""
        clamp = self.extrapolation == "clamp"
        i, t_mach = _cell(self.mach_axis, mach, clamp)
        j, t_alpha = _cell(self.alpha_axis, alpha, clamp)
        k, t_phi = _cell(self.phi_axis, phi, clamp)
        cell = self.values[i : i + 2, j : j + 2, k : k + 2]
        # Collapse the cell along phi, then alpha, then Mach
        at_phi = cell[:, :, 0] + t_phi * (cell[:, :, 1] - cell[:, :, 0])
        at_alpha = at_phi[:, 0] + t_alpha * (at_phi[:, 1] - at_phi[:, 0])
        return np.asarray(at_alpha[0] + t_mach * (at_alpha[1] - at_alpha[0]))


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
        filepath (str | Path): CSV with "Mach", "Alpha", "Phi", "Cx", "Cy",
            "Cz", "CMx" and "xCP" columns, with components along the axes of
            ``frame``.
        reference_area (Scalar): Area the coefficients are normalized by.
        reference_length (Scalar): Length the rolling moment is also
            normalized by and xCP is measured in.
        reference_point (Vector): Point xCP is measured from, from the nose
            tip in body axes.
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
