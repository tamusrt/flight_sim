"""Functions for loading aerodynamic CSV data."""

import math
from dataclasses import dataclass, field, fields
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd  # type: ignore

from flight_sim.aero_extend import extend_values
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


@dataclass(frozen=True)
class AeroTable(UnitChecked):
    """Trilinear lookup of aerodynamic coefficients over Mach, alpha_tot and phi_a.

    The frames and angles are defined in ``flight_sim.utilities.dcm``, with the
    angles in degrees; phi is wrapped into [0, 360). Outside the grid, each
    coefficient is extrapolated linearly from the nearest edge cell.
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

    # Axes the table's components are along; missile-frame components are
    # rotated into body axes after interpolation
    frame: Literal["body", "missile"] = "body"

    # Reference values in SI units, cached on construction
    reference_area_m2: float = field(init=False, repr=False, compare=False)
    reference_length_m: float = field(init=False, repr=False, compare=False)
    reference_point_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Cache the reference values in SI units."""
        super().__post_init__()
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
        i = _cell_index(self.mach_axis, mach)
        j = _cell_index(self.alpha_axis, alpha)
        k = _cell_index(self.phi_axis, phi)
        t_mach = (mach - self.mach_axis[i]) / (
            self.mach_axis[i + 1] - self.mach_axis[i]
        )
        t_alpha = (alpha - self.alpha_axis[j]) / (
            self.alpha_axis[j + 1] - self.alpha_axis[j]
        )
        t_phi = (phi - self.phi_axis[k]) / (self.phi_axis[k + 1] - self.phi_axis[k])
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


def _cell_index(axis: np.ndarray, point: float) -> int:
    """Return the index of the grid cell whose lower edge is at or below a point.

    Points beyond either end of the axis map to the first or last cell.
    """
    index = int(np.searchsorted(axis, point, side="right")) - 1
    return min(max(index, 0), len(axis) - 2)


def aero_table_from_csv(
    filepath: str,
    *,
    reference_area: Scalar,
    reference_length: Scalar,
    reference_point: Vector,
    frame: Literal["body", "missile"] = "body",
) -> AeroTable:
    """Read a flight sim CSV and build one lookup covering every coefficient.

    Args:
        filepath (str): CSV with "Mach", "Alpha" and "Phi" columns plus one
            column per ``AeroCoefficients`` field, with components along the
            axes of ``frame``.
        reference_area (Scalar): Area the coefficients are normalized by.
        reference_length (Scalar): Length the moments are also normalized by.
        reference_point (Vector): Point the moments are taken about, from the
            nose tip in body axes.
        frame (Literal["body", "missile"], optional): Axes of the CSV's
            components. Defaults to "body".

    Returns:
        AeroTable: Lookup over the CSV's grid, carried on to 180 degrees
            of angle of attack (``flight_sim.aero_extend``) when the CSV
            stops short of it.
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
    # a table that stops short of 180 degrees (RASAero's stops at 30) is
    # carried on with the crossflow model, for the tumble after apogee
    alpha_axis, values = extend_values(mach_axis, alpha_axis, values)
    return AeroTable(
        mach_axis,
        alpha_axis,
        phi_axis,
        values,
        reference_area=reference_area,
        reference_length=reference_length,
        reference_point=reference_point,
        frame=frame,
    )


def eng_to_csv(motor_file_path: str) -> str:
    """Convert a RASP ".eng" thrust curve to a CSV, leaving other files alone.

    The file holds ";" comment lines, one description line (name, diameter,
    length, delays, propellant mass, total mass, maker), then "time thrust"
    pairs in seconds and newtons. RASP curves start from zero thrust at
    ignition, so a (0, 0) point is added when the first sample is later than
    zero. Only the first motor of a file is read.

    Args:
        motor_file_path (str): Path to a motor file. Only a ".eng" file is
            converted; any other path, such as a CSV, is returned as given.

    Returns:
        str: Path of the CSV to read. For a ".eng" file this is a CSV with the
            same name beside it, written on every call, holding "Time" and
            "Thrust" columns in seconds and newtons.

    Raises:
        ValueError: If the ".eng" file has no thrust samples, a sample line
            without a thrust, or sample times that do not increase.
    """
    path = Path(motor_file_path)
    if path.suffix.lower() != ".eng":
        return motor_file_path

    samples: list[tuple[float, float]] = []
    description_seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split(";", 1)[0].strip()
        if not text:
            continue
        if not description_seen:
            description_seen = True  # First line is the motor description
            continue
        values = text.split()
        try:
            time = float(values[0])
        except ValueError:
            break  # The description line of a second motor
        try:
            thrust = float(values[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Bad thrust sample {text!r} in {motor_file_path}"
            ) from error
        samples.append((time, thrust))
    if not samples:
        raise ValueError(f"No thrust samples found in {motor_file_path}")
    times = [time for time, _ in samples]
    if any(later <= earlier for earlier, later in pairwise(times)):
        raise ValueError(f"Sample times do not increase in {motor_file_path}")
    if times[0] > 0.0:
        samples.insert(0, (0.0, 0.0))

    csv_path = path.with_suffix(".csv")
    rows = "".join(f"{time},{thrust}\n" for time, thrust in samples)
    csv_path.write_text("Time,Thrust\n" + rows, encoding="utf-8")
    return str(csv_path)
