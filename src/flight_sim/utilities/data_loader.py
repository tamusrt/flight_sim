"""Functions for loading aerodynamic CSV data."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd  # type: ignore


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


class AeroTable:
    """Trilinear lookup of body-frame coefficients over Mach, alpha_tot and phi_a.

    The angles are defined in ``flight_sim.utilities.dcm`` and given in
    degrees; phi is wrapped into [0, 360). Outside the grid, each coefficient
    is extrapolated linearly from the nearest edge cell.
    """

    def __init__(
        self,
        mach_axis: np.ndarray,
        alpha_axis: np.ndarray,
        phi_axis: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Store the grid axes and the coefficient values on that grid.

        Args:
            mach_axis (np.ndarray): Strictly increasing, at least two values.
            alpha_axis (np.ndarray): Degrees, strictly increasing, at least two.
            phi_axis (np.ndarray): Degrees, strictly increasing, at least two.
            values (np.ndarray): Shape (len(mach_axis), len(alpha_axis),
                len(phi_axis), 6), last axis in ``AeroCoefficients`` field order.
        """
        self.mach_axis = mach_axis
        self.alpha_axis = alpha_axis
        self.phi_axis = phi_axis
        self.values = values

    def __call__(self, mach: float, alpha: float, phi: float) -> AeroCoefficients:
        """Interpolate every coefficient at one flight condition.

        Args:
            mach (float): Mach number.
            alpha (float): Total angle of attack in degrees.
            phi (float): Aerodynamic roll angle in degrees.

        Returns:
            AeroCoefficients: The interpolated coefficients.
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
        return AeroCoefficients(
            *(at_alpha[0] + t_mach * (at_alpha[1] - at_alpha[0])).tolist()
        )


def _cell_index(axis: np.ndarray, point: float) -> int:
    """Return the index of the grid cell whose lower edge is at or below a point.

    Points beyond either end of the axis map to the first or last cell.
    """
    index = int(np.searchsorted(axis, point, side="right")) - 1
    return min(max(index, 0), len(axis) - 2)


def aero_table_from_csv(filepath: str) -> AeroTable:
    """Read a flight sim CSV and build one lookup covering every coefficient.

    Args:
        filepath (str): CSV with "Mach", "Alpha" and "Phi" columns plus one
            column per ``AeroCoefficients`` field.

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
    return AeroTable(mach_axis, alpha_axis, phi_axis, values)


def time_interpolator_from_csv(
    filepath: str, time_col: str, output_col: str
) -> Callable[[float], float]:
    """Read a CSV and build a 1D time-based interpolator for motor curves.

    Args:
        filepath (str): CSV holding the time and output columns.
        time_col (str): Column of increasing sample times, in seconds.
        output_col (str): Column to interpolate.

    Returns:
        Callable[[float], float]: Linear interpolation of the output column,
            returning 0.0 outside the sampled time range.
    """
    data_frame = pd.read_csv(filepath)
    times = data_frame[time_col].to_numpy(dtype=float)
    outputs = data_frame[output_col].to_numpy(dtype=float)

    def interpolate(time: float) -> float:
        """Return the output column's value at the given time in seconds."""
        return float(np.interp(time, times, outputs, left=0.0, right=0.0))

    return interpolate
