"""Functions for loading aerodynamic CSV data."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd  # type: ignore # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class AeroTable:
    """Bilinear lookup of several coefficients over a shared Mach/alpha grid.

    Outside the grid, each coefficient is extrapolated linearly from the
    nearest edge cell.
    """

    def __init__(
        self, mach_axis: np.ndarray, alpha_axis: np.ndarray, values: np.ndarray
    ) -> None:
        """Store the grid axes and the coefficient values on that grid.

        Args:
            mach_axis (np.ndarray): Strictly increasing Mach numbers, at least two.
            alpha_axis (np.ndarray): Strictly increasing angles of attack in
                degrees, at least two.
            values (np.ndarray): Coefficients with shape
                (len(mach_axis), len(alpha_axis), n_coefficients).
        """
        self.mach_axis = mach_axis
        self.alpha_axis = alpha_axis
        self.values = values

    def __call__(self, mach: float, alpha: float) -> list[float]:
        """Interpolate every coefficient at one flight condition.

        Args:
            mach (float): Mach number.
            alpha (float): Angle of attack in degrees.

        Returns:
            list[float]: One value per coefficient, in the table's column order.
        """
        i = _cell_index(self.mach_axis, mach)
        j = _cell_index(self.alpha_axis, alpha)
        t_mach = (mach - self.mach_axis[i]) / (
            self.mach_axis[i + 1] - self.mach_axis[i]
        )
        t_alpha = (alpha - self.alpha_axis[j]) / (
            self.alpha_axis[j + 1] - self.alpha_axis[j]
        )
        cell = self.values[i : i + 2, j : j + 2]
        low_mach = cell[0, 0] + t_alpha * (cell[0, 1] - cell[0, 0])
        high_mach = cell[1, 0] + t_alpha * (cell[1, 1] - cell[1, 0])
        result: list[float] = (low_mach + t_mach * (high_mach - low_mach)).tolist()
        return result


def _cell_index(axis: np.ndarray, point: float) -> int:
    """Return the index of the grid cell whose lower edge is at or below a point.

    Points beyond either end of the axis map to the first or last cell.
    """
    index = int(np.searchsorted(axis, point, side="right")) - 1
    return min(max(index, 0), len(axis) - 2)


def aero_table_from_csv(filepath: str, output_cols: Sequence[str]) -> AeroTable:
    """Read a flight sim CSV and build one lookup covering several coefficients.

    Args:
        filepath (str): CSV with "Mach" and "Alpha" columns plus one column per
            coefficient.
        output_cols (Sequence[str]): Coefficient columns to include, in the
            order the lookup returns them.

    Returns:
        AeroTable: Lookup over the CSV's Mach/alpha grid.
    """
    data_frame = pd.read_csv(filepath)

    mach_axis = np.sort(data_frame["Mach"].unique())
    alpha_axis = np.sort(data_frame["Alpha"].unique())

    values = np.stack(
        [
            data_frame.pivot_table(
                values=column, index="Mach", columns="Alpha"
            ).to_numpy()
            for column in output_cols
        ],
        axis=-1,
    )
    return AeroTable(mach_axis, alpha_axis, values)


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
