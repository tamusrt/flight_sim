"""Functions for loading aerodynamic CSV data."""

import numpy as np
import pandas as pd  # type: ignore # pylint: disable=import-error
from scipy.interpolate import RegularGridInterpolator  # type: ignore


def interpolator_from_csv(filepath: str, output_col: str) -> RegularGridInterpolator:
    """Reads a flight sim CSV and builds a multi-variable interpolator"""
    data_frame = pd.read_csv(filepath)

    mach_axis = np.sort(data_frame["Mach"].unique())
    alpha_axis = np.sort(data_frame["Alpha"].unique())

    grid_matrix = data_frame.pivot_table(
        values=output_col, index="Mach", columns="Alpha"
    ).to_numpy()

    return RegularGridInterpolator((mach_axis, alpha_axis), grid_matrix)
