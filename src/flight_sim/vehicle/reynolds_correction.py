"""Correction of tabulated drag for the Reynolds number of the flight."""

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from flight_sim.environment.atmosphere import (
    AtmosphereConditions,
    StandardAtmosphere1976,
    dynamic_viscosity,
)

# Reynolds number per unit Mach and length scales with density * a / mu
_SEA_LEVEL = StandardAtmosphere1976().conditions(0.0)
_SEA_LEVEL_RHO_A_OVER_MU = (
    _SEA_LEVEL.air_density
    * _SEA_LEVEL.speed_of_sound
    / dynamic_viscosity(_SEA_LEVEL.temperature)
)


@dataclass(frozen=True)
class ReynoldsCorrection:
    """Change in Cx moving an aero table's viscous drag from sea-level air to
    the flight's.

    The table is for sea-level air, so the flight's Reynolds number matches
    the table's at an equivalent Mach number. The correction is the change in
    viscous Cx between the two, read along the rows up to the roughness
    limit, where skin friction stops falling.
    """

    mach_axis: np.ndarray  # Strictly increasing

    # Reynolds-dependent part of the zero-alpha Cx at each Mach, at sea-level
    # air as RASAero gives it, NaN where unknown
    cx_viscous: np.ndarray

    def __post_init__(self) -> None:
        """Check the viscous Cx matches the Mach axis.

        Raises:
            ValueError: If there is not one viscous Cx per Mach.
        """
        if self.cx_viscous.shape != self.mach_axis.shape:
            raise ValueError("ReynoldsCorrection needs one CxViscous per Mach")

    @classmethod
    def from_csv(cls, filepath: str | Path) -> "ReynoldsCorrection":
        """Read a CSV with "Mach" and "CxViscous" columns, blank where unknown.

        An aero table works as is: only its alpha 0, phi 0 rows are used.
        """
        data_frame = pd.read_csv(filepath)
        if {"Alpha", "Phi"} <= set(data_frame.columns):
            data_frame = data_frame[(data_frame["Alpha"] == 0) & (data_frame["Phi"] == 0)]
        data_frame = data_frame.sort_values("Mach")
        return cls(
            data_frame["Mach"].to_numpy(dtype=float),
            data_frame["CxViscous"].to_numpy(dtype=float),
        )

    @cached_property
    def _rows(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Log Mach and viscous Cx of the rows up to the roughness limit."""
        known = (self.mach_axis > 0.0) & ~np.isnan(self.cx_viscous)
        machs, viscous = self.mach_axis[known], self.cx_viscous[known]
        if viscous.size < 2:
            return None
        # Rows end where the drag stops falling: the roughness limit
        rising = np.flatnonzero(np.diff(viscous) >= 0.0)
        end = rising[0] + 1 if rising.size else viscous.size
        return np.log(machs[:end]), viscous[:end]

    def __call__(self, mach: float, air: AtmosphereConditions) -> float:
        """Return the change in Cx at a Mach number in the flight's air.

        Args:
            mach (float): Mach number.
            air (AtmosphereConditions): Air the vehicle flies through.

        Returns:
            float: Change in Cx, zero without enough rows or air.
        """
        if self._rows is None or air.air_density <= 0.0 or mach <= 0.0:
            return 0.0
        # Same Mach and length, so Reynolds number scales with density * a / mu
        reynolds_ratio = (
            air.air_density * air.speed_of_sound / dynamic_viscosity(air.temperature)
        ) / _SEA_LEVEL_RHO_A_OVER_MU
        log_machs, viscous = self._rows
        drag = np.interp(
            [math.log(mach * reynolds_ratio), math.log(mach)], log_machs, viscous
        )
        return -float(drag[0] - drag[1])
