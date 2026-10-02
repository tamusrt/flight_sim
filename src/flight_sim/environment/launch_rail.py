"""Launch rail the rocket slides along before it flies free."""

import math
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from flight_sim.units import Scalar, UnitChecked, scalar
from flight_sim.utilities.dcm import world_direction
from flight_sim.utilities.quaternion import Quaternion


@dataclass
class LaunchRail(UnitChecked):
    """Straight rail running from the pad, at an elevation and azimuth."""

    length: Annotated[Scalar, "m"]

    elevation: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(90.0, "deg")
    )

    azimuth: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(0.0, "deg")
    )

    def direction(self) -> np.ndarray:
        """Return the world-frame unit vector pointing up the rail."""
        return world_direction(
            float(self.elevation.m_as("rad")), float(self.azimuth.m_as("rad"))
        )

    def orientation(self) -> Quaternion:
        """Return the attitude of a rocket on the rail.

        The rocket is tipped from vertical straight toward the azimuth, so the
        nose points up the rail with no roll about it.

        Returns:
            Quaternion: Body-to-world rotation.
        """
        half_tilt = (math.pi / 2 - float(self.elevation.m_as("rad"))) / 2
        azimuth = float(self.azimuth.m_as("rad"))
        # The tilt axis is horizontal, perpendicular to the azimuth
        return Quaternion(
            q_w=math.cos(half_tilt),
            q_y=-math.cos(azimuth) * math.sin(half_tilt),
            q_z=math.sin(azimuth) * math.sin(half_tilt),
        )
