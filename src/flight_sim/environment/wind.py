"""Wind models giving the velocity of the air at an altitude."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from flight_sim.units import Scalar, UnitChecked, scalar
from flight_sim.utilities.dcm import world_direction


class WindModel(ABC):
    """Wind the integrator samples at each altitude it visits."""

    @abstractmethod
    def velocity(self, altitude_m: float) -> np.ndarray:
        """Return the wind velocity at an altitude.

        Args:
            altitude_m (float): Altitude above sea level in metres.

        Returns:
            np.ndarray: Velocity of the air in the world frame in m/s.
        """


@dataclass
class UniformWind(WindModel, UnitChecked):
    """Horizontal wind of one speed and direction at every altitude."""

    speed: Annotated[Scalar, "m/s"] = field(default_factory=lambda: scalar(0.0, "m/s"))

    # Azimuth the wind blows from, as in weather reports
    from_azimuth: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(0.0, "deg")
    )

    _velocity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._velocity = -float(self.speed.m_as("m/s")) * world_direction(
            0.0, float(self.from_azimuth.m_as("rad"))
        )

    def velocity(self, altitude_m: float) -> np.ndarray:
        """Return the wind velocity in the world frame in m/s at any altitude."""
        return self._velocity
