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


# The layers of LayeredWind: (wavelength in metres of height, share of the base
# speed, phase in radians) for the speed, and (wavelength, degrees, phase) for
# the direction. The wavelengths do not divide each other, so the pattern does
# not repeat within a flight; the phases are fixed, so every run is the same.
SPEED_LAYERS: tuple[tuple[float, float, float], ...] = (
    (1900.0, 0.15, 0.4),
    (770.0, 0.10, 2.1),
    (310.0, 0.06, 4.0),
)
DIRECTION_LAYERS: tuple[tuple[float, float, float], ...] = (
    (2600.0, 12.0, 1.3),
    (1100.0, 6.0, 5.2),
    (450.0, 3.0, 0.7),
)


@dataclass
class LayeredWind(WindModel, UnitChecked):
    """A wind that changes with height in a fixed, repeatable pattern.

    The base speed and direction are the day's average wind. Above the pad the
    speed and the direction each wander with height as a sum of a few sine
    waves (``SPEED_LAYERS`` and ``DIRECTION_LAYERS``):

        speed(h)     = base * (1 + r(h) * sum a_i sin(2 pi h / L_i + p_i))
        direction(h) = base + r(h) * sum b_i sin(2 pi h / M_i + q_i)

    ``r(h) = 1 - exp(-h / ramp)`` fades the layers in over the first tens of
    metres, so the wind on the launch rail is exactly the base wind. Nothing is
    random: the same simulation always flies through the same wind, but the
    wind is uneven through the flight, so a descending canopy keeps meeting
    new wind and swings.
    """

    speed: Annotated[Scalar, "m/s"] = field(default_factory=lambda: scalar(0.0, "m/s"))
    from_azimuth: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(0.0, "deg")
    )
    # Height of the pad above sea level; the layers are counted from it
    ground_m: float = 0.0
    ramp_m: float = 30.0
    speed_layers: tuple[tuple[float, float, float], ...] = SPEED_LAYERS
    direction_layers: tuple[tuple[float, float, float], ...] = DIRECTION_LAYERS

    def speed_and_azimuth(self, height_m: float) -> tuple[float, float]:
        """Speed in m/s and from-azimuth in radians at a height above the pad."""
        h = max(height_m, 0.0)
        ramp = 1.0 - np.exp(-h / self.ramp_m) if self.ramp_m > 0 else 1.0
        wave = sum(
            a * np.sin(2 * np.pi * h / length + p) for length, a, p in self.speed_layers
        )
        turn = sum(
            b * np.sin(2 * np.pi * h / length + p)
            for length, b, p in self.direction_layers
        )
        speed = float(self.speed.m_as("m/s")) * max(1.0 + ramp * wave, 0.0)
        azimuth = float(self.from_azimuth.m_as("rad")) + np.radians(ramp * turn)
        return speed, float(azimuth)

    def velocity(self, altitude_m: float) -> np.ndarray:
        """Return the wind velocity in the world frame in m/s at an altitude."""
        speed, azimuth = self.speed_and_azimuth(altitude_m - self.ground_m)
        return -speed * world_direction(0.0, azimuth)
