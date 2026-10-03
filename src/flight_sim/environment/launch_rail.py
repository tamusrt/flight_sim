"""Launch rail the rocket slides along before it flies free."""

import math
from dataclasses import dataclass, field, replace
from typing import Annotated

import numpy as np

from flight_sim.flight_event import FlightEvent
from flight_sim.units import Scalar, UnitChecked, scalar, vector
from flight_sim.utilities.dcm import world_direction
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_state import RocketState


@dataclass(frozen=True)
class LaunchRail(UnitChecked):
    """Straight rail at an elevation and azimuth.

    The rail is not fixed in the world: its foot is wherever the rocket was
    when ``mount`` put it on the rail.
    """

    length: Annotated[Scalar, "m"]

    elevation: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(90.0, "deg")
    )

    azimuth: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(0.0, "deg")
    )

    # Coulomb friction between the rail and the rocket, kinetic and static
    friction_coefficient: float = 0.0

    # Event at which the rocket has slid the rail's length and leaves it
    exit_event: FlightEvent = field(init=False, repr=False, compare=False)

    _length_m: float = field(init=False, repr=False, compare=False)
    _direction: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Cache the rail's geometry in SI units."""
        super().__post_init__()
        direction = world_direction(
            float(self.elevation.m_as("rad")), float(self.azimuth.m_as("rad"))
        )
        object.__setattr__(self, "_length_m", float(self.length.m_as("m")))
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(
            self, "exit_event", FlightEvent("rail exit", self._distance_to_exit)
        )

    def direction(self) -> np.ndarray:
        """Return the world-frame unit vector pointing up the rail."""
        return self._direction

    def distance_along(self, position_m: np.ndarray, start_m: np.ndarray) -> float:
        """Return how far up the rail a position lies from the rail's foot.

        Args:
            position_m (np.ndarray): World-frame position in m.
            start_m (np.ndarray): World-frame position of the rail's foot in m.

        Returns:
            float: Distance along the rail in m, negative below the foot.
        """
        return float((position_m - start_m) @ self._direction)

    def mount(self, state: RocketState) -> RocketState:
        """Return a copy of a state put on the rail, with its foot at the rocket."""
        return replace(
            state,
            orientation=self.orientation(),
            rail_start=vector(state.position.m_as("m"), "m"),
        )

    def _distance_to_exit(self, _time: float, state: RocketState) -> float:
        """Return the distance in m the rocket has left to slide up the rail."""
        assert state.rail_start is not None, "Only a rocket on the rail can exit it"
        return self._length_m - self.distance_along(
            state.position.m_as("m"), state.rail_start.m_as("m")
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
