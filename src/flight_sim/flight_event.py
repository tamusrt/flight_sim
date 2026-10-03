"""Moments of the flight the integrator can land a step on."""

from collections.abc import Callable
from dataclasses import dataclass

from flight_sim.vehicle.rocket_state import RocketState


@dataclass(frozen=True)
class FlightEvent:
    """A moment of the flight the integrator can land a step on exactly.

    Attributes:
        name (str): Name of the event, such as "apogee".
        value (Callable[[float, RocketState], float]): Quantity, given the time
            in seconds and the state, that is positive before the event and
            zero or negative at and after it.
    """

    name: str
    value: Callable[[float, RocketState], float]

    def crossed(
        self,
        start_time: float,
        start: RocketState,
        end_time: float,
        end: RocketState,
    ) -> bool:
        """Return whether the event lies within a step.

        Args:
            start_time (float): Time at the start of the step, in seconds.
            start (RocketState): State at the start of the step.
            end_time (float): Time at the end of the step, in seconds.
            end (RocketState): State at the end of the step.

        Returns:
            bool: True if the value is positive at the start and zero or
                negative at the end.
        """
        return self.value(start_time, start) > 0.0 >= self.value(end_time, end)
