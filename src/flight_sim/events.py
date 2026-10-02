"""Flight events the integrator can land a step on, and the standard ones."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flight_sim.integration import IntegrationConfiguration, derivative_computation
from flight_sim.vehicle.rocket_properties import RocketProperties
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


# Vertical velocity falls through zero
APOGEE = FlightEvent(
    "apogee", lambda _time, state: float(state.velocity.m_as("m/s")[0])
)

# Altitude falls through zero
IMPACT = FlightEvent("impact", lambda _time, state: float(state.position.m_as("m")[0]))


def rail_exit(config: IntegrationConfiguration) -> FlightEvent:
    """Build the event at which the rocket is a rail length from the pad.

    Args:
        config (IntegrationConfiguration): Configuration giving the rail length.

    Returns:
        FlightEvent: The event, named "rail exit".
    """
    rail_length = float(config.rail_length.m_as("m"))
    return FlightEvent(
        "rail exit",
        lambda _time, state: (
            rail_length - float(np.linalg.norm(state.position.m_as("m")))
        ),
    )


def peak_vertical_velocity(
    properties: RocketProperties, config: IntegrationConfiguration
) -> FlightEvent:
    """Build the event at which the vertical acceleration falls through zero.

    Args:
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Environment models.

    Returns:
        FlightEvent: The event, named "peak vertical velocity".
    """

    def vertical_acceleration(time: float, state: RocketState) -> float:
        """Return the vertical acceleration in m/s**2."""
        derivative = derivative_computation(time, state, properties, config)
        return float(derivative.acceleration.m_as("m/s**2")[0])

    return FlightEvent("peak vertical velocity", vertical_acceleration)
