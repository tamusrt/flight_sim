"""Standard flight events the integrator can land a step on."""

from typing import Annotated

from flight_sim.flight_event import FlightEvent
from flight_sim.integration import IntegrationConfiguration, derivative_computation
from flight_sim.units import Scalar, scalar
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

# Vertical velocity falls through zero
APOGEE = FlightEvent(
    "apogee", lambda _time, state: float(state.velocity.m_as("m/s")[0])
)

# Altitude falls through zero
IMPACT = FlightEvent("impact", lambda _time, state: float(state.position.m_as("m")[0]))


def peak_vertical_velocity(
    properties: RocketProperties, config: IntegrationConfiguration
) -> FlightEvent:
    """Build the event at which the vertical acceleration falls through zero.

    Args:
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Truth models.

    Returns:
        FlightEvent: The event, named "peak vertical velocity".
    """

    def vertical_acceleration(time: float, state: RocketState) -> float:
        """Return the vertical acceleration in m/s**2."""
        derivative = derivative_computation(time, state, properties, config)
        return float(derivative.acceleration.m_as("m/s**2")[0])

    return FlightEvent("peak vertical velocity", vertical_acceleration)


def burnout(
    properties: RocketProperties,
    name: str = "burnout",
    delay: Annotated[Scalar, "s"] = scalar(0.0, "s"),
) -> FlightEvent:
    """Build the event ``delay`` after the motor's thrust curve ends.

    The motor is read at each call, so a motor lit by an event during the
    flight is judged as lit.
    """
    delay_s = float(delay.m_as("s"))
    return FlightEvent(
        name,
        lambda time, _state: properties.engine.burnout_time() + delay_s - time,
    )


def at_time(name: str, time: Scalar) -> FlightEvent:
    """Build the event at a fixed simulation time."""
    time_s = float(time.m_as("s"))
    return FlightEvent(name, lambda current_time, _state: time_s - current_time)
