"""Standard flight events the integrator can land a step on."""

from flight_sim.flight_event import FlightEvent
from flight_sim.integration import IntegrationConfiguration, derivative_computation
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
