"""Integration math for 6-DOF simulation"""

from dataclasses import dataclass, field

from flight_sim.environment.atmosphere import AtmosphereData
from flight_sim.environment.gravity import get_gravity
from flight_sim.units import Scalar, UnitChecked, Vector, scalar, vector, zero_vector
from flight_sim.vehicle.rocket_state import RocketState


@dataclass
class IntegrationConfiguration(UnitChecked):
    """Setting for the integration of the rocket's state over time."""

    time_step: Scalar = field(default_factory=lambda: scalar(0.01, "s"))
    max_time: Scalar = field(default_factory=lambda: scalar(100.0, "s"))
    initial_state: RocketState = field(default_factory=RocketState)
    atmosphere_data: AtmosphereData = field(default_factory=AtmosphereData)


@dataclass
class StateDerivative(UnitChecked):
    """Rate of change of a RocketState with respect to time."""

    # d(position)/dt
    velocity: Vector = field(default_factory=lambda: zero_vector("m/s"))

    # d(velocity)/dt
    acceleration: Vector = field(default_factory=lambda: zero_vector("m/s**2"))

    # d(angular_velocity)/dt
    angular_acceleration: Vector = field(
        default_factory=lambda: zero_vector("rad/s**2")
    )


def derivative_computation(
    state: RocketState, atmosphere: AtmosphereData
) -> StateDerivative:
    """Compute the time derivative of the rocket state.

    Args:
        state (RocketState): Current state of the vehicle.
        atmosphere (AtmosphereData): Conditions at the vehicle's altitude.

    Returns:
        StateDerivative: Rates of change to integrate over the next step.
    """
    # pylint: disable=unused-argument
    magnitude_of_gravity: Scalar = get_gravity(
        latitude=scalar(0.0, "deg"),
        longitude=scalar(0.0, "deg"),
        altitude=state.position[2],
    )
    # Gravity acts along -Z. The unit vector is dimensionless, so the product
    # keeps whatever acceleration units get_gravity returned.
    down: Vector = vector((0.0, 0.0, -1.0), "dimensionless")
    return StateDerivative(
        velocity=state.velocity,
        acceleration=magnitude_of_gravity * down,
        angular_acceleration=zero_vector("rad/s**2"),
    )


def step(state: RocketState, atmosphere: AtmosphereData, dt: Scalar) -> RocketState:
    """Advance the rocket state by one time step.

    Args:
        state (RocketState): Current state of the vehicle.
        atmosphere (AtmosphereData): Conditions at the vehicle's altitude.
        dt (Scalar): Length of the step, in any unit of time.

    Returns:
        RocketState: The state after advancing by one step.

    Raises:
        DimensionalityError: If dt is not a time.
    """
    time_step: Scalar = dt.to("s")
    derivatives = derivative_computation(state, atmosphere)

    return RocketState(
        position=state.position + derivatives.velocity * time_step,
        velocity=state.velocity + derivatives.acceleration * time_step,
        angular_velocity=(
            state.angular_velocity + derivatives.angular_acceleration * time_step
        ),
        orientation=state.orientation,
        current_mass=state.current_mass,
    )
