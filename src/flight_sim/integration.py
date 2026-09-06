"""Integration math for 6-DOF simulation"""

from dataclasses import dataclass, field
import numpy as np
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


def apply_derivative(
    state: RocketState, derivatives: StateDerivative, dt_scale: Scalar
) -> RocketState:
    """Helper to apply a scaled derivative to a state for RKF45 integration."""
    return RocketState(
        position=state.position + (derivatives.velocity * dt_scale),
        velocity=state.velocity + (derivatives.acceleration * dt_scale),
        angular_velocity=state.angular_velocity
        + (derivatives.angular_acceleration * dt_scale),
        orientation=state.orientation,
        current_mass=state.current_mass,
    )


# pylint: disable=too-many-locals
def rkf45_step(
    state: RocketState, atmosphere: AtmosphereData, dt: Scalar
) -> tuple[RocketState, RocketState]:
    """Advance the rocket state by one time step using RKF45 integration."""
    k1 = derivative_computation(state, atmosphere)

    dt_k2 = dt * 0.25
    state_k2 = apply_derivative(state, k1, dt_k2)
    k2 = derivative_computation(state_k2, atmosphere)

    dt_k3_1 = dt * (3 / 32)
    dt_k3_2 = dt * (9 / 32)
    state_k3_partial = apply_derivative(state, k1, dt_k3_1)
    state_k3 = apply_derivative(state_k3_partial, k2, dt_k3_2)
    k3 = derivative_computation(state_k3, atmosphere)

    dt_k4_1 = dt * (1932 / 2197)
    dt_k4_2 = dt * (-7200 / 2197)
    dt_k4_3 = dt * (7296 / 2197)
    state_k4_partial1 = apply_derivative(state, k1, dt_k4_1)
    state_k4_partial2 = apply_derivative(state_k4_partial1, k2, dt_k4_2)
    state_k4 = apply_derivative(state_k4_partial2, k3, dt_k4_3)
    k4 = derivative_computation(state_k4, atmosphere)

    dt_k5_1 = dt * (439 / 216)
    dt_k5_2 = dt * (-8)
    dt_k5_3 = dt * (3680 / 513)
    dt_k5_4 = dt * (-845 / 4104)
    state_k5_partial1 = apply_derivative(state, k1, dt_k5_1)
    state_k5_partial2 = apply_derivative(state_k5_partial1, k2, dt_k5_2)
    state_k5_partial3 = apply_derivative(state_k5_partial2, k3, dt_k5_3)
    state_k5 = apply_derivative(state_k5_partial3, k4, dt_k5_4)
    k5 = derivative_computation(state_k5, atmosphere)

    dt_k6_1 = dt * (-8 / 27)
    dt_k6_2 = dt * 2
    dt_k6_3 = dt * (-3544 / 2565)
    dt_k6_4 = dt * (1859 / 4104)
    dt_k6_5 = dt * (-11 / 40)
    state_k6_partial1 = apply_derivative(state, k1, dt_k6_1)
    state_k6_partial2 = apply_derivative(state_k6_partial1, k2, dt_k6_2)
    state_k6_partial3 = apply_derivative(state_k6_partial2, k3, dt_k6_3)
    state_k6_partial4 = apply_derivative(state_k6_partial3, k4, dt_k6_4)
    state_k6 = apply_derivative(state_k6_partial4, k5, dt_k6_5)
    k6 = derivative_computation(state_k6, atmosphere)

    state_5th = apply_derivative(state, k1, dt * (16 / 135))
    state_5th = apply_derivative(state_5th, k3, dt * (6656 / 12825))
    state_5th = apply_derivative(state_5th, k4, dt * (28561 / 56430))
    state_5th = apply_derivative(state_5th, k5, dt * (-9 / 50))
    state_5th = apply_derivative(state_5th, k6, dt * (2 / 55))

    state_4th = apply_derivative(state, k1, dt * (25 / 216))
    state_4th = apply_derivative(state_4th, k3, dt * (1408 / 2565))
    state_4th = apply_derivative(state_4th, k4, dt * (2197 / 4104))
    state_4th = apply_derivative(state_4th, k5, dt * (-1 / 5))

    return state_5th, state_4th


def step(state: RocketState, atmosphere: AtmosphereData, dt: Scalar) -> RocketState:
    """Advance the rocket state using adaptive RKF45 integration.

    Args:
        state (RocketState): Current state of the vehicle.
        atmosphere (AtmosphereData): Conditions at the vehicle's altitude.
        dt (Scalar): Length of the step, in any unit of time.

    Returns:
        RocketState: The state after advancing by one step.
    """
    # Strip units for while-loop
    target_time: float = dt.to("s").magnitude
    time_simulated: float = 0.0

    # Assume while step taken at once
    current_dt: float = target_time
    current_state = state

    # Current tolerance set (Can be adjusted if needed)
    tolerance = 1e-6

    while time_simulated < target_time:
        # Prevents the final step from overshooting the target time
        if current_dt > (target_time - time_simulated):
            current_dt = target_time - time_simulated

        # Wraps the raw float back into Pint Scalar for the RKF45 step
        dt_scalar = scalar(current_dt, "s")

        # Look into future with 4th and 5th order RKF45 steps
        state_5th, state_4th = rkf45_step(current_state, atmosphere, dt_scalar)

        # Calculate difference between two position estimates in meters
        position_difference = (
            (state_5th.position - state_4th.position).to("m").magnitude
        )
        error = float(np.linalg.norm(position_difference))

        # Check if the error is within the tolerance
        if error <= tolerance:
            current_state = state_5th
            time_simulated += current_dt

            if error > 0:
                scale_factor = 0.9 * (tolerance / error) ** 0.2
                current_dt *= min(scale_factor, 1.5)
            else:
                current_dt *= 1.5
        else:
            scale_factor = 0.9 * (tolerance / error) ** 0.2
            current_dt *= max(scale_factor, 0.1)
    return current_state

    # Route into your robust RKF45 mathematics
    return rkf45_step(state, atmosphere, time_step)
