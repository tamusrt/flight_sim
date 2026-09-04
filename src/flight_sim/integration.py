"""Integration math for 6-DOF simulation"""

from dataclasses import dataclass, field

import numpy as np

from flight_sim.environment.atmosphere import AtmosphereData
from flight_sim.environment.gravity import get_gravity
from flight_sim.vehicle.rocket_state import RocketState


@dataclass
class IntegrationConfiguration:
    """Setting for the integration of the rocket's state over time."""

    time_step: float = 0.01  # Time step for integration (seconds)
    max_time: float = 100.0  # Maximum simulation time (seconds)
    initial_state: RocketState = field(default_factory=RocketState)
    atmosphere_data: AtmosphereData = field(default_factory=AtmosphereData)


def derivative_computation(
    state: RocketState, atmosphere: AtmosphereData
) -> RocketState:
    """Caculates change of rocket during time step"""
    # pylint: disable=unused-argument
    magnitude_of_gravity = get_gravity(
        latitude=0.0, longitude=0.0, altitude=state.position[2]
    )
    gravity_acceleration = np.array([0.0, 0.0, -magnitude_of_gravity])
    return RocketState(
        position=state.velocity,
        velocity=gravity_acceleration,
        angular_velocity=np.zeros(3),
        orientation=state.orientation,
        current_mass=state.current_mass,
    )


def step(state: RocketState, atmosphere: AtmosphereData, dt: float) -> RocketState:
    """Advances the rocket state by one time step"""
    derivatives = derivative_computation(state, atmosphere)

    return RocketState(
        position=state.position + derivatives.position * dt,
        velocity=state.velocity + derivatives.velocity * dt,
        angular_velocity=state.angular_velocity + derivatives.angular_velocity * dt,
        orientation=state.orientation,
        current_mass=state.current_mass,
    )
