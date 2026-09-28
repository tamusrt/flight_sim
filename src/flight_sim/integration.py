"""Integration math for 6-DOF simulation.

Public functions take and return quantities. Private functions take and
return arrays in SI units.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, NamedTuple

import numpy as np

from flight_sim.environment.atmosphere import (
    AtmosphereConditions,
    AtmosphereModel,
    StandardAtmosphere1976,
)
from flight_sim.environment.gravity import GravityModel, WGS84Gravity
from flight_sim.units import Scalar, UnitChecked, Vector, scalar, vector
from flight_sim.utilities.data_loader import AeroCoefficients
from flight_sim.utilities.quaternion import (
    Quaternion,
    cross,
    quaternion_rates,
    rotation_matrix,
)
from flight_sim.vehicle.engine import Engine
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

# Layout of the state array: metres, m/s, rad/s, quaternion, kilograms
_POSITION = slice(0, 3)
_VELOCITY = slice(3, 6)
_ANGULAR_VELOCITY = slice(6, 9)
_ORIENTATION = slice(9, 13)  # q_w, q_x, q_y, q_z
_MASS = 13
_STATE_SIZE = 14

# Fehlberg RKF45 tableau: stage time fractions, stage weights, and the weights
# of the 5th and 4th order solutions.
_RKF_NODES = (0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2)
_RKF_STAGE_WEIGHTS = (
    np.array([]),
    np.array([1 / 4]),
    np.array([3 / 32, 9 / 32]),
    np.array([1932 / 2197, -7200 / 2197, 7296 / 2197]),
    np.array([439 / 216, -8.0, 3680 / 513, -845 / 4104]),
    np.array([-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40]),
)
_RKF_5TH_ORDER = np.array([16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
_RKF_4TH_ORDER = np.array([25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0])


@dataclass
class IntegrationConfiguration(UnitChecked):
    """Environment models and settings the integrator holds fixed for a flight."""

    atmosphere: AtmosphereModel = field(default_factory=StandardAtmosphere1976)

    gravity: GravityModel = field(default_factory=WGS84Gravity)

    launch_latitude: Annotated[Scalar, "rad"] = field(
        default_factory=lambda: scalar(0.0, "deg")
    )

    # Largest position disagreement between the 4th and 5th order RKF45
    # estimates that a step may have and still be accepted
    position_tolerance: Annotated[Scalar, "m"] = field(
        default_factory=lambda: scalar(1e-7, "m")
    )

    # Shortest step the integrator takes. A step this short is accepted
    # whatever its error, with a warning.
    min_time_step: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(1e-9, "s")
    )

    # Width of the time bracket within which locate_event finds an event
    event_time_tolerance: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(1e-9, "s")
    )

    # Most bracket refinements locate_event makes before settling on the
    # bracket's later end
    max_event_iterations: int = 100


@dataclass
class StateDerivative(UnitChecked):
    """Rate of change of a RocketState with respect to time."""

    # d(position)/dt
    velocity: Annotated[Vector, "m/s"]

    # d(velocity)/dt
    acceleration: Annotated[Vector, "m/s**2"]

    # d(angular_velocity)/dt
    angular_acceleration: Annotated[Vector, "rad/s**2"]

    # d(orientation)/dt
    orientation_derivative: Quaternion

    mass_derivative: Annotated[Scalar, "kg/s"]


class _StepInputs(NamedTuple):
    """Constants of one step, in SI units."""

    inertia: np.ndarray  # kg*m**2, body axes
    lever_arm_body: np.ndarray  # m, aero reference point relative to the CG
    reference_area: float  # m**2
    reference_diameter: float  # m
    properties: RocketProperties
    atmosphere: Callable[[float], AtmosphereConditions]  # From the altitude in m
    gravity: Callable[[float, float], float]  # From the latitude in rad, altitude in m
    latitude_rad: float
    tolerance_m: float
    min_dt_s: float


def _step_inputs(
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
) -> _StepInputs:
    """Collect the constants of one step from the state, properties and config."""
    return _StepInputs(
        inertia=state.inertia.m_as("kg*m**2"),
        lever_arm_body=properties.reference_point.m_as("m")
        - state.cg_location.m_as("m"),
        reference_area=float(properties.reference_area.m_as("m**2")),
        reference_diameter=float(properties.reference_diameter.m_as("m")),
        properties=properties,
        atmosphere=config.atmosphere.conditions,
        gravity=config.gravity.magnitude,
        latitude_rad=float(config.launch_latitude.m_as("rad")),
        tolerance_m=float(config.position_tolerance.m_as("m")),
        min_dt_s=float(config.min_time_step.m_as("s")),
    )


def _pack(state: RocketState) -> np.ndarray:
    """Return the integrated fields of a state as an SI array."""
    orientation = state.orientation
    return np.concatenate(
        (
            state.position.m_as("m"),
            state.velocity.m_as("m/s"),
            state.angular_velocity.m_as("rad/s"),
            (
                orientation.q_w,
                orientation.q_x,
                orientation.q_y,
                orientation.q_z,
                float(state.current_mass.m_as("kg")),
            ),
        )
    )


def _unpack(values: np.ndarray, template: RocketState) -> RocketState:
    """Rebuild a RocketState from an SI array.

    Args:
        values (np.ndarray): Integrated fields in the layout ``_pack`` produces.
        template (RocketState): State supplying the fields the integrator
            holds constant: inertia, CG location, and reference point.

    Returns:
        RocketState: The integrated fields combined with the template's
            constant ones.
    """
    q_w, q_x, q_y, q_z = values[_ORIENTATION].tolist()
    return RocketState(
        position=vector(values[_POSITION], "m"),
        velocity=vector(values[_VELOCITY], "m/s"),
        angular_velocity=vector(values[_ANGULAR_VELOCITY], "rad/s"),
        orientation=Quaternion(q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z),
        current_mass=scalar(float(values[_MASS]), "kg"),
        inertia=template.inertia,
        reference_point=template.reference_point,
        cg_location=template.cg_location,
    )


def _thrust_acceleration(
    time: float, mass: float, nose_vector: np.ndarray, engine: Engine
) -> tuple[np.ndarray, float]:
    """Compute the acceleration the motor produces and the rate it burns mass.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        mass (float): Current mass in kilograms.
        nose_vector (np.ndarray): Direction the nose points in the world frame.
        engine (Engine): Motor model.

    Returns:
        tuple[np.ndarray, float]: Acceleration in the world frame in m/s**2
            and the mass flow rate in kg/s, negative while burning.
    """
    thrust_magnitude = engine.get_thrust(time)
    if thrust_magnitude > 0 and mass > 0:
        return (
            nose_vector * (thrust_magnitude / mass),
            engine.get_mass_flow(time, thrust_magnitude),
        )
    return np.zeros(3), 0.0


def _lift_and_torque(
    flight_vector: np.ndarray,
    rotation: np.ndarray,
    force_scale: float,
    coefficients: AeroCoefficients,
    inputs: _StepInputs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lift force and the aerodynamic torque about the CG.

    Args:
        flight_vector (np.ndarray): Unit vector along the airspeed, world frame.
        rotation (np.ndarray): Body-to-world rotation matrix.
        force_scale (float): Dynamic pressure times reference area, in N.
        coefficients (AeroCoefficients): Coefficients at this flight condition.
        inputs (_StepInputs): Constants of the step.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lift force in the world frame in N and
            torque in the body frame in N*m.
    """
    # Direction the nose points in the world frame: the body Z axis
    nose_vector = rotation[:, 2]
    pitch_axis = cross(flight_vector, nose_vector)
    lift_raw_dir = cross(pitch_axis, flight_vector)

    lift_norm = math.sqrt(float(lift_raw_dir @ lift_raw_dir))
    if lift_norm > 1e-6:
        lift_force = lift_raw_dir * (force_scale * coefficients.cl / lift_norm)
    else:
        lift_force = np.zeros(3)

    # Reference diameter acts as "lever arm" length
    moment_scale = force_scale * inputs.reference_diameter

    # Pitch moment about the reference point, transferred to the CG
    # by the lift acting through the lever arm (world frame)
    lever_arm_world = rotation @ inputs.lever_arm_body
    m_cg_world = pitch_axis * (moment_scale * coefficients.cm) + cross(
        lever_arm_world, lift_force
    )

    # Inertia is expressed in body axes, so the torque must be too
    body_torque = rotation.T @ m_cg_world

    # Apply the direct aerodynamic moments to the body frame
    # (Assuming Z is Roll, Y is Pitch, X is Yaw)
    body_torque[2] += moment_scale * coefficients.c_roll
    body_torque[0] += moment_scale * coefficients.cn
    return lift_force, body_torque


def _aero_loads(
    velocity: np.ndarray, altitude: float, rotation: np.ndarray, inputs: _StepInputs
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the aerodynamic force and torque on the vehicle.

    The loads follow the airspeed: the velocity relative to the wind.

    Args:
        velocity (np.ndarray): Velocity in the world frame in m/s.
        altitude (float): Altitude in metres, for the atmosphere.
        rotation (np.ndarray): Body-to-world rotation matrix.
        inputs (_StepInputs): Constants of the step.

    Returns:
        tuple[np.ndarray, np.ndarray]: Force in the world frame in N and
            torque about the CG in the body frame in N*m.
    """
    # Atmosphere at this stage's own altitude
    conditions = inputs.atmosphere(altitude)
    airspeed = velocity - conditions.wind
    speed = math.sqrt(float(airspeed @ airspeed))
    if speed <= 0:
        return np.zeros(3), np.zeros(3)

    # Unit vector pointing where the rocket is travelling through the air
    flight_vector = airspeed / speed
    current_alpha = _angle_of_attack(flight_vector, rotation[:, 2])
    coefficients = inputs.properties.aero_coefficients(
        speed / conditions.speed_of_sound, current_alpha
    )

    # Dynamic pressure times reference area gives force per unit coefficient
    force_scale = 0.5 * conditions.air_density * speed**2 * inputs.reference_area

    # Drag pushes straight back along the airspeed
    aero_force = flight_vector * (-force_scale * coefficients.cd)
    if current_alpha <= 0.001:
        return aero_force, np.zeros(3)

    lift_force, body_torque = _lift_and_torque(
        flight_vector, rotation, force_scale, coefficients, inputs
    )
    # Side force acts along the body X axis
    side_force = rotation[:, 0] * (force_scale * coefficients.cy)
    return aero_force + lift_force + side_force, body_torque


def _angle_of_attack(flight_vector: np.ndarray, nose_vector: np.ndarray) -> float:
    """Return the angle in degrees between two unit vectors."""
    dot_product = min(max(float(flight_vector @ nose_vector), -1.0), 1.0)
    return math.degrees(math.acos(dot_product))


def _state_rates(time: float, values: np.ndarray, inputs: _StepInputs) -> np.ndarray:
    """Compute the time derivative of a state array.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        values (np.ndarray): State in the layout ``_pack`` produces.
        inputs (_StepInputs): Constants of the step.

    Returns:
        np.ndarray: Rate of change of each element of ``values``.
    """
    altitude = float(values[_POSITION][2])
    velocity = values[_VELOCITY]
    mass = float(values[_MASS])
    orientation: tuple[float, float, float, float] = tuple(
        values[_ORIENTATION].tolist()
    )
    rotation = rotation_matrix(*orientation)

    # Gravity acts along -Z
    acceleration = np.array([0.0, 0.0, -inputs.gravity(inputs.latitude_rad, altitude)])
    thrust_acceleration, mass_flow_rate = _thrust_acceleration(
        time, mass, rotation[:, 2], inputs.properties.engine
    )
    acceleration += thrust_acceleration

    angular_acceleration = np.zeros(3)
    if mass > 0:
        aero_force, body_torque = _aero_loads(velocity, altitude, rotation, inputs)
        acceleration += aero_force / mass
        angular_acceleration = body_torque / inputs.inertia

    rates = np.empty(_STATE_SIZE)
    rates[_POSITION] = velocity
    rates[_VELOCITY] = acceleration
    rates[_ANGULAR_VELOCITY] = angular_acceleration
    rates[_ORIENTATION] = quaternion_rates(*orientation, values[_ANGULAR_VELOCITY])
    rates[_MASS] = mass_flow_rate
    return rates


def derivative_computation(
    time: float,
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
) -> StateDerivative:
    """Compute the time derivative of the rocket state.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        state (RocketState): Current state of the vehicle.
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Environment models.

    Returns:
        StateDerivative: Rates of change to integrate over the next step.
    """
    rates = _state_rates(time, _pack(state), _step_inputs(state, properties, config))
    q_dot = rates[_ORIENTATION].tolist()
    return StateDerivative(
        velocity=vector(rates[_POSITION], "m/s"),
        acceleration=vector(rates[_VELOCITY], "m/s**2"),
        angular_acceleration=vector(rates[_ANGULAR_VELOCITY], "rad/s**2"),
        orientation_derivative=Quaternion(
            q_w=q_dot[0], q_x=q_dot[1], q_y=q_dot[2], q_z=q_dot[3]
        ),
        mass_derivative=scalar(float(rates[_MASS]), "kg/s"),
    )


def rkf45_step(
    time: float, values: np.ndarray, dt: float, inputs: _StepInputs
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a state array by one RKF45 step.

    Args:
        time (float): Time at the start of the step, in seconds.
        values (np.ndarray): State in the layout ``_pack`` produces.
        dt (float): Step length in seconds.
        inputs (_StepInputs): Constants of the step.

    Returns:
        tuple[np.ndarray, np.ndarray]: The 5th and 4th order estimates of the
            state at ``time + dt``.
    """
    stages = np.empty((len(_RKF_NODES), _STATE_SIZE))
    for index, (node, weights) in enumerate(
        zip(_RKF_NODES, _RKF_STAGE_WEIGHTS, strict=True)
    ):
        stage_values = values + dt * (weights @ stages[:index]) if index else values
        stages[index] = _state_rates(time + node * dt, stage_values, inputs)

    return values + dt * (_RKF_5TH_ORDER @ stages), values + dt * (
        _RKF_4TH_ORDER @ stages
    )


def _adaptive_step(
    time: float, values: np.ndarray, dt: float, inputs: _StepInputs
) -> tuple[np.ndarray, float, float]:
    """Take one RKF45 step, shrinking it until its error is within tolerance.

    Args:
        time (float): Time at the start of the step, in seconds.
        values (np.ndarray): State in the layout ``_pack`` produces.
        dt (float): Step length to try first, in seconds.
        inputs (_StepInputs): Constants of the step.

    Returns:
        tuple[np.ndarray, float, float]: The new state, the length of the
            step it took, and the step length to try next, in seconds.

    Raises:
        FloatingPointError: If the error estimate is NaN or infinite.

    Warns:
        RuntimeWarning: If the step is accepted over tolerance because it is
            already at the minimum step length.
    """
    while True:
        # Look into future with 4th and 5th order RKF45 steps
        values_5th, values_4th = rkf45_step(time, values, dt, inputs)

        # Difference between the two position estimates, in metres
        position_difference = values_5th[_POSITION] - values_4th[_POSITION]
        error = math.sqrt(float(position_difference @ position_difference))

        if error <= inputs.tolerance_m:
            break
        if not math.isfinite(error):
            raise FloatingPointError(
                f"RKF45 error estimate is {error} at t = {time} s with dt = {dt} s"
            )
        if dt <= inputs.min_dt_s:
            warnings.warn(
                f"Accepted a step at t = {time} s with position error {error:.3g} m"
                f" over the {inputs.tolerance_m} m tolerance at the minimum step"
                f" length of {inputs.min_dt_s} s",
                RuntimeWarning,
                stacklevel=2,
            )
            break
        dt = max(
            dt * max(0.9 * (inputs.tolerance_m / error) ** 0.2, 0.1), inputs.min_dt_s
        )

    # Keep the orientation a unit quaternion
    orientation = values_5th[_ORIENTATION]
    values_5th[_ORIENTATION] = orientation / math.sqrt(float(orientation @ orientation))

    if error > 0:
        next_dt = dt * min(0.9 * (inputs.tolerance_m / error) ** 0.2, 1.5)
    else:
        next_dt = dt * 1.5
    return values_5th, dt, max(next_dt, inputs.min_dt_s)


def adaptive_step(
    time: float,
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
    dt: Scalar,
) -> tuple[RocketState, Scalar, Scalar]:
    """Advance the rocket state by one step whose length the error sets.

    The step is ``dt`` or shorter: RKF45 shrinks it until the position error
    is within tolerance. Pass the returned next step length back in on the
    following call so step lengths adapt to the flight. Inertia and CG
    location are held constant across the step.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        state (RocketState): Current state of the vehicle.
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Environment models and tolerances.
        dt (Scalar): Step length to try, in any unit of time.

    Returns:
        tuple[RocketState, Scalar, Scalar]: The new state, the length of the
            step taken, and the step length to try next.
    """
    values, dt_taken, next_dt = _adaptive_step(
        time,
        _pack(state),
        float(dt.m_as("s")),
        _step_inputs(state, properties, config),
    )
    return _unpack(values, state), scalar(dt_taken, "s"), scalar(next_dt, "s")


def step(
    time: float,
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
    dt: Scalar,
) -> RocketState:
    """Advance the rocket state by exactly ``dt`` using adaptive RKF45 steps.

    Inertia and CG location are held constant across the step.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        state (RocketState): Current state of the vehicle.
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Environment models and tolerances.
        dt (Scalar): Length of the step, in any unit of time.

    Returns:
        RocketState: The state after advancing by one step.
    """
    target_time = float(dt.m_as("s"))
    time_simulated = 0.0

    # Assume whole step taken at once
    current_dt = target_time
    inputs = _step_inputs(state, properties, config)
    current_values = _pack(state)

    while time_simulated < target_time:
        # Prevents the final step from overshooting the target time
        current_dt = min(current_dt, target_time - time_simulated)
        current_values, dt_taken, current_dt = _adaptive_step(
            time + time_simulated, current_values, current_dt, inputs
        )
        time_simulated += dt_taken

    return _unpack(current_values, state)


def locate_event(
    time: float,
    start: RocketState,
    end: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
    dt: Scalar,
    event: Callable[[float, RocketState], float],
) -> tuple[RocketState, Scalar]:
    """Integrate from the start of a step to where an event quantity reaches zero.

    Finds the crossing with the Illinois variant of regula falsi.

    Args:
        time (float): Time at the start of the step, in seconds.
        start (RocketState): State at the start of the step, where ``event``
            is positive.
        end (RocketState): State at the end of the step, where ``event`` is
            zero or negative.
        properties (RocketProperties): Aerodynamic and motor properties.
        config (IntegrationConfiguration): Environment models and tolerances.
        dt (Scalar): Length of the step from ``start`` to ``end``.
        event (Callable[[float, RocketState], float]): Quantity, given the
            time in seconds and the state, whose crossing from positive to
            zero or negative marks the event.

    Returns:
        tuple[RocketState, Scalar]: The state at the event, on or just past
            the crossing, and the time from ``start`` to it.
    """
    low_time, high_time = 0.0, float(dt.m_as("s"))
    low_value = event(time, start)
    high_value = event(time + high_time, end)
    high_state = end
    last_moved = 0
    time_tolerance = float(config.event_time_tolerance.m_as("s"))

    for _ in range(config.max_event_iterations):
        if high_time - low_time <= time_tolerance:
            break
        trial_time = (low_time * high_value - high_time * low_value) / (
            high_value - low_value
        )
        trial_state = step(time, start, properties, config, scalar(trial_time, "s"))
        trial_value = event(time + trial_time, trial_state)

        if trial_value == 0.0:
            return trial_state, scalar(trial_time, "s")
        if trial_value < 0.0:
            high_time, high_value, high_state = trial_time, trial_value, trial_state
            if last_moved == 1:
                # Illinois rule
                low_value /= 2.0
            last_moved = 1
        else:
            low_time, low_value = trial_time, trial_value
            if last_moved == -1:
                high_value /= 2.0
            last_moved = -1

    return high_state, scalar(high_time, "s")
