"""Integration math for 6-DOF simulation.

The integrator works on a flat array of plain SI floats and converts to and
from unit-checked quantities only at its public entry points.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from flight_sim.environment.atmosphere import AtmosphereData, standard_conditions
from flight_sim.environment.gravity import normal_gravity
from flight_sim.units import Scalar, UnitChecked, Vector, scalar, vector, zero_vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import Quaternion, RocketState

# Layout of the flat state array: metres, m/s, rad/s, quaternion, kilograms
_POSITION = slice(0, 3)
_VELOCITY = slice(3, 6)
_ANGULAR_VELOCITY = slice(6, 9)
_ORIENTATION = slice(9, 13)  # q_w, q_x, q_y, q_z
_MASS = 13
_STATE_SIZE = 14

_LAUNCH_LATITUDE_RAD = 0.0

# Largest position disagreement between the 4th and 5th order RKF45 estimates
# that a step may have and still be accepted
_TOLERANCE_M = 1e-7

# Width of the time bracket, in seconds, within which locate_event finds an event
_EVENT_TIME_TOLERANCE_S = 1e-9
_EVENT_MAX_ITERATIONS = 100

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

    # d(orientation)/dt
    orientation_derivative: Quaternion = field(
        default_factory=lambda: Quaternion(q_w=0.0)
    )

    mass_derivative: Scalar = field(default_factory=lambda: scalar(0.0, "kg/s"))


class _StepInputs(NamedTuple):
    """Quantities held fixed across one step, as plain SI values."""

    inertia: np.ndarray  # kg*m**2, body axes
    lever_arm_body: np.ndarray  # m, aero reference point relative to the CG
    reference_area: float  # m**2
    reference_diameter: float  # m
    properties: RocketProperties


def _step_inputs(state: RocketState, properties: RocketProperties) -> _StepInputs:
    """Strip the units from everything the derivative reads besides the state."""
    return _StepInputs(
        inertia=state.inertia.m_as("kg*m**2"),
        lever_arm_body=properties.reference_point.m_as("m")
        - state.cg_location.m_as("m"),
        reference_area=float(properties.reference_area.m_as("m**2")),
        reference_diameter=float(properties.reference_diameter.m_as("m")),
        properties=properties,
    )


def _pack(state: RocketState) -> np.ndarray:
    """Flatten the integrated fields of a state into one SI array."""
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


def _rotation_matrix(q_w: float, q_x: float, q_y: float, q_z: float) -> np.ndarray:
    """Return the body-to-world rotation matrix of a quaternion of any length."""
    s = 2.0 / (q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
    return np.array(
        [
            [
                1.0 - s * (q_y * q_y + q_z * q_z),
                s * (q_x * q_y - q_z * q_w),
                s * (q_x * q_z + q_y * q_w),
            ],
            [
                s * (q_x * q_y + q_z * q_w),
                1.0 - s * (q_x * q_x + q_z * q_z),
                s * (q_y * q_z - q_x * q_w),
            ],
            [
                s * (q_x * q_z - q_y * q_w),
                s * (q_y * q_z + q_x * q_w),
                1.0 - s * (q_x * q_x + q_y * q_y),
            ],
        ]
    )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two 3-vectors."""
    a_x, a_y, a_z = a.tolist()
    b_x, b_y, b_z = b.tolist()
    return np.array(
        [a_y * b_z - a_z * b_y, a_z * b_x - a_x * b_z, a_x * b_y - a_y * b_x]
    )


def _quaternion_rates(
    q_w: float, q_x: float, q_y: float, q_z: float, angular_velocity: np.ndarray
) -> np.ndarray:
    """Evaluate q_dot = 0.5 * Xi(q) * omega, with omega in rad/s, body frame."""
    p, q, r = angular_velocity.tolist()
    return 0.5 * np.array(
        [
            -q_x * p - q_y * q - q_z * r,
            q_w * p - q_z * q + q_y * r,
            q_z * p + q_w * q - q_x * r,
            -q_y * p + q_x * q + q_w * r,
        ]
    )


def quaternion_kinematics(
    orientation: Quaternion, angular_velocity: Vector
) -> Quaternion:
    """Compute the quaternion rate from the body-frame angular velocity.

    Evaluates the kinematic differential equation q_dot = 0.5 * Xi(q) * omega.

    Args:
        orientation (Quaternion): Current body-to-world orientation.
        angular_velocity (Vector): Angular velocity expressed in the body frame.

    Returns:
        Quaternion: Rate of change of each orientation component, in 1/s.
    """
    q_dot = _quaternion_rates(
        orientation.q_w,
        orientation.q_x,
        orientation.q_y,
        orientation.q_z,
        angular_velocity.m_as("rad/s"),
    ).tolist()
    return Quaternion(q_w=q_dot[0], q_x=q_dot[1], q_y=q_dot[2], q_z=q_dot[3])


# pylint: disable=too-many-locals
def _state_rates(time: float, values: np.ndarray, inputs: _StepInputs) -> np.ndarray:
    """Compute the time derivative of a flat SI state array.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        values (np.ndarray): State in the layout ``_pack`` produces.
        inputs (_StepInputs): Mass properties and aero data.

    Returns:
        np.ndarray: Rate of change of each element of ``values``.
    """
    properties = inputs.properties
    velocity = values[_VELOCITY]
    mass = float(values[_MASS])
    q_w, q_x, q_y, q_z = values[_ORIENTATION].tolist()

    rotation = _rotation_matrix(q_w, q_x, q_y, q_z)
    # Direction the nose points in the world frame: the body Z axis
    nose_vector = rotation[:, 2]

    # Gravity acts along -Z
    acceleration = np.array(
        [0.0, 0.0, -normal_gravity(_LAUNCH_LATITUDE_RAD, float(values[2]))]
    )

    thrust_magnitude = float(properties.thrust_curve(time))
    if thrust_magnitude > 0 and mass > 0:
        acceleration += nose_vector * (thrust_magnitude / mass)
        mass_flow_rate = -thrust_magnitude * properties.mass_flow_multiplier
    else:
        mass_flow_rate = 0.0

    angular_acceleration = np.zeros(3)
    speed = math.sqrt(float(velocity @ velocity))

    if speed > 0 and mass > 0:
        # Unit vector pointing where the rocket is travelling
        flight_vector = velocity / speed
        dot_product = min(max(float(flight_vector @ nose_vector), -1.0), 1.0)

        current_alpha = math.degrees(math.acos(dot_product))
        # Atmosphere at this stage's own altitude
        conditions = standard_conditions(float(values[2]))
        current_mach = speed / conditions.speed_of_sound
        cd, cl, cy, c_roll, cm, cn = properties.aero_coefficients(
            current_mach, current_alpha
        )

        # Dynamic pressure times reference area gives force per unit coefficient
        force_scale = 0.5 * conditions.air_density * speed**2 * inputs.reference_area
        # Reference diameter acts as "lever arm" length
        moment_scale = force_scale * inputs.reference_diameter

        # Drag pushes straight back along the flight path
        aero_force = flight_vector * (-force_scale * cd)

        if current_alpha > 0.001:
            pitch_axis = _cross(flight_vector, nose_vector)
            lift_raw_dir = _cross(pitch_axis, flight_vector)

            lift_norm = math.sqrt(float(lift_raw_dir @ lift_raw_dir))
            if lift_norm > 1e-6:
                lift_force = lift_raw_dir * (force_scale * cl / lift_norm)
            else:
                lift_force = np.zeros(3)

            # Pitch moment about the reference point, transferred to the CG
            # by the lift acting through the lever arm (world frame)
            lever_arm_world = rotation @ inputs.lever_arm_body
            m_cg_world = pitch_axis * (moment_scale * cm) + _cross(
                lever_arm_world, lift_force
            )

            # Inertia is expressed in body axes, so the torque must be too
            body_torque = rotation.T @ m_cg_world

            # Apply the direct aerodynamic moments to the body frame
            # (Assuming Z is Roll, Y is Pitch, X is Yaw)
            body_torque[2] += moment_scale * c_roll
            body_torque[0] += moment_scale * cn

            # Side force acts along the body X axis
            side_force = rotation[:, 0] * (force_scale * cy)

            aero_force = aero_force + lift_force + side_force
            angular_acceleration = body_torque / inputs.inertia

        acceleration += aero_force / mass

    rates = np.empty(_STATE_SIZE)
    rates[_POSITION] = velocity
    rates[_VELOCITY] = acceleration
    rates[_ANGULAR_VELOCITY] = angular_acceleration
    rates[_ORIENTATION] = _quaternion_rates(
        q_w, q_x, q_y, q_z, values[_ANGULAR_VELOCITY]
    )
    rates[_MASS] = mass_flow_rate
    return rates


def derivative_computation(
    time: float,
    state: RocketState,
    properties: RocketProperties,
) -> StateDerivative:
    """Compute the time derivative of the rocket state.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        state (RocketState): Current state of the vehicle.
        properties (RocketProperties): Aerodynamic properties of rocket

    Returns:
        StateDerivative: Rates of change to integrate over the next step.
    """
    rates = _state_rates(time, _pack(state), _step_inputs(state, properties))
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
    """Advance a flat SI state by one RKF45 step.

    Args:
        time (float): Time at the start of the step, in seconds.
        values (np.ndarray): State in the layout ``_pack`` produces.
        dt (float): Step length in seconds.
        inputs (_StepInputs): Quantities held fixed across the step.

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


def _accepted_step(
    time: float, values: np.ndarray, dt: float, inputs: _StepInputs
) -> tuple[np.ndarray, float, float]:
    """Take one RKF45 step, shrinking it until its error is within tolerance.

    Args:
        time (float): Time at the start of the step, in seconds.
        values (np.ndarray): State in the layout ``_pack`` produces.
        dt (float): Step length to try first, in seconds.
        inputs (_StepInputs): Quantities held fixed across the step.

    Returns:
        tuple[np.ndarray, float, float]: The new state, the length of the
            step it took, and the step length to try next, in seconds.
    """
    while True:
        # Look into future with 4th and 5th order RKF45 steps
        values_5th, values_4th = rkf45_step(time, values, dt, inputs)

        # Difference between the two position estimates, in metres
        position_difference = values_5th[_POSITION] - values_4th[_POSITION]
        error = math.sqrt(float(position_difference @ position_difference))

        # Check if the error is within the tolerance
        if error <= _TOLERANCE_M:
            break
        scale_factor = 0.9 * (_TOLERANCE_M / error) ** 0.2
        dt *= max(scale_factor, 0.1)

    # Additive RKF45 stages drift the quaternion off unit length
    orientation = values_5th[_ORIENTATION]
    values_5th[_ORIENTATION] = orientation / math.sqrt(float(orientation @ orientation))

    if error > 0:
        next_dt = dt * min(0.9 * (_TOLERANCE_M / error) ** 0.2, 1.5)
    else:
        next_dt = dt * 1.5
    return values_5th, dt, next_dt


def adaptive_step(
    time: float,
    state: RocketState,
    properties: RocketProperties,
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
        dt (Scalar): Step length to try, in any unit of time.

    Returns:
        tuple[RocketState, Scalar, Scalar]: The new state, the length of the
            step taken, and the step length to try next.
    """
    values, dt_taken, next_dt = _accepted_step(
        time,
        _pack(state),
        float(dt.m_as("s")),
        _step_inputs(state, properties),
    )
    return _unpack(values, state), scalar(dt_taken, "s"), scalar(next_dt, "s")


def step(
    time: float,
    state: RocketState,
    properties: RocketProperties,
    dt: Scalar,
) -> RocketState:
    """Advance the rocket state by exactly ``dt`` using adaptive RKF45 steps.

    Inertia and CG location are held constant across the step.

    Args:
        time (float): Time since ignition in seconds, for the thrust curve.
        state (RocketState): Current state of the vehicle.
        properties (RocketProperties): Aerodynamic and motor properties.
        dt (Scalar): Length of the step, in any unit of time.

    Returns:
        RocketState: The state after advancing by one step.
    """
    target_time = float(dt.m_as("s"))
    time_simulated = 0.0

    # Assume whole step taken at once
    current_dt = target_time
    inputs = _step_inputs(state, properties)
    current_values = _pack(state)

    while time_simulated < target_time:
        # Prevents the final step from overshooting the target time
        current_dt = min(current_dt, target_time - time_simulated)
        current_values, dt_taken, current_dt = _accepted_step(
            time + time_simulated, current_values, current_dt, inputs
        )
        time_simulated += dt_taken

    return _unpack(current_values, state)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def locate_event(
    time: float,
    start: RocketState,
    end: RocketState,
    properties: RocketProperties,
    dt: Scalar,
    event: Callable[[float, RocketState], float],
) -> tuple[RocketState, Scalar]:
    """Integrate from the start of a step to where an event quantity reaches zero.

    Finds the crossing with the Illinois variant of regula falsi, integrating
    from ``start`` afresh for every trial time, so the returned state comes
    from the integrator rather than from interpolation.

    Args:
        time (float): Time at the start of the step, in seconds.
        start (RocketState): State at the start of the step, where ``event``
            is positive.
        end (RocketState): State at the end of the step, where ``event`` is
            zero or negative.
        properties (RocketProperties): Aerodynamic and motor properties.
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

    for _ in range(_EVENT_MAX_ITERATIONS):
        if high_time - low_time <= _EVENT_TIME_TOLERANCE_S:
            break
        trial_time = (low_time * high_value - high_time * low_value) / (
            high_value - low_value
        )
        trial_state = step(time, start, properties, scalar(trial_time, "s"))
        trial_value = event(time + trial_time, trial_state)

        if trial_value == 0.0:
            return trial_state, scalar(trial_time, "s")
        if trial_value < 0.0:
            high_time, high_value, high_state = trial_time, trial_value, trial_state
            if last_moved == 1:
                # Halving the stale end keeps both ends of the bracket moving
                low_value /= 2.0
            last_moved = 1
        else:
            low_time, low_value = trial_time, trial_value
            if last_moved == -1:
                high_value /= 2.0
            last_moved = -1

    return high_state, scalar(high_time, "s")
