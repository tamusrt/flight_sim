"""Integration math for 6-DOF simulation."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore

from flight_sim.environment.atmosphere import AtmosphereData
from flight_sim.environment.gravity import get_gravity
from flight_sim.units import Scalar, UnitChecked, Vector, scalar, vector, zero_vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import Quaternion, RocketState


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
    q_0 = orientation.q_w
    q_1 = orientation.q_x
    q_2 = orientation.q_y
    q_3 = orientation.q_z

    xi_matrix = np.array(
        [
            [-q_1, -q_2, -q_3],
            [q_0, -q_3, q_2],
            [q_3, q_0, -q_1],
            [-q_2, q_1, q_0],
        ]
    )
    omega_raw = angular_velocity.m_as("rad/s")
    q_dot = 0.5 * (xi_matrix @ omega_raw)

    return Quaternion(
        q_w=float(q_dot[0]),
        q_x=float(q_dot[1]),
        q_y=float(q_dot[2]),
        q_z=float(q_dot[3]),
    )


# pylint: disable=unused-argument,too-many-locals,too-many-statements
def derivative_computation(
    time: float,
    state: RocketState,
    atmosphere: AtmosphereData,
    properties: RocketProperties,
) -> StateDerivative:
    """Compute the time derivative of the rocket state.

    Args:
        state (RocketState): Current state of the vehicle.
        atmosphere (AtmosphereData): Conditions at the vehicle's altitude.
        properties (RocketProperties): Aerodynamic properties of rocket

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

    total_acceleration = magnitude_of_gravity * down

    rocket_rotation = Rotation.from_quat(
        [
            state.orientation.q_x,
            state.orientation.q_y,
            state.orientation.q_z,
            state.orientation.q_w,
        ]
    )  # Turns quaternion into something usable in 3D
    pad_vector = np.array([0.0, 0.0, 1.0])
    nose_vector = rocket_rotation.apply(
        pad_vector
    )  # Apply rocket's current spin and tilt to pad_vector,
    # resulting in vector pointing where rocket's physical tip is

    thrust_magnitude = float(properties.thrust_curve(time))

    if thrust_magnitude > 0 and state.current_mass.magnitude > 0:
        thrust_arr = nose_vector * thrust_magnitude
        thrust_vector = vector(
            (float(thrust_arr[0]), float(thrust_arr[1]), float(thrust_arr[2])), "N"
        )
        thrust_accel = thrust_vector / state.current_mass
        total_acceleration += thrust_accel

        dm_dt = -thrust_magnitude * properties.mass_flow_multiplier
        mass_flow_rate = scalar(dm_dt, "kg/s")
    else:
        mass_flow_rate = scalar(0.0, "kg/s")

    speed = scalar(
        float(np.linalg.norm(state.velocity.to("m/s").magnitude)), "m/s"
    )  # Velocity vector magnitude "speed"

    if speed.magnitude > 0 and state.current_mass.magnitude > 0:
        velocity_raw = state.velocity.to("m/s").magnitude
        speed_raw = speed.to("m/s").magnitude
        flight_vector = (
            velocity_raw / speed_raw
        )  # Divides 3D velocity by a scalar to get directional arrow
        # pointing where the wind is hitting the rocket
        dot_product = np.clip(
            np.dot(flight_vector, nose_vector), -1.0, 1.0
        )  # Dot product between flight vector and direction of nose tip

        current_alpha = float(np.degrees(np.arccos(dot_product)))
        current_mach = float((speed / atmosphere.speed_of_sound).magnitude)

        cd = float(
            float(properties.cd_table([current_mach, current_alpha]).item())
        )  # Interpolates grid from RocketProperties to find CD
        cl = float(float(properties.cl_table([current_mach, current_alpha]).item()))
        cy = float(properties.cy_table([current_mach, current_alpha]).item())

        c_roll = float(properties.c_roll_table([current_mach, current_alpha]).item())
        cm = float(float(properties.cm_table([current_mach, current_alpha]).item()))
        cn = float(properties.cn_table([current_mach, current_alpha]).item())

        q = 0.5 * atmosphere.air_density * (speed**2)  # Dynamic Pressure
        drag_force_magnitude = q * properties.reference_area * cd
        lift_force_magnitude = q * properties.reference_area * cl
        pitch_torque_magnitude = (
            q * properties.reference_area * properties.reference_diameter * cm
        )  # Reference diameter acts as "lever arm" length

        drag_mag_raw = float(drag_force_magnitude.m_as("N"))
        lift_mag_raw = float(lift_force_magnitude.m_as("N"))
        torque_mag_raw = float(pitch_torque_magnitude.m_as("N*m"))

        velocity_dir_raw = velocity_raw / speed_raw

        drag_arr = -velocity_dir_raw * drag_mag_raw
        drag_force_vector = vector(
            (float(drag_arr[0]), float(drag_arr[1]), float(drag_arr[2])), "N"
        )  # Negative sign to make force push backwards

        drag_accel = drag_force_vector / state.current_mass  # F = ma --> a = F/m

        if current_alpha > 0.001:
            pitch_axis = np.cross(flight_vector, nose_vector)
            lift_raw_dir = np.cross(pitch_axis, flight_vector)

            lift_norm = np.linalg.norm(lift_raw_dir)
            if lift_norm > 1e-6:
                lift_dir_array = lift_raw_dir / lift_norm
            else:
                lift_dir_array = np.array([0.0, 0.0, 0.0])

            lift_arr = lift_dir_array * lift_mag_raw

            lift_force_vector = vector(
                (float(lift_arr[0]), float(lift_arr[1]), float(lift_arr[2])), "N"
            )  # Multiply direction array by Pint magnitude to retain units

            r_ref = properties.reference_point.m_as("m")
            r_cg = state.cg_location.m_as("m")
            lever_arm_body = r_ref - r_cg  # Becomes [0, 0, 1.5], same as CG

            lever_arm_world = rocket_rotation.apply(
                lever_arm_body
            )  # Must calculate based on global world frame.
            # Rotate the lever arm into the 3D world frame.

            m_ref_world = (
                pitch_axis * torque_mag_raw
            )  # Calculates the baseline twisting force (Assuming around nose).

            cg_shift_torque = np.cross(lever_arm_world, lift_arr)

            m_cg_world = (
                m_ref_world + cg_shift_torque
            )  # Calculates amount of torque generated by normal force,
            # and adds it to find dynamic torque on CG.

            body_torque = rocket_rotation.inv().apply(
                m_cg_world
            )  # Convert world torque back to the rocket's body frame,
            # as the rocket's moment of inertia only exists in the Body Frame.

            roll_torque_magnitude = (
                q * properties.reference_area * properties.reference_diameter * c_roll
            )
            yaw_torque_magnitude = (
                q * properties.reference_area * properties.reference_diameter * cn
            )
            side_force_magnitude = q * properties.reference_area * cy

            # Apply the direct aerodynamic moments to the body frame
            # (Assuming Z is Roll, Y is Pitch, X is Yaw)
            body_torque[2] += float(roll_torque_magnitude.m_as("N*m"))
            body_torque[0] += float(yaw_torque_magnitude.m_as("N*m"))

            # To apply cy (Side Force), map it to
            # the body X-axis and rotate it to the world frame
            side_force_body = np.array(
                [float(side_force_magnitude.m_as("N")), 0.0, 0.0]
            )
            side_force_world = rocket_rotation.apply(side_force_body)
            side_force_vector = vector(
                (
                    float(side_force_world[0]),
                    float(side_force_world[1]),
                    float(side_force_world[2]),
                ),
                "N",
            )
            lift_force_vector = lift_force_vector + side_force_vector

            # Torque to angular accelertaion (alpha = Torque / Inertia)
            inertia_arr = state.inertia.m_as("kg*m**2")
            angular_arr = body_torque / inertia_arr

            angular_accel_vector = vector(
                (float(angular_arr[0]), float(angular_arr[1]), float(angular_arr[2])),
                "rad/s**2",
            )
        else:
            lift_force_vector = vector(
                (0.0, 0.0, 0.0), "N"
            )  # If flying perfectly straight, lift force is zero
            angular_accel_vector = vector((0.0, 0.0, 0.0), "rad/s**2")

        lift_accel = lift_force_vector / state.current_mass
        total_acceleration += drag_accel + lift_accel

    else:
        # If not moving, no aerodynamic forces or torques
        angular_accel_vector = vector((0.0, 0.0, 0.0), "rad/s**2")

    return StateDerivative(
        velocity=state.velocity,
        acceleration=total_acceleration,
        angular_acceleration=angular_accel_vector,
        orientation_derivative=quaternion_kinematics(
            state.orientation, state.angular_velocity
        ),
        mass_derivative=mass_flow_rate,
    )


def apply_derivative(
    state: RocketState, derivatives: StateDerivative, dt_scale: Scalar
) -> RocketState:
    """Helper to apply a scaled derivative to a state for RKF45 integration."""
    # Quaternion components are plain floats, so the time scale is stripped too
    dt_scale_raw = float(dt_scale.m_as("s"))
    orientation = Quaternion(
        q_w=state.orientation.q_w
        + (derivatives.orientation_derivative.q_w * dt_scale_raw),
        q_x=state.orientation.q_x
        + (derivatives.orientation_derivative.q_x * dt_scale_raw),
        q_y=state.orientation.q_y
        + (derivatives.orientation_derivative.q_y * dt_scale_raw),
        q_z=state.orientation.q_z
        + (derivatives.orientation_derivative.q_z * dt_scale_raw),
    )

    return RocketState(
        position=state.position + (derivatives.velocity * dt_scale),
        velocity=state.velocity + (derivatives.acceleration * dt_scale),
        angular_velocity=state.angular_velocity
        + (derivatives.angular_acceleration * dt_scale),
        orientation=orientation,
        current_mass=state.current_mass + (derivatives.mass_derivative * dt_scale),
    )


# pylint: disable=too-many-locals
def rkf45_step(
    time: float,
    state: RocketState,
    atmosphere: AtmosphereData,
    properties: RocketProperties,
    dt: Scalar,
) -> tuple[RocketState, RocketState]:
    """Advance the rocket state by one time step using RKF45 integration."""

    dt_raw = dt.to("s").magnitude

    k1 = derivative_computation(time, state, atmosphere, properties)

    dt_k2 = dt * 0.25
    state_k2 = apply_derivative(state, k1, dt_k2)
    k2 = derivative_computation(
        time + (dt_raw * 0.25), state_k2, atmosphere, properties
    )

    dt_k3_1 = dt * (3 / 32)
    dt_k3_2 = dt * (9 / 32)
    state_k3_partial = apply_derivative(state, k1, dt_k3_1)
    state_k3 = apply_derivative(state_k3_partial, k2, dt_k3_2)
    k3 = derivative_computation(
        time + (dt_raw * (3 / 8)), state_k3, atmosphere, properties
    )

    dt_k4_1 = dt * (1932 / 2197)
    dt_k4_2 = dt * (-7200 / 2197)
    dt_k4_3 = dt * (7296 / 2197)
    state_k4_partial1 = apply_derivative(state, k1, dt_k4_1)
    state_k4_partial2 = apply_derivative(state_k4_partial1, k2, dt_k4_2)
    state_k4 = apply_derivative(state_k4_partial2, k3, dt_k4_3)
    k4 = derivative_computation(
        time + (dt_raw * (12 / 13)), state_k4, atmosphere, properties
    )

    dt_k5_1 = dt * (439 / 216)
    dt_k5_2 = dt * (-8)
    dt_k5_3 = dt * (3680 / 513)
    dt_k5_4 = dt * (-845 / 4104)
    state_k5_partial1 = apply_derivative(state, k1, dt_k5_1)
    state_k5_partial2 = apply_derivative(state_k5_partial1, k2, dt_k5_2)
    state_k5_partial3 = apply_derivative(state_k5_partial2, k3, dt_k5_3)
    state_k5 = apply_derivative(state_k5_partial3, k4, dt_k5_4)
    k5 = derivative_computation(time + dt_raw, state_k5, atmosphere, properties)

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
    k6 = derivative_computation(time + (dt_raw * 0.5), state_k6, atmosphere, properties)

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


def step(
    time: float,
    state: RocketState,
    atmosphere: AtmosphereData,
    properties: RocketProperties,
    dt: Scalar,
) -> RocketState:
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
        current_dt = min(current_dt, target_time - time_simulated)

        # Wraps the raw float back into Pint Scalar for the RKF45 step
        dt_scalar = scalar(current_dt, "s")

        # Look into future with 4th and 5th order RKF45 steps
        state_5th, state_4th = rkf45_step(
            time + time_simulated, current_state, atmosphere, properties, dt_scalar
        )

        # Calculate difference between two position estimates in meters
        position_difference = (
            (state_5th.position - state_4th.position).to("m").magnitude
        )
        error = float(np.linalg.norm(position_difference))

        # Check if the error is within the tolerance
        if error <= tolerance:
            current_state = state_5th
            # Additive RKF45 stages drift the quaternion off unit length
            current_state.orientation = state_5th.orientation.normalized()
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
