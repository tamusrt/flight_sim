"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass


@dataclass
class RocketState:
    """Dataclass representing the current state of the vehicle."""

    # pylint: disable=too-many-instance-attributes

    # Position Coordinates (meters)
    x_pos: float = 0.0
    y_pos: float = 0.0
    z_pos: float = 0.0

    # Velocity (m/s)
    x_vel: float = 0.0
    y_vel: float = 0.0
    z_vel: float = 0.0

    # Orientation Quaternions
    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0

    # Angular Velocities (rad/sec)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    # Mass Properties (kg)
    current_mass: float = 0.0
