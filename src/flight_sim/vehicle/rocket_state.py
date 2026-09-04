"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass, field

import numpy as np


def zero_array() -> np.ndarray:
    """Return a zero-valued three-dimensional array."""
    return np.zeros(3)


@dataclass
class Quaternion:
    """Quaternion representaion to track orientation"""

    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0


@dataclass
class RocketState:
    """Dataclass representing the current 6-DOF state of the vehicle."""

    # Position Coordinates (meters)
    position: np.ndarray = field(default_factory=zero_array)

    # Velocity (m/s)
    velocity: np.ndarray = field(default_factory=zero_array)

    # Angular Velocities (rad/sec)
    angular_velocity: np.ndarray = field(default_factory=zero_array)

    # Orientatiom
    orientation: Quaternion = field(default_factory=Quaternion)

    # Mass Properties (kg)
    current_mass: float = 0.0
