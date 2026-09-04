"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass, field

from flight_sim.units import Scalar, UnitChecked, Vector, scalar, zero_vector


@dataclass
class Quaternion:
    """Quaternion representation tracking vehicle orientation."""

    # Components are dimensionless by definition, so they stay plain floats.
    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0


@dataclass
class RocketState(UnitChecked):
    """Current 6-DOF state of the vehicle."""

    # Position Coordinates
    position: Vector = field(default_factory=lambda: zero_vector("m"))

    # Velocity
    velocity: Vector = field(default_factory=lambda: zero_vector("m/s"))

    # Angular Velocities
    angular_velocity: Vector = field(default_factory=lambda: zero_vector("rad/s"))

    # Orientation
    orientation: Quaternion = field(default_factory=Quaternion)

    # Mass Properties
    current_mass: Scalar = field(default_factory=lambda: scalar(0.0, "kg"))
