"""Vehicle module containing the 6-DOF state structures of the rocket."""

import math
from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import Scalar, UnitChecked, Vector, vector, zero_vector


@dataclass
class Quaternion:
    """Quaternion representation tracking vehicle orientation."""

    # Components are dimensionless by definition, so they stay plain floats.
    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0

    def normalized(self) -> "Quaternion":
        """Rescale to unit length so the quaternion stays a valid rotation.

        Returns:
            Quaternion: A unit quaternion pointing the same way as this one.
        """
        norm = math.sqrt(self.q_w**2 + self.q_x**2 + self.q_y**2 + self.q_z**2)
        return Quaternion(
            q_w=self.q_w / norm,
            q_x=self.q_x / norm,
            q_y=self.q_y / norm,
            q_z=self.q_z / norm,
        )


@dataclass
class RocketState(UnitChecked):
    """Current 6-DOF state of the vehicle."""

    # Mass Properties
    current_mass: Annotated[Scalar, "kg"]

    # I_xx (pitch), I_yy (yaw), I_zz (roll) in kg*m^2
    inertia: Annotated[Vector, "kg*m**2"]

    # Center of gravoty relative to the nose tip
    cg_location: Annotated[Vector, "m"]

    # Position Coordinates
    position: Annotated[Vector, "m"] = field(default_factory=lambda: zero_vector("m"))

    # Velocity
    velocity: Annotated[Vector, "m/s"] = field(
        default_factory=lambda: zero_vector("m/s")
    )

    # Angular Velocities
    angular_velocity: Annotated[Vector, "rad/s"] = field(
        default_factory=lambda: zero_vector("rad/s")
    )

    # Orientation
    orientation: Quaternion = field(default_factory=Quaternion)

    # Reference point used by standard_aero.csv
    reference_point: Annotated[Vector, "m"] = field(
        default_factory=lambda: vector((0.0, 0.0, 0.0), "m")
    )
