"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import Scalar, UnitChecked, Vector, vector, zero_vector
from flight_sim.utilities.quaternion import Quaternion


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
