"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import Scalar, UnitChecked, Vector, zero_vector
from flight_sim.utilities.quaternion import Quaternion


@dataclass
class RocketState(UnitChecked):
    """Current 6-DOF state of the vehicle.

    Frames are defined in ``flight_sim.utilities.dcm``. ``orientation`` is a
    scalar-first quaternion rotating body axes into world axes;
    ``angular_velocity`` and ``inertia`` are in body axes.
    """

    # Mass Properties
    current_mass: Annotated[Scalar, "kg"]

    # Diagonal inertia (I_xx, I_yy, I_zz) in body axes
    inertia: Annotated[Vector, "kg*m**2"]

    # Center of gravity from the nose tip in body axes, so aft is -X
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

    # Where the configured launch rail took hold of the rocket, which is the
    # rail's foot; None when no rail holds it
    rail_start: Annotated[Vector | None, "m"] = None
