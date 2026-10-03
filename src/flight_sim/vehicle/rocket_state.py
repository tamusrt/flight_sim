"""Vehicle module containing the 6-DOF state structures of the rocket."""

from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import UnitChecked, Vector, zero_vector
from flight_sim.utilities.quaternion import Quaternion


@dataclass
class RocketState(UnitChecked):
    """Current 6-DOF state of the vehicle.

    Frames are defined in ``flight_sim.utilities.dcm``. ``orientation`` is a
    scalar-first quaternion rotating body axes into world axes;
    ``angular_velocity`` is in body axes.
    """

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

    # Where the configured launch rail took hold of the rocket, or None off the
    # rail. Ignored with no rail configured; cleared on the rail's exit event.
    rail_start: Annotated[Vector | None, "m"] = None
