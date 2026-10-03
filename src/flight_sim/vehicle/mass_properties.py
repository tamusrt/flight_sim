"""Mass, center of gravity and inertia of the rocket's parts."""

from dataclasses import dataclass, field
from typing import Annotated, NamedTuple

import numpy as np

from flight_sim.units import Matrix, Scalar, UnitChecked, Vector

_IDENTITY = np.eye(3)


class MassPropertiesSI(NamedTuple):
    """Mass properties of a part, as plain SI values."""

    mass: float  # kg
    cg_location: np.ndarray  # m, from the nose tip in body axes, so aft is -X
    inertia: np.ndarray  # kg*m**2, 3x3 about the part's own CG in body axes


@dataclass(frozen=True)
class MassProperties(UnitChecked):
    """Mass, CG and inertia of a part of the rocket."""

    mass: Annotated[Scalar, "kg"]

    # From the nose tip in body axes, so aft is -X
    cg_location: Annotated[Vector, "m"]

    # About the part's own CG, in body axes
    inertia: Annotated[Matrix, "kg*m**2"]

    si: MassPropertiesSI = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the values and cache them in SI units.

        Raises:
            ValueError: If the mass is negative, the CG location is not a
                3-vector, or the inertia is not a symmetric 3x3 matrix.
        """
        super().__post_init__()
        mass = float(self.mass.m_as("kg"))
        cg_location = np.array(self.cg_location.m_as("m"), dtype=float)
        inertia = np.array(self.inertia.m_as("kg*m**2"), dtype=float)
        if mass < 0.0:
            raise ValueError(f"Mass must not be negative, got {mass} kg")
        if cg_location.shape != (3,):
            raise ValueError(f"Expected a 3-vector CG, got shape {cg_location.shape}")
        if inertia.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 inertia, got shape {inertia.shape}")
        if not np.allclose(inertia, inertia.T):
            raise ValueError(f"Inertia must be symmetric, got {inertia.tolist()}")
        cg_location.flags.writeable = False
        inertia.flags.writeable = False
        object.__setattr__(self, "si", MassPropertiesSI(mass, cg_location, inertia))


def combine(*parts: MassPropertiesSI) -> MassPropertiesSI:
    """Combine parts into one body by the parallel-axis theorem.

    Args:
        *parts (MassPropertiesSI): Parts in the same body axes.

    Returns:
        MassPropertiesSI: Total mass, mass-weighted CG, and inertia about
            that CG.

    Raises:
        ValueError: If the total mass is not positive.
    """
    mass = sum(part.mass for part in parts)
    if not mass > 0.0:
        raise ValueError(f"Total mass must be positive, got {mass} kg")
    cg_location = np.zeros(3)
    for part in parts:
        cg_location += part.mass * part.cg_location
    cg_location /= mass
    inertia = np.zeros((3, 3))
    for part in parts:
        offset = part.cg_location - cg_location
        inertia += part.inertia + part.mass * (
            float(offset @ offset) * _IDENTITY - np.outer(offset, offset)
        )
    return MassPropertiesSI(mass, cg_location, inertia)
