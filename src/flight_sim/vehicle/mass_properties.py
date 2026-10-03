"""Mass, center of gravity and inertia of the rocket's parts."""

from dataclasses import dataclass, field
from functools import reduce
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
        """Check the inertia and cache the values in SI units.

        Raises:
            ValueError: If the inertia is not symmetric.
        """
        super().__post_init__()
        inertia = self.inertia.m_as("kg*m**2")
        if not np.allclose(inertia, inertia.T):
            raise ValueError(f"Inertia must be symmetric, got {inertia.tolist()}")
        si = MassPropertiesSI(
            float(self.mass.m_as("kg")), self.cg_location.m_as("m"), inertia
        )
        object.__setattr__(self, "si", si)


def _combine_pair(a: MassPropertiesSI, b: MassPropertiesSI) -> MassPropertiesSI:
    """Combine two parts, using their reduced mass for the parallel-axis term."""
    mass = a.mass + b.mass
    offset = a.cg_location - b.cg_location
    inertia = a.inertia + b.inertia
    inertia += (a.mass * b.mass / mass) * (
        float(offset @ offset) * _IDENTITY - np.outer(offset, offset)
    )
    return MassPropertiesSI(mass, b.cg_location + (a.mass / mass) * offset, inertia)


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
    return reduce(_combine_pair, parts)
