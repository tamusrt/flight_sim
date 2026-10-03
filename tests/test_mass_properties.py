"""Mass property tests: SI caching and the parallel-axis combination."""

import numpy as np
import pytest

from flight_sim.units import matrix, scalar, vector
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)


def _point_mass(mass: float, x: float) -> MassPropertiesSI:
    """Return a point mass on the body X axis."""
    return MassPropertiesSI(mass, np.array([x, 0.0, 0.0]), np.zeros((3, 3)))


def test_two_point_masses_combine_by_the_parallel_axis_theorem() -> None:
    """Two point masses give the textbook CG and inertia of a dumbbell."""
    combined = combine(_point_mass(1.0, -1.0), _point_mass(3.0, -3.0))

    # CG at -2.5 m, so the masses sit 1.5 m and 0.5 m from it
    assert combined.mass == pytest.approx(4.0)
    assert combined.cg_location == pytest.approx([-2.5, 0.0, 0.0])
    transverse = 1.0 * 1.5**2 + 3.0 * 0.5**2
    assert combined.inertia == pytest.approx(np.diag([0.0, transverse, transverse]))


def test_off_axis_offset_gives_products_of_inertia() -> None:
    """An offset in two axes adds the matching product of inertia."""
    combined = combine(
        MassPropertiesSI(1.0, np.array([1.0, 1.0, 0.0]), np.zeros((3, 3))),
        MassPropertiesSI(1.0, np.array([-1.0, -1.0, 0.0]), np.zeros((3, 3))),
    )

    assert combined.cg_location == pytest.approx(np.zeros(3))
    assert combined.inertia == pytest.approx(
        np.array([[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [0.0, 0.0, 4.0]])
    )


@pytest.mark.parametrize("parts", [(), (_point_mass(0.0, -1.0),)])
def test_combining_no_mass_is_rejected(parts: tuple[MassPropertiesSI, ...]) -> None:
    """A body with no mass has no CG."""
    with pytest.raises(ValueError, match="Total mass"):
        combine(*parts)


def test_mass_properties_cache_si_values() -> None:
    """The SI form holds the same values in kg, m and kg*m**2."""
    properties = MassProperties(
        mass=scalar(2000.0, "g"),
        cg_location=vector((-100.0, 0.0, 0.0), "cm"),
        inertia=matrix(np.diag((1.0, 2.0, 2.0)), "kg*m**2"),
    )

    assert properties.si.mass == pytest.approx(2.0)
    assert properties.si.cg_location == pytest.approx([-1.0, 0.0, 0.0])
    assert properties.si.inertia == pytest.approx(np.diag((1.0, 2.0, 2.0)))


def test_mass_properties_reject_an_asymmetric_inertia() -> None:
    """An inertia tensor is symmetric."""
    with pytest.raises(ValueError, match="symmetric"):
        MassProperties(
            scalar(1.0, "kg"),
            vector((0.0, 0.0, 0.0), "m"),
            matrix([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "kg*m**2"),
        )
