"""Mass property tests: validation and the parallel-axis combination."""

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.units import matrix, scalar, ureg, vector
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


def test_combining_one_part_returns_it() -> None:
    """A single part is its own combination."""
    inertia = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]])
    part = MassPropertiesSI(5.0, np.array([-1.5, 0.01, 0.0]), inertia)

    combined = combine(part)

    assert combined.mass == pytest.approx(5.0)
    assert combined.cg_location == pytest.approx(part.cg_location)
    assert combined.inertia == pytest.approx(inertia)


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


def test_mass_properties_are_immutable() -> None:
    """The cached SI form cannot go stale through reassignment."""
    properties = MassProperties(
        scalar(1.0, "kg"), vector((0.0, 0.0, 0.0), "m"), matrix(np.eye(3), "kg*m**2")
    )

    with pytest.raises(AttributeError):
        properties.mass = scalar(2.0, "kg")  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        properties.si.inertia[0, 0] = 5.0


@pytest.mark.parametrize(
    ("mass", "cg_location", "inertia", "message"),
    [
        (1.0, (0.0, 0.0, 0.0), [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]], "symm"),
        (1.0, (0.0, 0.0, 0.0), np.ones((2, 2)), "3x3"),
        (1.0, (0.0, 0.0), np.eye(3), "3-vector"),
        (-1.0, (0.0, 0.0, 0.0), np.eye(3), "negative"),
    ],
)
def test_mass_properties_reject_invalid_values(
    mass: float,
    cg_location: tuple[float, ...],
    inertia: object,
    message: str,
) -> None:
    """Asymmetric or misshapen inertias, bad CGs and negative masses are rejected."""
    with pytest.raises(ValueError, match=message):
        MassProperties(
            mass=scalar(mass, "kg"),
            cg_location=ureg.Quantity(np.array(cg_location), "m"),
            inertia=ureg.Quantity(np.array(inertia, dtype=float), "kg*m**2"),
        )


def test_mass_properties_reject_wrong_units() -> None:
    """An inertia that is not a mass times an area is rejected."""
    with pytest.raises(DimensionalityError, match=r"MassProperties\.inertia"):
        MassProperties(
            scalar(1.0, "kg"), vector((0.0, 0.0, 0.0), "m"), matrix(np.eye(3), "kg*m")
        )
