"""Unit-system tests: quantity construction and dataclass unit checking."""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.integration import (
    SimConfiguration,
    StateDerivative,
    TruthConfiguration,
)
from flight_sim.units import (
    Scalar,
    UnitChecked,
    Vector,
    matrix,
    scalar,
    vector,
)
from flight_sim.utilities.data_loader import AeroTable
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_state import RocketState


def _rocket_state(**overrides: Any) -> RocketState:
    """Return a valid RocketState with the given fields replaced."""
    return replace(RocketState(), **overrides)


def test_scalar_and_vector_constructors() -> None:
    """The constructors attach the requested units to the requested magnitudes."""
    assert scalar(500.0, "kg").m_as("kg") == pytest.approx(500.0)
    assert np.allclose(vector((1.0, 2.0, 3.0), "m").m_as("m"), [1.0, 2.0, 3.0])


def test_matrix_constructor() -> None:
    """The matrix constructor copies the components and attaches the units."""
    components = np.diag((1.0, 2.0, 3.0))
    inertia = matrix(components, "kg*m**2")
    components[0, 0] = 5.0

    assert inertia.m_as("g*m**2") == pytest.approx(1000.0 * np.diag((1.0, 2.0, 3.0)))


def test_quantities_convert_between_units() -> None:
    """A quantity built in one unit reads back correctly in a compatible one."""
    assert vector((0.0, 0.0, 100.0), "ft").m_as("m")[2] == pytest.approx(30.48)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: _rocket_state(position=vector((0.0, 0.0, 0.0), "m/s")),
            "RocketState.position",
        ),
        (
            lambda: MassProperties(
                scalar(1.0, "kg"),
                vector((0.0, 0.0, 0.0), "m"),
                matrix(np.eye(3), "kg*m"),
            ),
            "MassProperties.inertia",
        ),
        (
            lambda: SimConfiguration(position_tolerance=scalar(1.0, "s")),
            "SimConfiguration.position_tolerance",
        ),
        (
            lambda: TruthConfiguration(launch_elevation=scalar(1.0, "s")),
            "TruthConfiguration.launch_elevation",
        ),
        (
            lambda: AeroTable(
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                np.zeros((2, 2, 2, 6)),
                reference_area=scalar(1.0, "m"),
                reference_length=scalar(1.0, "m"),
                reference_point=vector((0.0, 0.0, 0.0), "m"),
            ),
            "AeroTable.reference_area",
        ),
        (
            lambda: StateDerivative(
                velocity=vector((0.0, 0.0, 0.0), "m/s"),
                acceleration=vector((0.0, 0.0, 0.0), "m/s"),
                angular_acceleration=vector((0.0, 0.0, 0.0), "rad/s**2"),
                orientation_derivative=Quaternion(q_w=0.0),
            ),
            "StateDerivative.acceleration",
        ),
    ],
)
def test_wrong_dimensionality_is_rejected(
    factory: Callable[[], object], field_name: str
) -> None:
    """Every unit-checked dataclass names the offending field when units are wrong."""
    with pytest.raises(DimensionalityError, match=re.escape(field_name)):
        factory()


def test_equivalent_units_are_accepted() -> None:
    """Any unit of the right dimension is accepted, not just the default's."""
    state = _rocket_state(position=vector((0.0, 0.0, 100.0), "ft"))

    assert state.position[2].m_as("m") == pytest.approx(30.48)


def test_non_quantity_fields_are_ignored() -> None:
    """Fields whose defaults are not quantities are left alone by the check."""
    state = _rocket_state(orientation=Quaternion(q_w=0.5))

    assert state.orientation.q_w == pytest.approx(0.5)


def test_defaults_carry_expected_units() -> None:
    """The fields that keep a default are built in their declared units."""
    state = _rocket_state()

    assert state.position.check("[length]")
    assert state.velocity.check("[length] / [time]")
    assert state.angular_velocity.check("1 / [time]")


@dataclass
class _RequiredField(UnitChecked):
    """A required field declares its units through its annotation."""

    mass: Annotated[Scalar, "kg"]


def test_required_field_is_checked() -> None:
    """A required field with a unit annotation rejects the wrong dimension."""
    with pytest.raises(DimensionalityError, match=re.escape("_RequiredField.mass")):
        _RequiredField(scalar(1.0, "m"))


@dataclass
class _UnannotatedField(UnitChecked):
    """A field with no unit annotation declares no units."""

    mass: Scalar


def test_field_without_units_is_not_checked() -> None:
    """A field with no unit annotation is left alone, whatever it holds."""
    instance = _UnannotatedField(scalar(1.0, "m"))

    assert instance.mass.check("[length]")


@dataclass
class _OptionalField(UnitChecked):
    """An optional field declares its units through its annotation."""

    start: Annotated[Vector | None, "m"] = None


def test_optional_field_accepts_none_and_checks_quantities() -> None:
    """A unit-annotated field holding None is skipped; a quantity is checked."""
    assert _OptionalField().start is None
    assert _OptionalField(vector((1.0, 2.0, 3.0), "m")).start is not None
    with pytest.raises(DimensionalityError, match=re.escape("_OptionalField.start")):
        _OptionalField(vector((1.0, 2.0, 3.0), "s"))
