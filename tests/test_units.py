"""Unit-system tests: quantity construction and dataclass unit checking."""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.integration import IntegrationConfiguration, StateDerivative
from flight_sim.units import Scalar, UnitChecked, scalar, vector, zero_vector
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_state import RocketState


def _rocket_state(**overrides: Any) -> RocketState:
    """Return a valid RocketState with the given fields replaced."""
    state = RocketState(
        current_mass=scalar(25.0, "kg"),
        inertia=vector((150.0, 150.0, 2.5), "kg*m**2"),
        cg_location=vector((0.0, 0.0, -1.5), "m"),
    )
    return replace(state, **overrides)


def test_scalar_and_vector_constructors() -> None:
    """The constructors attach the requested units to the requested magnitudes."""
    assert scalar(500.0, "kg").m_as("kg") == pytest.approx(500.0)
    assert np.allclose(vector((1.0, 2.0, 3.0), "m").m_as("m"), [1.0, 2.0, 3.0])
    assert np.allclose(zero_vector("m/s").m_as("m/s"), np.zeros(3))


def test_quantities_convert_between_units() -> None:
    """A quantity built in one unit reads back correctly in a compatible one."""
    assert vector((0.0, 0.0, 100.0), "ft").m_as("m")[2] == pytest.approx(30.48)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (
            lambda: _rocket_state(position=zero_vector("m/s")),
            "RocketState.position",
        ),
        (
            lambda: _rocket_state(current_mass=scalar(1.0, "m")),
            "RocketState.current_mass",
        ),
        (
            lambda: _rocket_state(inertia=zero_vector("kg*m")),
            "RocketState.inertia",
        ),
        (
            lambda: IntegrationConfiguration(position_tolerance=scalar(1.0, "s")),
            "IntegrationConfiguration.position_tolerance",
        ),
        (
            lambda: StateDerivative(
                velocity=zero_vector("m/s"),
                acceleration=zero_vector("m/s"),
                angular_acceleration=zero_vector("rad/s**2"),
                orientation_derivative=Quaternion(q_w=0.0),
                mass_derivative=scalar(0.0, "kg/s"),
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
    assert state.reference_point.check("[length]")


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
