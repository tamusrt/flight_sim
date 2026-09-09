"""Unit-system tests: quantity construction and dataclass unit checking."""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.environment.atmosphere import AtmosphereData
from flight_sim.integration import IntegrationConfiguration, StateDerivative
from flight_sim.units import Scalar, UnitChecked, scalar, vector, zero_vector
from flight_sim.vehicle.rocket_state import Quaternion, RocketState


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
        (lambda: RocketState(position=zero_vector("m/s")), "RocketState.position"),
        (
            lambda: RocketState(current_mass=scalar(1.0, "m")),
            "RocketState.current_mass",
        ),
        (
            lambda: AtmosphereData(air_density=scalar(1.0, "Pa")),
            "AtmosphereData.air_density",
        ),
        (
            lambda: StateDerivative(acceleration=zero_vector("m/s")),
            "StateDerivative.acceleration",
        ),
        (
            lambda: IntegrationConfiguration(time_step=scalar(1.0, "kg")),
            "IntegrationConfiguration.time_step",
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
    state = RocketState(position=vector((0.0, 0.0, 100.0), "ft"))

    assert state.position[2].m_as("m") == pytest.approx(30.48)


def test_non_quantity_fields_are_ignored() -> None:
    """Fields whose defaults are not quantities are left alone by the check."""
    state = RocketState(orientation=Quaternion(q_w=0.5))

    assert state.orientation.q_w == pytest.approx(0.5)


def test_defaults_carry_expected_units() -> None:
    """The default state and configuration are built in the documented units."""
    assert RocketState().position.check("[length]")
    assert RocketState().current_mass.check("[mass]")
    assert AtmosphereData().pressure.check("[pressure]")
    assert IntegrationConfiguration().time_step.m_as("s") == pytest.approx(0.01)
    assert IntegrationConfiguration().max_time.m_as("s") == pytest.approx(100.0)


@dataclass
class _RequiredField(UnitChecked):
    """A field with no default declares no units, so it cannot be checked."""

    mass: Scalar


def test_field_without_a_default_is_not_checked() -> None:
    """A required field declares no units, so nothing constrains it."""
    instance = _RequiredField(scalar(1.0, "m"))

    assert instance.mass.check("[length]")
