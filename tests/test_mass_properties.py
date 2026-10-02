"""Tests for the mass property interpolation through the burn."""

import pytest

from flight_sim.units import scalar, vector
from flight_sim.vehicle.mass_properties import MassPropertiesTable
from flight_sim.vehicle.rocket_state import RocketState

_TABLE = MassPropertiesTable(
    launch_mass_kg=60.0,
    launch_cg_m=-3.3,
    launch_inertia_kg_m2=(0.24, 97.0, 97.0),
    burnout_mass_kg=40.0,
    burnout_cg_m=-3.1,
    burnout_inertia_kg_m2=(0.18, 83.0, 83.0),
)


def _state(mass_kg: float) -> RocketState:
    return RocketState(
        current_mass=scalar(mass_kg, "kg"),
        inertia=vector((1.0, 1.0, 1.0), "kg*m**2"),
        cg_location=vector((0.0, 0.0, 0.0), "m"),
    )


@pytest.mark.parametrize(
    ("mass_kg", "cg_m", "inertia"),
    [
        (60.0, -3.3, (0.24, 97.0, 97.0)),
        (50.0, -3.2, (0.21, 90.0, 90.0)),
        (40.0, -3.1, (0.18, 83.0, 83.0)),
        (65.0, -3.3, (0.24, 97.0, 97.0)),  # Above launch mass: held at launch
        (35.0, -3.1, (0.18, 83.0, 83.0)),  # Below burnout: held at burnout
    ],
)
def test_cg_and_inertia_follow_the_mass(
    mass_kg: float, cg_m: float, inertia: tuple[float, float, float]
) -> None:
    """Values interpolate linearly in mass and hold outside the burn."""
    state = _state(mass_kg)
    _TABLE.apply(state)
    assert state.cg_location.m_as("m") == pytest.approx([cg_m, 0.0, 0.0])
    assert state.inertia.m_as("kg*m**2") == pytest.approx(inertia)


def test_no_propellant_counts_as_burnt_out() -> None:
    """A table with no mass change reports the burnout values."""
    table = MassPropertiesTable(40.0, -3.0, (1.0, 2.0, 2.0), 40.0, -2.9, (1, 1, 1))
    assert table.burned_fraction(40.0) == 1.0
