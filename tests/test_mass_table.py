"""Tests for turning ignition and burnout numbers into engine parts."""

import numpy as np
import pytest

from flight_sim.vehicle.mass_properties import combine
from flight_sim.vehicle.mass_table import MassPropertiesTable

_TABLE = MassPropertiesTable(
    launch_mass_kg=60.0,
    launch_cg_m=-3.3,
    launch_inertia_kg_m2=(0.24, 97.0, 97.0),
    burnout_mass_kg=40.0,
    burnout_cg_m=-3.1,
    burnout_inertia_kg_m2=(0.18, 83.0, 83.0),
)
_DIAMETER_M = 0.152


def test_propellant_is_the_drop_in_mass() -> None:
    """What burns off is what the rocket loses."""
    assert _TABLE.propellant_mass_kg == pytest.approx(20.0)


def test_parts_add_up_to_the_rocket_at_ignition_and_at_burnout() -> None:
    """Mass, CG and pitch inertia come out as given at both moments."""
    dry, casing, grain = _TABLE.parts(_DIAMETER_M)

    launch = combine(dry.si, casing.si, grain.mass_properties(0.0))
    assert launch.mass == pytest.approx(60.0)
    assert launch.cg_location == pytest.approx([-3.3, 0.0, 0.0])
    assert launch.inertia[1, 1] == pytest.approx(97.0)
    assert launch.inertia[2, 2] == pytest.approx(97.0)

    burnout = combine(dry.si, casing.si, grain.mass_properties(1.0))
    assert burnout.mass == pytest.approx(40.0)
    assert burnout.cg_location == pytest.approx([-3.1, 0.0, 0.0])
    assert burnout.inertia == pytest.approx(np.diag((0.18, 83.0, 83.0)))


def test_grain_sits_where_the_propellant_was() -> None:
    """The grain's CG balances the two CGs: 40 kg at -3.1 m and 20 kg at x make -3.3."""
    grain = _TABLE.parts(_DIAMETER_M)[2]
    assert grain.cg_location.m_as("m") == pytest.approx([-3.7, 0.0, 0.0])
    assert grain.mass.m_as("kg") == pytest.approx(20.0)
    assert grain.outer_diameter.m_as("m") == pytest.approx(_DIAMETER_M)


def test_the_rocket_gets_lighter_and_its_cg_moves_forward_through_the_burn() -> None:
    """Part way through the burn the numbers are between the two ends."""
    dry, casing, grain = _TABLE.parts(_DIAMETER_M)
    middle = combine(dry.si, casing.si, grain.mass_properties(0.5))
    assert middle.mass == pytest.approx(50.0)
    assert -3.3 < middle.cg_location[0] < -3.1
    assert 83.0 < middle.inertia[1, 1] < 97.0


def test_a_roll_inertia_a_grain_cannot_have_is_matched_as_closely_as_it_can() -> None:
    """More roll inertia than the diameter allows gives a thin-walled grain."""
    heavy = MassPropertiesTable(
        60.0, -3.3, (5.0, 97.0, 97.0), 40.0, -3.1, (0.18, 83.0, 83.0)
    )
    grain = heavy.parts(_DIAMETER_M)[2]
    assert 0.0 <= grain.core_diameter.m_as("m") < _DIAMETER_M
    light = MassPropertiesTable(
        60.0, -3.3, (0.18, 97.0, 97.0), 40.0, -3.1, (0.18, 83.0, 83.0)
    )
    assert light.parts(_DIAMETER_M)[2].core_diameter.m_as("m") == 0.0


@pytest.mark.parametrize("launch_mass", [40.0, 30.0])
def test_a_rocket_that_does_not_lose_mass_is_rejected(launch_mass: float) -> None:
    """With nothing to burn there is no grain to make."""
    table = MassPropertiesTable(
        launch_mass, -3.3, (0.24, 97.0, 97.0), 40.0, -3.1, (0.18, 83.0, 83.0)
    )
    with pytest.raises(ValueError, match="lose mass"):
        table.parts(_DIAMETER_M)
