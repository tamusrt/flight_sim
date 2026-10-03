"""Tests for picking Sol Invictus or Morpheus from the command line."""

import numpy as np
import pytest

from flight_sim.__main__ import (
    INVICTUS,
    MORPHEUS,
    RocketProfile,
    get_launch_state,
    get_profile_properties,
    parse_rocket,
    profile_for,
)
from flight_sim.vehicle.engine import SolidEngine


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data/ras_alpha_files/invictus_aero.csv", INVICTUS),
        ("data/ras_alpha_files/morpheus_aero.csv", MORPHEUS),
        ("C:/aero/MORPH_run2.csv", MORPHEUS),
        ("data/morph_dir/invictus_aero.csv", INVICTUS),  # only the file name counts
    ],
)
def test_the_aero_file_name_picks_the_rocket(path: str, expected: object) -> None:
    """A name containing "morph" means Morpheus; any other means Invictus."""
    assert profile_for(path) is expected


def test_no_flags_fly_invictus_and_morpheus_flag_flies_morpheus() -> None:
    """The default rocket is Invictus; --morpheus adds Morpheus and its aero CSV."""
    assert parse_rocket(()) == (INVICTUS, INVICTUS.aero_file)
    assert parse_rocket(("--morpheus",)) == (MORPHEUS, MORPHEUS.aero_file)
    assert parse_rocket(("--aero", "x/morph_a.csv")) == (MORPHEUS, "x/morph_a.csv")
    assert parse_rocket(("--aero", "x/other.csv")) == (INVICTUS, "x/other.csv")


def test_morpheus_mass_drop_is_its_propellant_and_it_starts_lighter() -> None:
    """The O3400 burns 11.272 kg, the whole drop from launch to burnout mass."""
    table = MORPHEUS.mass_properties
    assert table.propellant_mass_kg == pytest.approx(11.272, abs=1e-3)
    assert table.launch_mass_kg == pytest.approx(28.902)
    assert INVICTUS.mass_properties.launch_mass_kg == pytest.approx(65.19)


@pytest.mark.parametrize("profile", [INVICTUS, MORPHEUS])
def test_the_flown_rocket_weighs_what_the_profile_says_through_the_burn(
    profile: RocketProfile,
) -> None:
    """At ignition and after burnout the engine model gives the profile's numbers."""
    table = profile.mass_properties
    properties = get_profile_properties(profile)
    engine = properties.engine
    assert isinstance(engine, SolidEngine)
    burn_s = engine.times[-1]  # pylint: disable=no-member

    launch = properties.mass_properties(0.0)
    assert launch.mass == pytest.approx(table.launch_mass_kg)
    assert launch.cg_location == pytest.approx([table.launch_cg_m, 0.0, 0.0])
    assert launch.inertia[1, 1] == pytest.approx(table.launch_inertia_kg_m2[1])

    burnout = properties.mass_properties(burn_s + 1.0)
    assert burnout.mass == pytest.approx(table.burnout_mass_kg)
    assert burnout.cg_location == pytest.approx([table.burnout_cg_m, 0.0, 0.0])
    assert burnout.inertia == pytest.approx(np.diag(table.burnout_inertia_kg_m2))

    middle = properties.mass_properties(0.5 * burn_s)
    assert table.burnout_mass_kg < middle.mass < table.launch_mass_kg


def test_launch_state_lies_along_the_profiles_rail() -> None:
    """Each rocket leans onto its own rail."""
    for profile in (INVICTUS, MORPHEUS):
        state = get_launch_state(profile)
        assert state.orientation == profile.rail.orientation()
