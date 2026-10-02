"""Tests for picking Sol Invictus or Morpheus from the command line."""

import pytest

from flight_sim.__main__ import (
    INVICTUS,
    MORPHEUS,
    get_default_state,
    get_launch_state,
    parse_rocket,
    profile_for,
)


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
    assert table.launch_mass_kg - table.burnout_mass_kg == pytest.approx(
        MORPHEUS.propellant_mass_kg, abs=1e-3
    )
    assert get_default_state(MORPHEUS).current_mass.m_as("kg") == pytest.approx(28.902)
    assert get_default_state().current_mass.m_as("kg") == pytest.approx(65.19)


def test_launch_state_lies_along_the_profiles_rail() -> None:
    """Each rocket leans onto its own rail."""
    for profile in (INVICTUS, MORPHEUS):
        state = get_launch_state(profile)
        assert state.orientation == profile.rail.orientation()
