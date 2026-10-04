"""Tests for the OpenRocket reader and the profile built from it."""

import math
from pathlib import Path

import pytest
from ork_fixture import write_aero_csv, write_eng, write_ork

from flight_sim.ork import load_ork, mass_properties, saved_motor
from flight_sim.ork_profile import original_descent, pad_pressure_pa, profile_from_ork


@pytest.fixture(name="files")
def fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The synthetic .ork, aero CSV and .eng."""
    return (
        write_ork(tmp_path / "t.ork"),
        write_aero_csv(tmp_path / "t.csv"),
        write_eng(tmp_path / "t.eng"),
    )


def test_geometry_is_read_from_the_file(files: tuple[Path, Path, Path]) -> None:
    """Geometry is read from the file."""
    ork = load_ork(files[0])
    assert ork.name == "Test"
    assert ork.reference_diameter_m == pytest.approx(0.1)
    assert ork.nose_length_m == pytest.approx(0.5)
    assert ork.nose_shape == "haack"
    assert ork.length_m == pytest.approx(0.5 + 1.5 + 0.06)
    assert ork.tail_aft_radius_m == pytest.approx(0.035)
    assert ork.tail_fore_radius_m == pytest.approx(0.05)  # "auto 0.05"
    assert ork.fins.count == 4
    # Fins are placed from the aft end of their tube: trailing edge at the tube's end
    assert ork.fins.root_trailing_edge_x_m == pytest.approx(2.0)
    assert ork.fins.leading_edge_x_m == pytest.approx(1.75)


def test_component_masses_use_overrides_then_materials(
    files: tuple[Path, Path, Path],
) -> None:
    """Component masses use overrides then materials."""
    ork = load_ork(files[0])
    by_name = {item.name: item for item in ork.items}
    assert by_name["Nose"].mass_kg == pytest.approx(0.4)
    assert by_name["Nose"].overridden
    assert by_name["Payload"].mass_kg == pytest.approx(1.0)
    wall = 2700.0 * math.pi * (0.05**2 - 0.048**2) * 1.5
    assert by_name["Tube"].mass_kg == pytest.approx(wall)
    plate = 4 * 0.5 * (0.25 + 0.08) * 0.12 * 0.004 * 1780.0
    assert by_name["Fins"].mass_kg == pytest.approx(plate)


def test_saved_sim_is_read(files: tuple[Path, Path, Path]) -> None:
    """Saved sim is read."""
    ork = load_ork(files[0])
    sim = ork.sims["calm"]
    assert sim.summary["maxaltitude"] == pytest.approx(3000.0)
    assert sim.conditions["launchrodangle"] == pytest.approx(5.0)
    assert sim.conditions["basetemperature"] == pytest.approx(295.0)  # in <atmosphere>
    assert len(sim.series["Time"]) == 41


def test_motor_is_what_the_saved_mass_leaves(files: tuple[Path, Path, Path]) -> None:
    """Motor is what the saved mass leaves."""
    ork = load_ork(files[0])
    motor = saved_motor(ork, "calm")
    assert motor.propellant_kg == pytest.approx(4.0)
    assert motor.mass_kg == pytest.approx(16.3 - ork.airframe_mass_kg)
    liftoff, cg0, _ = mass_properties(ork, motor, 0.0)
    burnout, cg1, _ = mass_properties(ork, motor, 1.0)
    assert liftoff - burnout == pytest.approx(4.0)
    assert liftoff == pytest.approx(16.3)
    assert 0.0 < cg1 < ork.length_m and cg0 != cg1


def test_unknown_sim_is_an_error(files: tuple[Path, Path, Path]) -> None:
    """Unknown sim is an error."""
    with pytest.raises(KeyError):
        saved_motor(load_ork(files[0]), "nope")


def test_the_rocket_comes_down_under_the_original_descent(
    files: tuple[Path, Path, Path],
) -> None:
    """An ``.ork`` with no parachutes gets the Sol Invictus recovery for its size."""
    profile = profile_from_ork(files[0], str(files[1]), str(files[2]), sim="calm")
    recovery = profile.recovery
    assert [stage.name for stage in recovery.stages] == ["main reefed", "main reef cut"]
    assert recovery.body_drag_area_m2 == pytest.approx(0.55 * profile.reference_area_m2)
    assert original_descent(0.02).recovery.body_drag_area_m2 == pytest.approx(0.011)


def test_pad_pressure_falls_with_elevation() -> None:
    """Pad pressure falls with elevation."""
    assert pad_pressure_pa(101325.0, 288.15, 0.0) == pytest.approx(101325.0)
    assert pad_pressure_pa(101325.0, 288.15, 1000.0) < 91000.0


def test_profile_takes_conditions_from_the_saved_sim(
    files: tuple[Path, Path, Path],
) -> None:
    """Profile takes conditions from the saved sim."""
    profile = profile_from_ork(files[0], str(files[1]), str(files[2]), sim="calm")
    assert (
        profile.scheme is not None and profile.scheme.kind == "reefed single separation"
    )
    assert profile.reference_area_m2 == pytest.approx(math.pi * 0.05**2)
    assert profile.fins.fin_count == 4
    assert profile.rail.length.m_as("m") == pytest.approx(5.0)
    assert profile.rail.elevation.m_as("deg") == pytest.approx(85.0)
    assert profile.wind.speed.m_as("m/s") == pytest.approx(3.0)
    assert profile.pad_elevation_m == pytest.approx(100.0)
    assert profile.pad_temperature_k == pytest.approx(295.0)
    # the motor masses come from the saved run, so liftoff mass is OpenRocket's own
    assert profile.mass_properties.launch_mass_kg == pytest.approx(16.3)
    assert profile.mass_properties.propellant_mass_kg == pytest.approx(4.0)
