"""Tests for reading the motor files the propulsion model writes."""

from pathlib import Path

import pytest
from ork_fixture import write_aero_csv, write_eng, write_ork, write_rse

from flight_sim.motor_file import eng_for_sim, load_motor
from flight_sim.ork import load_ork, saved_motor
from flight_sim.ork_profile import motor_with_file, profile_from_ork


def test_eng_with_comments_is_read(tmp_path: Path) -> None:
    """Comment lines, the description line and a curve that starts after zero."""
    path = tmp_path / "m.eng"
    path.write_text(
        "; first comment\n; ENGINE\nENGINE 152.4 2617.7 0 22.1 42.7 TAMU ; note\n"
        "0.05 3600\n0.10 3600 ; inline\n10.0 0\n;\n",
        encoding="utf-8",
    )
    motor = load_motor(path)
    assert motor.name == "ENGINE" and motor.maker == "TAMU"
    assert motor.propellant_kg == pytest.approx(22.1)
    assert motor.total_kg == pytest.approx(42.7)
    assert motor.thrust[0] == [0.0, 0.0] and motor.thrust[1] == [0.05, 3600.0]
    assert motor.burn_s == pytest.approx(10.0) and motor.peak_n == 3600.0
    # ramp, plateau, then a slow fall to zero at 10 s
    assert motor.impulse_ns == pytest.approx(90 + 180 + 17820)


def test_rse_gives_the_same_numbers_in_kilograms(tmp_path: Path) -> None:
    """RockSim grams become kilograms and the samples start from zero thrust."""
    motor = load_motor(write_rse(tmp_path / "m.rse"))
    assert motor.name == "TEST-RSE"
    assert motor.propellant_kg == pytest.approx(
        3.5
    ) and motor.total_kg == pytest.approx(9.0)
    assert motor.thrust[0] == [0.0, 0.0] and motor.thrust[-1] == [4.0, 0.0]
    assert motor.impulse_ns == pytest.approx(
        800 * 3.98 + 0.5 * 800 * 0.01 + 0.5 * 800 * 0.01
    )


def test_rse_is_converted_to_an_eng_for_the_sim(tmp_path: Path) -> None:
    """The sim reads ``.eng``; an ``.rse`` becomes one, an ``.eng`` is used as it is."""
    eng = write_eng(tmp_path / "m.eng")
    assert eng_for_sim(eng, tmp_path) == eng
    made = eng_for_sim(write_rse(tmp_path / "r.rse"), tmp_path)
    assert made.suffix == ".eng"
    again = load_motor(made)
    assert again.propellant_kg == pytest.approx(3.5) and again.thrust[-1] == [4.0, 0.0]


def test_bad_files_are_refused(tmp_path: Path) -> None:
    """No samples, times that go backwards and unknown suffixes are errors."""
    empty = tmp_path / "e.eng"
    empty.write_text("NAME 1 1 0 1 2 X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no thrust samples"):
        load_motor(empty)
    back = tmp_path / "b.eng"
    back.write_text("NAME 1 1 0 1 2 X\n1 5\n0.5 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not increase"):
        load_motor(back)
    with pytest.raises(ValueError, match=r"\.eng or \.rse"):
        load_motor(tmp_path / "m.csv")


def test_the_ork_masses_are_flown_with_the_files_thrust(tmp_path: Path) -> None:
    """The motor masses come from the .ork's saved simulation, not the motor file."""
    ork = load_ork(write_ork(tmp_path / "t.ork"))
    saved = saved_motor(ork)
    motor = motor_with_file(saved, load_motor(write_rse(tmp_path / "m.rse")))
    assert motor.propellant_kg == pytest.approx(saved.propellant_kg)
    assert motor.mass_kg == pytest.approx(saved.mass_kg)
    assert (motor.x_m, motor.length_m) == (saved.x_m, saved.length_m)
    profile = profile_from_ork(
        write_ork(tmp_path / "t.ork"),
        str(write_aero_csv(tmp_path / "a.csv")),
        str(tmp_path / "m.rse"),
    )
    props = profile.mass_properties
    assert props.propellant_mass_kg == pytest.approx(saved.propellant_kg)
    assert profile.motor_file.endswith(".eng")
