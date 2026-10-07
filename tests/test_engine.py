"""Engine model tests."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flight_sim.units import matrix, scalar, vector
from flight_sim.vehicle.engine import (
    PropellantGrain,
    SolidEngine,
    solid_engine_from_csv,
)
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties

_MOTOR_CSV = "tests/test_data/standard_motor.csv"
_MOTOR_ENG = "tests/test_data/standard_motor.eng"
_MOTOR_RSE = "tests/test_data/standard_motor.rse"

_SEA_LEVEL_PA = 101325.0

# Grain dimensions in m
_OUTER_RADIUS, _CORE_RADIUS, _LENGTH = 0.04, 0.015, 0.75

_GRAIN = PropellantGrain(
    mass=scalar(5.0, "kg"),
    length=scalar(_LENGTH, "m"),
    outer_diameter=scalar(2 * _OUTER_RADIUS, "m"),
    core_diameter=scalar(2 * _CORE_RADIUS, "m"),
    cg_location=vector((-2.6, 0.0, 0.0), "m"),
)

_CASING = MassProperties(
    mass=scalar(3.0, "kg"),
    cg_location=vector((-2.5, 0.0, 0.0), "m"),
    inertia=matrix(np.diag((0.006, 0.2, 0.2)), "kg*m**2"),
)


def _standard_engine() -> SolidEngine:
    """Return the standard motor: 2000 N for 2 s, then a linear tail-off."""
    return solid_engine_from_csv(_MOTOR_CSV, _GRAIN, _CASING)


@pytest.mark.parametrize("burned_fraction", [0.0, 0.5, 1.0])
def test_grain_burns_out_from_its_core(burned_fraction: float) -> None:
    """The grain thins radially as a hollow cylinder of fixed length and CG."""
    remaining = 1.0 - burned_fraction
    mass = 5.0 * remaining
    inner_sq = _OUTER_RADIUS**2 - remaining * (_OUTER_RADIUS**2 - _CORE_RADIUS**2)
    radii_sq = _OUTER_RADIUS**2 + inner_sq
    axial = 0.5 * mass * radii_sq
    transverse = mass * (3 * radii_sq + _LENGTH**2) / 12

    properties = _GRAIN.mass_properties(burned_fraction)

    assert properties.mass == pytest.approx(mass)
    assert properties.cg_location == pytest.approx([-2.6, 0.0, 0.0])
    assert properties.inertia == pytest.approx(
        np.diag((axial, transverse, transverse)), abs=1e-15
    )


@pytest.mark.parametrize(
    ("outer", "core", "message"),
    [(0.08, 0.08, "core diameter"), (0.08, -0.01, "core diameter")],
)
def test_grain_rejects_a_core_outside_the_grain(
    outer: float, core: float, message: str
) -> None:
    """The core must be at least zero and narrower than the grain."""
    with pytest.raises(ValueError, match=message):
        replace(
            _GRAIN,
            outer_diameter=scalar(outer, "m"),
            core_diameter=scalar(core, "m"),
        )


def test_burnout_leaves_only_the_casing() -> None:
    """Once the curve ends, the motor has burned exactly its propellant."""
    engine = _standard_engine()

    burned_out = engine.mass_properties(10.0)

    assert engine.burned_fraction(10.0) == 1.0
    assert engine.mass_properties(0.0).mass - burned_out.mass == pytest.approx(5.0)
    assert burned_out.mass == pytest.approx(3.0)
    assert burned_out.cg_location == pytest.approx([-2.5, 0.0, 0.0])
    assert burned_out.inertia == pytest.approx(np.diag((0.006, 0.2, 0.2)))


def test_mid_burn_mass_follows_the_impulse_delivered() -> None:
    """The mass matches integrating a mass flow of -thrust * m_prop / J_total."""
    engine = _standard_engine()
    time = 3.0  # Within the tail-off
    times = np.linspace(0.0, time, 4001)
    thrust = np.array([engine.get_thrust(sample, _SEA_LEVEL_PA) for sample in times])
    burned = float(np.trapezoid(thrust, times)) * 5.0 / engine.total_impulse

    assert engine.mass_properties(time).mass == pytest.approx(8.0 - burned, rel=1e-7)


def test_solid_engine_from_csv_integrates_the_thrust_curve() -> None:
    """The total impulse is the area under the CSV thrust curve."""
    engine = _standard_engine()

    # 2000 N for 2 s, then a linear tail-off to zero over 2 s
    assert engine.total_impulse == pytest.approx(6000.0)
    assert engine.get_thrust(1.0, _SEA_LEVEL_PA) == pytest.approx(2000.0)
    assert engine.get_thrust(3.0, _SEA_LEVEL_PA) == pytest.approx(1000.0)
    assert engine.get_thrust(-0.1, _SEA_LEVEL_PA) == 0.0
    assert engine.get_thrust(4.1, _SEA_LEVEL_PA) == 0.0
    assert engine.burned_fraction(3.0) == pytest.approx(5500.0 / 6000.0)


def test_solid_engine_from_eng_starts_the_curve_at_zero() -> None:
    """The first motor's curve is read, ramping up from an implicit (0, 0)."""
    engine = SolidEngine.from_eng(_MOTOR_ENG, _GRAIN, _CASING)

    # 100 N*s of ramp, 3800 N*s of hold and 2000 N*s of tail-off
    assert engine.total_impulse == pytest.approx(5900.0)
    assert engine.get_thrust(0.05, _SEA_LEVEL_PA) == pytest.approx(1000.0)
    assert engine.get_thrust(3.0, _SEA_LEVEL_PA) == pytest.approx(1000.0)


def test_solid_engine_from_rse_reads_the_first_motor() -> None:
    """The first motor's curve matches the standard CSV curve."""
    engine = SolidEngine.from_rse(_MOTOR_RSE, _GRAIN, _CASING)

    assert engine.times == pytest.approx([0.0, 2.0, 4.0])
    assert engine.thrusts == pytest.approx([2000.0, 2000.0, 0.0])


@pytest.mark.parametrize(
    ("loader", "suffix", "contents"),
    [
        (SolidEngine.from_eng, ".eng", "; Only a comment\n"),
        (SolidEngine.from_rse, ".rse", "<engine-database/>"),
    ],
)
def test_solid_engine_rejects_a_motor_file_without_a_curve(
    loader: Callable[[str, PropellantGrain, MassProperties], SolidEngine],
    suffix: str,
    contents: str,
    tmp_path: Path,
) -> None:
    """A motor file with no thrust samples cannot build a motor."""
    motor_file = tmp_path / f"empty{suffix}"
    motor_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="no thrust curve"):
        loader(str(motor_file), _GRAIN, _CASING)


def test_late_ignition_delays_the_burn() -> None:
    """A motor lit at 2 s holds its propellant until then, then burns as usual."""
    engine = _standard_engine()
    delayed = replace(engine, ignition_time=scalar(2.0, "s"))

    assert delayed.get_thrust(1.0, _SEA_LEVEL_PA) == 0.0
    assert delayed.mass_properties(1.0).mass == pytest.approx(8.0)
    for time in (0.5, 3.0, 6.0):
        expected = engine.mass_properties(time)
        actual = delayed.mass_properties(time + 2.0)
        assert delayed.get_thrust(time + 2.0, _SEA_LEVEL_PA) == pytest.approx(
            engine.get_thrust(time, _SEA_LEVEL_PA)
        )
        assert actual.mass == pytest.approx(expected.mass)
        assert actual.inertia == pytest.approx(expected.inertia)


@pytest.mark.parametrize(
    ("time", "ambient_pressure", "expected"),
    [
        (1.0, _SEA_LEVEL_PA, 2000.0),  # At the reference pressure
        (1.0, 0.0, 2000.0 + _SEA_LEVEL_PA * 0.002),  # Vacuum
        (3.0, 50000.0, 1000.0 + (_SEA_LEVEL_PA - 50000.0) * 0.002),
        (4.0, 0.0, _SEA_LEVEL_PA * 0.002),  # Last sample, still burning
        (4.1, 0.0, 0.0),  # Burned out
        (-0.1, 0.0, 0.0),  # Before ignition
    ],
)
def test_pressure_thrust_adds_to_the_curve_during_the_burn(
    time: float, ambient_pressure: float, expected: float
) -> None:
    """The nozzle adds (reference - ambient pressure) * exit area while burning."""
    engine = replace(_standard_engine(), nozzle_exit_area=scalar(20.0, "cm**2"))

    assert engine.get_thrust(time, ambient_pressure) == pytest.approx(expected)


def test_pressure_thrust_leaves_the_burn_rate_unchanged() -> None:
    """Propellant still burns in proportion to the curve's impulse."""
    engine = _standard_engine()
    nozzled = replace(engine, nozzle_exit_area=scalar(20.0, "cm**2"))

    assert nozzled.total_impulse == pytest.approx(engine.total_impulse)
    assert nozzled.burned_fraction(3.0) == pytest.approx(engine.burned_fraction(3.0))


def test_overexpanded_thrust_is_never_negative() -> None:
    """Ambient pressure far above the reference cannot pull the rocket back."""
    engine = replace(
        _standard_engine(),
        nozzle_exit_area=scalar(20.0, "cm**2"),
        reference_pressure=scalar(0.0, "Pa"),
    )

    assert engine.get_thrust(3.9, _SEA_LEVEL_PA) == 0.0


def test_solid_engine_rejects_a_negative_nozzle_exit_area() -> None:
    """The nozzle exit area must be at least zero."""
    with pytest.raises(ValueError, match="exit area"):
        replace(_standard_engine(), nozzle_exit_area=scalar(-1.0, "cm**2"))


@pytest.mark.parametrize(
    ("times", "thrusts", "message"),
    [
        ([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], "increasing"),
        ([0.0, 1.0], [0.0, 0.0], "impulse"),
    ],
)
def test_solid_engine_rejects_a_malformed_thrust_curve(
    times: list[float], thrusts: list[float], message: str
) -> None:
    """The thrust curve needs increasing sample times and some impulse."""
    with pytest.raises(ValueError, match=message):
        SolidEngine(
            times=np.array(times),
            thrusts=np.array(thrusts),
            grain=_GRAIN,
            casing=_CASING,
        )


def test_rocket_mass_properties_combine_dry_and_engine(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The rocket is its dry parts plus the motor at the same moment."""
    dry = MassProperties(
        scalar(17.0, "kg"),
        vector((-1.0, 0.0, 0.0), "m"),
        matrix(np.diag((2.4, 135.0, 135.0)), "kg*m**2"),
    )
    properties = RocketProperties(
        baseline_rocket_properties.aero_table, _standard_engine(), dry
    )

    at_ignition = properties.mass_properties(0.0)

    # 17 kg at -1 m, 3 kg of casing at -2.5 m and 5 kg of grain at -2.6 m
    assert at_ignition.mass == pytest.approx(25.0)
    assert at_ignition.cg_location[0] == pytest.approx(
        (17.0 * -1.0 + 3.0 * -2.5 + 5.0 * -2.6) / 25.0
    )
