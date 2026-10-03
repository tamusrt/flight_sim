"""Engine model tests."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from flight_sim.units import Scalar, matrix, scalar, ureg, vector
from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import (
    PropellantGrain,
    SolidEngine,
    solid_engine_from_csv,
)
from flight_sim.vehicle.mass_properties import MassProperties, combine
from flight_sim.vehicle.rocket_properties import RocketProperties

_MOTOR_CSV = "tests/test_data/standard_motor.csv"

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


def _assert_same_mass_properties(engine: SolidEngine, other: SolidEngine) -> None:
    """Assert two engines match at paired times, the second ignited 2 s later."""
    for time in (0.0, 0.5, 2.0, 3.0, 4.0, 6.0):
        delayed = other.mass_properties(time + 2.0)
        expected = engine.mass_properties(time)
        assert other.get_thrust(time + 2.0) == pytest.approx(engine.get_thrust(time))
        assert delayed.mass == pytest.approx(expected.mass)
        assert delayed.cg_location == pytest.approx(expected.cg_location)
        assert delayed.inertia == pytest.approx(expected.inertia)


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


def test_unburned_grain_core_is_the_initial_core() -> None:
    """At ignition the axial inertia is that of the as-cast hollow cylinder."""
    properties = _GRAIN.mass_properties(0.0)

    assert properties.inertia[0, 0] == pytest.approx(
        0.5 * 5.0 * (_OUTER_RADIUS**2 + _CORE_RADIUS**2)
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mass": scalar(-1.0, "kg")}, "mass"),
        ({"length": scalar(0.0, "m")}, "length"),
        ({"cg_location": ureg.Quantity(np.zeros(2), "m")}, "3-vector"),
    ],
)
def test_grain_rejects_invalid_mass_length_and_cg(
    overrides: dict[str, Any], message: str
) -> None:
    """A grain has a non-negative mass, a positive length and a 3-vector CG."""
    with pytest.raises(ValueError, match=message):
        replace(_GRAIN, **overrides)


def test_burnout_leaves_only_the_casing() -> None:
    """Once the curve ends, the motor has burned exactly its propellant."""
    engine = _standard_engine()

    burned_out = engine.mass_properties(10.0)

    assert engine.burned_fraction(10.0) == 1.0
    assert engine.mass_properties(0.0).mass - burned_out.mass == pytest.approx(5.0)
    assert burned_out.mass == pytest.approx(3.0)
    assert burned_out.cg_location == pytest.approx([-2.5, 0.0, 0.0])
    assert burned_out.inertia == pytest.approx(np.diag((0.006, 0.2, 0.2)))


@pytest.mark.parametrize("time", [1.0, 2.0, 3.0, 3.9])
def test_mid_burn_mass_follows_the_impulse_delivered(time: float) -> None:
    """The mass matches integrating a mass flow of -thrust * m_prop / J_total."""
    engine = _standard_engine()
    times = np.linspace(0.0, time, 4001)
    thrust = np.array([engine.get_thrust(sample) for sample in times])
    burned = float(np.trapezoid(thrust, times)) * 5.0 / engine.total_impulse

    assert engine.mass_properties(time).mass == pytest.approx(8.0 - burned, rel=1e-7)


def test_solid_engine_without_impulse_burns_nothing() -> None:
    """A motor with no impulse keeps its propellant and gives no thrust."""
    engine = SolidEngine(
        times=np.array([0.0, 1.0]), thrusts=np.zeros(2), grain=_GRAIN, casing=_CASING
    )

    assert engine.total_impulse == 0.0
    assert engine.get_thrust(0.5) == 0.0
    assert engine.burned_fraction(5.0) == 0.0
    assert engine.mass_properties(5.0).mass == pytest.approx(8.0)


def test_solid_engine_from_csv_integrates_the_thrust_curve() -> None:
    """The total impulse is the area under the CSV thrust curve."""
    engine = _standard_engine()

    # 2000 N for 2 s, then a linear tail-off to zero over 2 s
    assert engine.total_impulse == pytest.approx(6000.0)
    assert engine.get_thrust(1.0) == pytest.approx(2000.0)
    assert engine.get_thrust(3.0) == pytest.approx(1000.0)
    assert engine.get_thrust(-0.1) == 0.0
    assert engine.get_thrust(4.1) == 0.0
    assert engine.burned_fraction(3.0) == pytest.approx(5500.0 / 6000.0)


@pytest.mark.parametrize("ignition_time", [scalar(2.0, "s"), scalar(2000.0, "ms")])
def test_late_ignition_delays_the_burn(ignition_time: Scalar) -> None:
    """A motor lit at 2 s holds its propellant until then, then burns as usual."""
    engine = _standard_engine()
    delayed = replace(engine, ignition_time=ignition_time)

    assert delayed.get_thrust(1.0) == 0.0
    assert delayed.burned_fraction(1.0) == 0.0
    assert delayed.mass_properties(1.0).mass == pytest.approx(8.0)
    _assert_same_mass_properties(engine, delayed)


@pytest.mark.parametrize(
    ("times", "thrusts", "message"),
    [
        ([0.0], [1.0], "two samples"),
        ([0.0, 1.0], [1.0, 2.0, 3.0], "matching"),
        ([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], "increasing"),
        ([0.0, 1.0], [1.0, -2.0], "negative"),
    ],
)
def test_solid_engine_rejects_a_malformed_thrust_curve(
    times: list[float], thrusts: list[float], message: str
) -> None:
    """The thrust curve needs matching, increasing samples of non-negative thrust."""
    with pytest.raises(ValueError, match=message):
        SolidEngine(
            times=np.array(times),
            thrusts=np.array(thrusts),
            grain=_GRAIN,
            casing=_CASING,
        )


def test_solid_engine_rejects_a_massless_casing() -> None:
    """A burned-out motor still needs mass."""
    casing = replace(_CASING, mass=scalar(0.0, "kg"))

    with pytest.raises(ValueError, match="Casing mass"):
        solid_engine_from_csv(_MOTOR_CSV, _GRAIN, casing)


def test_rocket_mass_properties_combine_dry_and_engine(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The rocket is its dry parts plus the motor at the same moment."""
    aero_table: AeroTable = baseline_rocket_properties.aero_table
    dry = MassProperties(
        scalar(17.0, "kg"),
        vector((-1.0, 0.0, 0.0), "m"),
        matrix(np.diag((2.4, 135.0, 135.0)), "kg*m**2"),
    )
    engine = _standard_engine()
    properties = RocketProperties(aero_table, engine, dry)

    for time in (0.0, 3.0, 10.0):
        combined = properties.mass_properties(time)
        expected = combine(dry.si, engine.mass_properties(time))
        assert combined.mass == pytest.approx(expected.mass)
        assert combined.cg_location == pytest.approx(expected.cg_location)
        assert combined.inertia == pytest.approx(expected.inertia)

    # 17 kg at -1 m and 8 kg of motor near -2.56 m
    assert properties.mass_properties(0.0).mass == pytest.approx(25.0)
    assert properties.mass_properties(0.0).cg_location[0] == pytest.approx(
        (17.0 * -1.0 + 3.0 * -2.5 + 5.0 * -2.6) / 25.0
    )
