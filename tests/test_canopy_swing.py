"""Tests for the two-body canopy model and the descent flown with it."""

# The rocket state helper repeats the one in test_descent
# pylint: disable=duplicate-code

import math
from itertools import pairwise

import numpy as np
import pytest

from flight_sim.canopy_swing import CanopySwing, Gusts
from flight_sim.descent import Parachute, RecoverySystem, simulate_descent
from flight_sim.environment.atmosphere import LaunchSiteAtmosphere
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.integration import IntegrationConfiguration
from flight_sim.swing_descent import simulate_swing_descent
from flight_sim.units import scalar, vector
from flight_sim.utilities.dcm import body_to_world
from flight_sim.vehicle.rocket_state import RocketState

_G = 9.80665
_DOWN = np.array([-1.0, 0.0, 0.0])
_UP_FORCE = np.array([1.0, 0.0, 0.0])


def test_drag_fraction_is_full_then_tapers_then_collapsed() -> None:
    """All of it inside the full-drag angle, the floor beyond the no-drag angle."""
    swing = CanopySwing(line_length_m=10.0)
    assert swing.drag_fraction(0.0) == 1.0
    assert swing.drag_fraction(math.radians(20.0)) == 1.0
    assert swing.drag_fraction(math.radians(75.0)) == pytest.approx(0.03)
    assert swing.drag_fraction(math.radians(170.0)) == pytest.approx(0.03)
    middle = swing.drag_fraction(math.radians(47.5))
    assert middle == pytest.approx(0.515)  # Half way down the cosine taper
    angles = np.radians(np.linspace(0.0, 180.0, 200))
    fractions = [swing.drag_fraction(float(a)) for a in angles]
    assert all(b <= a + 1e-12 for a, b in pairwise(fractions))


def test_canopy_inertia_adds_the_air_in_a_hemisphere() -> None:
    """The canopy's mass plus the air in a hemisphere of its diameter."""
    swing = CanopySwing(line_length_m=10.0, canopy_mass_kg=1.0)
    expected = 1.0 + 1.2 * math.pi / 12.0 * 3.0**3
    assert swing.canopy_inertia_kg(1.2, 3.0) == pytest.approx(expected)


def test_steady_hang_has_no_acceleration_and_carries_the_weight() -> None:
    """Drag balancing the weight leaves the line straight and the tension at M g."""
    swing = CanopySwing(line_length_m=10.0)
    mass = 40.0
    body_accel, line_accel, tension = swing.accelerations(
        mass,
        5.0,
        line=_DOWN,
        line_rate=np.zeros(3),
        canopy_force=mass * _G * _UP_FORCE,
        body_force=-mass * _G * _UP_FORCE,
    )
    assert body_accel == pytest.approx(np.zeros(3), abs=1e-9)
    assert line_accel == pytest.approx(np.zeros(3), abs=1e-9)
    assert tension == pytest.approx(mass * _G)


def test_heavy_canopy_limit_is_a_simple_pendulum() -> None:
    """With a fixed canopy the line swings back at g sin(theta) / L."""
    swing = CanopySwing(line_length_m=10.0)
    theta = math.radians(20.0)
    line = np.array([-math.cos(theta), math.sin(theta), 0.0])
    mass = 40.0
    body_accel, line_accel, tension = swing.accelerations(
        mass,
        1e9,
        line=line,
        line_rate=np.zeros(3),
        canopy_force=np.zeros(3),
        body_force=-mass * _G * _UP_FORCE,
    )
    assert float(np.linalg.norm(line_accel)) == pytest.approx(
        _G * math.sin(theta) / 10.0, rel=1e-6
    )
    assert float(line_accel @ line) == pytest.approx(0.0, abs=1e-9)
    assert tension == pytest.approx(mass * _G * math.cos(theta), rel=1e-6)
    # The line is rigid: the rocket's acceleration follows the swinging line
    assert body_accel == pytest.approx(10.0 * line_accel, abs=1e-6)


def test_gusts_are_repeatable_and_horizontal() -> None:
    """The same seed gives the same gusts, and none blows up or down."""
    swing = CanopySwing(line_length_m=10.0)
    first, second = Gusts(swing, 4.0), Gusts(swing, 4.0)
    values = [first.advance(0.1).copy() for _ in range(50)]
    again = [second.advance(0.1).copy() for _ in range(50)]
    assert np.array_equal(values, again)
    assert all(v[0] == 0.0 for v in values)
    assert max(float(np.linalg.norm(v)) for v in values) > 0.0


def _state(altitude_m: float, mass_kg: float = 40.0) -> RocketState:
    """A rocket at rest at an altitude."""
    return RocketState(
        current_mass=scalar(mass_kg, "kg"),
        inertia=vector((0.2, 90.0, 90.0), "kg*m**2"),
        cg_location=vector((-3.0, 0.0, 0.0), "m"),
        position=vector((altitude_m, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
    )


def _config(wind_m_s: float = 0.0) -> IntegrationConfiguration:
    """Standard day, constant gravity, wind blowing along +Y."""
    return IntegrationConfiguration(
        atmosphere=LaunchSiteAtmosphere(wind_m_s=np.array([0.0, wind_m_s, 0.0])),
        gravity=ConstantGravity(_G),
    )


_CANOPY = RecoverySystem((Parachute("main", 3.0, 1.5),), body_drag_area_m2=0.01)


def test_two_body_steady_descent_matches_the_point_mass() -> None:
    """In still air both models land at the same speed and time, near terminal."""
    config = _config()
    point = simulate_descent(0.0, _state(600.0), config, _CANOPY)
    both = simulate_swing_descent(
        0.0,
        _state(600.0),
        config,
        _CANOPY,
        swing=CanopySwing(line_length_m=8.0, turbulence=0.0),
    )
    assert both.descent.landed
    assert both.descent.times_s[-1] == pytest.approx(point.times_s[-1], rel=0.01)
    density = config.atmosphere.conditions(0.0).air_density
    rate = math.sqrt(2 * 40.0 * _G / (density * 1.5 * math.pi * 1.5**2))
    landing = -both.descent.states[-1].velocity.m_as("m/s")[0]
    assert landing == pytest.approx(rate, rel=0.03)
    # Still air and a straight hang: the line stays within a few degrees
    assert max(both.swing.angles_rad) < math.radians(5.0)
    assert both.swing.drag_fractions[-1] == 1.0


def test_rocket_hangs_nose_up_the_line() -> None:
    """The body axis shown follows the line once the canopy is out."""
    both = simulate_swing_descent(
        0.0, _state(600.0), _config(), _CANOPY, swing=CanopySwing(line_length_m=8.0)
    )
    last = both.descent.states[-1]
    axis = body_to_world(last.orientation)[:, 0]
    # It settles onto the line as a damped oscillation, so a few degrees behind
    assert float(axis @ -both.swing.line_directions[-1]) > math.cos(math.radians(6.0))


def test_a_collapsed_canopy_gives_almost_no_drag() -> None:
    """With the canopy always beyond the no-drag angle the fall is nearly free."""
    swing = CanopySwing(
        line_length_m=8.0,
        full_drag_deg=0.0,
        no_drag_deg=0.001,
        collapsed_drag_fraction=0.0,
    )
    fall = simulate_swing_descent(0.0, _state(300.0), _config(), _CANOPY, swing=swing)
    free = math.sqrt(2 * 300.0 / _G)
    assert fall.descent.times_s[-1] < 1.3 * free
    assert max(fall.swing.drag_fractions) <= 1.0
    assert fall.swing.drag_fractions[-1] == 0.0


def test_before_the_canopy_there_is_no_line() -> None:
    """The trace is zero until the release, and the same length as the samples."""
    delayed = RecoverySystem(
        (Parachute("main", 3.0, 1.5, deploy_delay_s=5.0),), body_drag_area_m2=0.01
    )
    both = simulate_swing_descent(
        0.0, _state(300.0), _config(), delayed, swing=CanopySwing(line_length_m=8.0)
    )
    assert len(both.swing.line_directions) == len(both.descent.times_s)
    assert float(np.linalg.norm(both.swing.line_directions[0])) == 0.0
    assert float(np.linalg.norm(both.swing.line_directions[-1])) == pytest.approx(1.0)
    assert both.swing.drag_fractions[0] == 0.0


def test_the_wind_swings_the_hang_a_little_and_gusts_repeat() -> None:
    """Turbulence in a steady wind swings the rocket a few degrees, repeatably."""
    swing = CanopySwing(line_length_m=8.0)
    config = _config(wind_m_s=5.0)
    first = simulate_swing_descent(0.0, _state(400.0), config, _CANOPY, swing=swing)
    second = simulate_swing_descent(0.0, _state(400.0), config, _CANOPY, swing=swing)
    # After the opening transient has died away
    settled = np.degrees(first.swing.angles_rad)[len(first.swing.angles_rad) // 2 :]
    assert 0.1 < float(settled.max()) < 30.0
    assert first.swing.angles_rad == second.swing.angles_rad
