"""The layered wind, the aero table past 30 degrees, pitch damping and the tumble."""

import math
from dataclasses import replace

import numpy as np
import pytest

from flight_sim import tumble
from flight_sim.__main__ import (
    INVICTUS,
    get_default_config,
    get_launch_state,
    get_profile_properties,
)
from flight_sim.aero_extend import extend
from flight_sim.environment.wind import LayeredWind
from flight_sim.events import APOGEE
from flight_sim.integration import adaptive_step
from flight_sim.units import scalar, vector
from flight_sim.vehicle.rocket_properties import PitchDamping, TrapezoidFinSet


def test_layered_wind_is_the_base_on_the_pad_uneven_above_and_repeatable() -> None:
    """The rail sees the day's wind; higher up it wanders, the same every time."""
    wind = LayeredWind(
        speed=scalar(5.0, "m/s"), from_azimuth=scalar(270.0, "deg"), ground_m=100.0
    )
    again = LayeredWind(
        speed=scalar(5.0, "m/s"), from_azimuth=scalar(270.0, "deg"), ground_m=100.0
    )
    assert wind.velocity(100.0) == pytest.approx([0.0, 5.0, 0.0], abs=1e-9)
    heights = range(0, 6000, 50)
    speeds = [float(np.linalg.norm(wind.velocity(100.0 + h))) for h in heights]
    assert max(speeds) - min(speeds) > 1.0  # uneven through the flight
    assert all(s >= 0.0 for s in speeds)
    for h in (300.0, 1234.5, 5000.0):
        assert np.array_equal(wind.velocity(h), again.velocity(h))
        assert wind.velocity(h)[0] == 0.0  # horizontal


def test_extension_joins_the_table_and_turns_round_tail_first() -> None:
    """Normal force continues from the last row; axial force flips past 90 degrees."""
    alphas = np.arange(0.0, 31.0)
    cn = (
        12.0 * np.sin(np.radians(alphas)) * np.cos(np.radians(alphas))
        + 6.0 * np.sin(np.radians(alphas)) ** 2
    )
    ca = 0.5 * np.cos(np.radians(alphas))
    xcp = np.full_like(alphas, 25.0)
    added, cn_new, ca_new, xcp_new = extend(alphas, 0.3, cn, ca, xcp)
    assert added[0] == 40.0 and added[-1] == 180.0 and len(added) == 15
    assert cn_new[0] == pytest.approx(cn[-1], rel=0.35)
    assert all(c >= 0.0 for c in cn_new)
    assert cn_new[added.index(90.0)] > 0.5 * cn[-1]  # side-on: all crossflow
    assert ca_new[added.index(90.0)] == pytest.approx(0.0, abs=1e-9)
    assert ca_new[-1] < 0.0  # base first: the axial force points forward
    assert all(x > 0.0 for x in xcp_new)


def test_damping_opposes_a_pitch_rate_and_ignores_roll() -> None:
    """The torque is against the turn, also when the rocket is not moving."""
    damping = PitchDamping(body_length_m=5.0, body_diameter_m=0.15, fin_station_m=4.6)
    fins = TrapezoidFinSet(
        fin_count=4,
        root_chord_m=0.3,
        tip_chord_m=0.1,
        span_m=0.15,
        sweep_length_m=0.2,
        body_radius_m=0.075,
    )
    for speed in (0.0, 100.0):
        torque = damping.torque(
            np.array([3.0, 0.0, 1.0]),
            airspeed_m_s=speed,
            mach=speed / 340.0,
            air_density=1.0,
            cg_m=3.0,
            fins=fins,
        )
        assert torque[0] == 0.0 and torque[1] == 0.0 and torque[2] < 0.0
    still = damping.torque(
        np.zeros(3), airspeed_m_s=50.0, mach=0.15, air_density=1.0, cg_m=3.0, fins=fins
    )
    assert np.array_equal(still, np.zeros(3))


def _apogee() -> tuple[float, object, object, object]:
    properties = get_profile_properties(INVICTUS)
    state, config = get_launch_state(INVICTUS), get_default_config(INVICTUS)
    time, step = 0.0, scalar(0.01, "s")
    while True:
        state, taken, step, hit = adaptive_step(
            time, state, properties, config, step, events=(APOGEE,)
        )
        time += float(taken.m_as("s"))
        if hit is APOGEE:
            return time, state, properties, config


def test_the_kick_starts_a_tumble_that_the_air_then_damps() -> None:
    """Falling flat from a still apogee, the kick turns the rocket over and it slows."""
    time, state, properties, config = _apogee()
    still = replace(state, velocity=vector((0.0, 0.0, 0.0), "m/s"))
    flown = tumble.fly(
        time,
        still,
        properties,
        config,
        until_s=time + 6.0,
        fire_s=time + 0.5,
        separation_speed_m_s=8.0,
        nose_share=0.1,
        kick=tumble.SeparationKick(tipoff_deg_s=60.0),
    )
    assert flown[-1][0] == pytest.approx(time + 6.0)
    rates = [float(np.linalg.norm(s.angular_velocity.m_as("rad/s"))) for _, s in flown]
    kicked = next(i for i, (t, _) in enumerate(flown) if t >= time + 0.5)
    assert rates[kicked] > rates[kicked - 1] + math.radians(50.0)
    assert max(rates) < 5.0  # the air holds the turn to a tumble, not a spin-up
    assert len(tumble.resample(flown, 0.5)) <= 14
