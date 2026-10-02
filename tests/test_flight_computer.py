"""Tests for the flight computer's view of the flight."""

import math
from dataclasses import replace

import numpy as np
import pytest

from flight_sim.__main__ import get_default_properties
from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import Parachute, RecoverySystem, ReefedParachute
from flight_sim.environment.atmosphere import LaunchSiteAtmosphere
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.flight_computer import FlightComputer, plan_recovery
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_extension import (
    EJECTION_CHARGE,
    canopy_swing_for,
    frames,
    log_events,
)
from flight_sim.recovery_motion import CORD_TO_BODY_M, EjectionCharge
from flight_sim.units import scalar, vector
from flight_sim.vehicle.rocket_state import RocketState
from flight_sim.visualize import TelemetryLog

_G = 9.80665


def _ballistic(apogee_s: float, up_to_s: float) -> list[tuple[float, RocketState]]:
    """A vertical vacuum arc peaking at apogee_s, sampled every 0.05 s."""
    samples = []
    for t in np.arange(0.0, up_to_s + 1e-9, 0.05):
        height = _G * apogee_s * t - 0.5 * _G * t**2
        samples.append(
            (
                float(t),
                RocketState(
                    current_mass=scalar(40.0, "kg"),
                    inertia=vector((0.2, 80.0, 80.0), "kg*m**2"),
                    cg_location=vector((-3.0, 0.0, 0.0), "m"),
                    position=vector((height, 0.0, 0.0), "m"),
                    velocity=vector((_G * (apogee_s - t), 0.0, 0.0), "m/s"),
                ),
            )
        )
    return samples


def test_apogee_is_detected_shortly_after_the_true_apogee() -> None:
    """Filter lag, the pressure window and accelerometer bias make it late."""
    computer = FlightComputer()
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    detection = computer.detect_apogee(computer.sense(_ballistic(20.0, 26.0), config))

    assert detection.detected_s is not None
    assert 20.0 < detection.detected_s < 21.5
    # Bias reads the velocity high, so its vote comes a little after apogee
    assert detection.velocity_vote_s == pytest.approx(20.0 + 0.05 * 20.0 / _G, abs=0.05)
    # The nose stays up in this test, so the attitude vote never agrees
    assert detection.attitude_vote_s is None


def test_lockout_ignores_the_boost() -> None:
    """Nothing is counted before the lockout ends."""
    computer = FlightComputer(lockout_s=30.0)
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    detection = computer.detect_apogee(computer.sense(_ballistic(20.0, 35.0), config))
    assert detection.detected_s == pytest.approx(30.0, abs=0.02)


def test_hot_day_makes_the_barometric_altitude_read_low() -> None:
    """The standard atmosphere underestimates height above a hot pad."""
    hot = IntegrationConfiguration(
        atmosphere=LaunchSiteAtmosphere(
            pad_elevation_m=890.0, pad_temperature_k=303.15, pad_pressure_pa=91432.8
        )
    )
    computer = FlightComputer(baro_noise_pa=0.0)
    trace = computer.sense(_ballistic(20.0, 20.0), hot)
    top = int(np.argmax(trace.altitude))
    assert trace.baro_altitude[top] < trace.altitude[top]


def test_plan_puts_the_canopy_at_line_stretch_after_the_charge() -> None:
    """Detect, wait the delay, eject, then inflate; the cut follows the barometer."""
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    recovery = RecoverySystem(
        (
            ReefedParachute(
                "main",
                diameter_m=3.0,
                drag_coefficient=2.2,
                reefed_opening_diameter_m=1.2,
                reefed_drag_coefficient=0.7,
                disreef_altitude_m=600.0,
            ),
        ),
        body_drag_area_m2=0.01,
    )
    computer = FlightComputer(apogee_delay_s=1.0, main_altitude_m=600.0, lockout_s=5.0)
    flight = _ballistic(20.0, 20.0)
    plan = plan_recovery(flight, config, recovery, computer, EjectionCharge())

    assert plan.apogee.detected_s is not None
    assert plan.fire_s == pytest.approx(plan.apogee.detected_s + 1.0)
    assert plan.line_stretch_s == pytest.approx(
        plan.fire_s + plan.separation.line_stretch_s
    )
    reefed, cut = plan.descent.deployments
    assert reefed.time_s == pytest.approx(plan.line_stretch_s)
    assert plan.main_command_s is not None
    assert cut.altitude_m == pytest.approx(plan.main_true_altitude_m, abs=1e-3)
    # Standard day: the barometer reads the truth, give or take filter lag
    assert plan.main_true_altitude_m == pytest.approx(600.0, abs=15.0)
    assert math.isfinite(plan.separation.snatch_force_n)


def test_plan_with_a_swing_model_flies_the_two_body_descent() -> None:
    """Asking for the swing returns the line along the same timeline."""
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    recovery = RecoverySystem(
        (Parachute("main", diameter_m=3.0, drag_coefficient=1.5),),
        body_drag_area_m2=0.01,
    )
    computer = FlightComputer(apogee_delay_s=1.0, lockout_s=5.0)
    flight = _ballistic(8.0, 8.0)
    point = plan_recovery(flight, config, recovery, computer, EjectionCharge())
    swung = plan_recovery(
        flight,
        config,
        recovery,
        computer,
        EjectionCharge(),
        swing=CanopySwing(line_length_m=8.0),
    )
    assert point.swing is None
    assert swung.swing is not None
    assert len(swung.swing.line_directions) == len(swung.descent.times_s)
    assert swung.line_stretch_s == pytest.approx(point.line_stretch_s)
    assert swung.descent.times_s[-1] == pytest.approx(
        point.descent.times_s[-1], rel=0.02
    )


def test_full_recovery_extension_logs_the_timeline_and_frames() -> None:
    """The opt-in model plans with its own settings and draws every sample."""
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    recovery = RecoverySystem(
        (Parachute("main", diameter_m=3.0, drag_coefficient=1.5),),
        body_drag_area_m2=0.01,
    )
    flight = _ballistic(12.0, 12.0)
    swing = canopy_swing_for(recovery, 3.0)
    # Cord leg to the body, lines, and the centre of gravity past the joint
    assert swing.line_length_m == pytest.approx(
        CORD_TO_BODY_M + 0.9 * 3.0 + 3.0 - 0.9144
    )
    plan = plan_recovery(
        flight,
        config,
        recovery,
        FlightComputer(apogee_delay_s=1.0, lockout_s=5.0),
        EJECTION_CHARGE,
        swing=swing,
    )
    drawn = list(frames(plan))
    assert len(drawn) == len(plan.descent.times_s)
    assert drawn[0][1].line is None and drawn[0][1].nose_sep == 0.0
    ejecting = [f for t, f in drawn if plan.fire_s <= t < plan.line_stretch_s]
    assert ejecting and all(f.nose_sep > 0.0 for f in ejecting)
    hanging = drawn[-1][1]
    assert hanging.line is not None and hanging.nose_dir is not None
    assert hanging.drag_fraction == 1.0

    log = TelemetryLog(get_default_properties(), config)
    log_events(log, plan)
    kinds = [e["kind"] for e in log.events]
    assert kinds[-3:] == ["charge", "stretch", "deploy"]
    # Frames need the swing trace; a point-mass plan is refused
    with pytest.raises(ValueError, match="swing"):
        list(frames(replace(plan, swing=None)))
