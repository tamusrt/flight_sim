"""Tests for the nose ejection and the swing under the canopy."""

import math

import numpy as np
import pytest

from flight_sim.recovery_motion import (
    CORD_TO_BODY_M,
    CORD_TO_NOSE_M,
    EjectionCharge,
    NoseSwing,
)

_G = 9.80665


def test_ejection_in_still_air_follows_the_energy_balance() -> None:
    """Gas work over the stroke sets the speed; the cord comes tight later."""
    charge = EjectionCharge(nose_mass_kg=4.0)
    result = charge.separate(total_mass_kg=44.0, airspeed_m_s=0.0, air_density=0.0)

    reduced = 4.0 * 40.0 / 44.0
    force = (
        charge.pressure_pa * math.pi * charge.bore_radius_m**2
        - 3 * charge.pin_strength_n
    )
    speed = math.sqrt(2 * force * charge.stroke_m / reduced)
    assert result.separated
    assert result.speed_m_s == pytest.approx(speed)
    assert result.line_stretch_s == pytest.approx(
        (charge.cord_length_m - charge.stroke_m) / speed
    )
    assert result.snatch_force_n == pytest.approx(
        speed * math.sqrt(charge.cord_stiffness_n_m * reduced)
    )
    assert result.distances_m[-1] == pytest.approx(charge.cord_length_m)


def test_drag_on_the_nose_slows_the_ejection() -> None:
    """In fast air the nose arrives later and slower than in still air."""
    charge = EjectionCharge(nose_mass_kg=4.0)
    still = charge.separate(44.0, 0.0, 0.0)
    windy = charge.separate(44.0, 80.0, 0.6)
    assert windy.line_stretch_s > still.line_stretch_s
    assert windy.stretch_speed_m_s < still.stretch_speed_m_s
    assert windy.snatch_force_n < still.snatch_force_n


def test_shear_pins_can_hold_a_weak_charge() -> None:
    """Below the pins' strength nothing separates."""
    charge = EjectionCharge(pressure_pa=5000.0, shear_pins=4)
    result = charge.separate(44.0, 0.0, 0.0)
    assert not result.separated
    assert result.line_stretch_s == math.inf


def test_cord_legs_add_up_to_the_charge_cord() -> None:
    """The ejection runs out the same cord the swing hangs from."""
    assert EjectionCharge().cord_length_m == pytest.approx(
        CORD_TO_NOSE_M + CORD_TO_BODY_M
    )


def _swing(
    model: NoseSwing, accel: np.ndarray, seconds: float, start: np.ndarray
) -> tuple[list[float], np.ndarray]:
    """Swing of the nose for a constant canopy acceleration."""
    times = [float(x) for x in np.arange(0.0, seconds, 0.05)]
    velocities = np.array([accel * t for t in times])
    return times, model.swing(times, velocities, start_s=0.0, start=start)


def test_nose_settles_down_to_the_side_of_the_body_leg() -> None:
    """With no acceleration the nose hangs a little to the side of vertical."""
    start = np.array([0.0, 1.0, 0.0])  # Trailing horizontally after the opening
    _, nose = _swing(NoseSwing(length_m=7.0), np.zeros(3), 60.0, start)
    assert math.degrees(math.acos(-nose[-1][0])) == pytest.approx(12.0, abs=1.0)


def test_nose_swing_period_matches_a_pendulum() -> None:
    """A small swing returns through vertical every half period of 2 pi sqrt(L/g)."""
    model = NoseSwing(length_m=7.0, damping=0.0, spread_deg=0.0)
    tilt = math.radians(5.0)
    start = np.array([-math.cos(tilt), math.sin(tilt), 0.0])
    times, nose = _swing(model, np.zeros(3), 12.0, start)
    sideways = nose[:, 1]
    crossings = [
        times[i] for i in range(1, len(times)) if sideways[i - 1] > 0.0 >= sideways[i]
    ]
    quarter = 2 * math.pi * math.sqrt(7.0 / _G) / 4
    assert crossings[0] == pytest.approx(quarter, rel=0.03)


def test_deceleration_trails_the_nose_forward() -> None:
    """Braking at a sideways rate a hangs it at atan(a / g) from vertical."""
    model = NoseSwing(length_m=7.0, spread_deg=0.0)
    braking = np.array([0.0, -3.0, 0.0])  # Slowing while moving along +Y
    _, nose = _swing(model, braking, 80.0, np.array([-1.0, 0.0, 0.0]))
    angle = math.degrees(math.atan2(nose[-1][1], -nose[-1][0]))
    assert angle == pytest.approx(math.degrees(math.atan(3.0 / _G)), abs=0.5)


def test_nose_stays_at_its_start_before_line_stretch() -> None:
    """Until the start time the nose is where it was put."""
    model = NoseSwing(length_m=7.0)
    times = [float(x) for x in np.arange(0.0, 10.0, 0.1)]
    start = np.array([0.0, 1.0, 0.0])
    nose = model.swing(times, np.zeros((len(times), 3)), start_s=5.0, start=start)
    assert nose[0] == pytest.approx(start)
    assert nose[49] == pytest.approx(start)
    assert nose[-1] != pytest.approx(start)
