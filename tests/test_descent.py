"""Tests for the point-mass descent under parachutes."""

import math
from dataclasses import replace

import numpy as np
import pytest

from flight_sim.descent import (
    Parachute,
    RecoverySystem,
    ReefedParachute,
    parachute_diameter_for_descent_rate,
    simulate_descent,
)
from flight_sim.environment.atmosphere import (
    LaunchSiteAtmosphere,
    StandardAtmosphere1976,
    VacuumAtmosphere,
)
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.integration import IntegrationConfiguration
from flight_sim.launch_rail import LaunchRail
from flight_sim.units import scalar, vector
from flight_sim.utilities.dcm import body_to_world
from flight_sim.vehicle.rocket_state import RocketState

_G = 9.80665


def _state(
    altitude_m: float,
    velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass_kg: float = 40.0,
) -> RocketState:
    """Return a rocket at an altitude, spinning slowly."""
    return RocketState(
        current_mass=scalar(mass_kg, "kg"),
        inertia=vector((0.2, 90.0, 90.0), "kg*m**2"),
        cg_location=vector((-3.0, 0.0, 0.0), "m"),
        position=vector((altitude_m, 0.0, 0.0), "m"),
        velocity=vector(velocity_m_s, "m/s"),
        angular_velocity=vector((0.5, 0.0, 0.0), "rad/s"),
    )


def _sea_level() -> IntegrationConfiguration:
    return IntegrationConfiguration(gravity=ConstantGravity(_G))


def test_vacuum_descent_is_free_fall() -> None:
    """With no air the rocket falls as 0.5 * g * t**2, canopies or not."""
    config = IntegrationConfiguration(
        atmosphere=VacuumAtmosphere(), gravity=ConstantGravity(9.81)
    )
    recovery = RecoverySystem((Parachute("main", 3.0, 1.5),))
    result = simulate_descent(10.0, _state(1000.0), config, recovery)

    assert result.landed
    fall_time = result.times_s[-1] - 10.0
    assert fall_time == pytest.approx(math.sqrt(2 * 1000.0 / 9.81), rel=1e-6)
    impact_speed = float(-result.states[-1].velocity.m_as("m/s")[0])
    assert impact_speed == pytest.approx(9.81 * fall_time, rel=1e-6)
    assert result.states[-1].position.m_as("m")[0] == 0.0


def test_samples_are_evenly_spaced_and_carry_the_mass() -> None:
    """Samples come every output interval and keep the apogee mass."""
    recovery = RecoverySystem((Parachute("drogue", 1.0, 1.5),))
    result = simulate_descent(
        0.0, _state(300.0), _sea_level(), recovery, output_interval_s=0.5
    )
    gaps = np.diff(result.times_s[:-1])
    assert gaps == pytest.approx(0.5, abs=1e-9)
    assert result.times_s[-1] - result.times_s[-2] <= 0.5 + 1e-9
    assert all(float(s.current_mass.m_as("kg")) == 40.0 for s in result.states)
    assert all(
        float(np.linalg.norm(s.angular_velocity.m_as("rad/s"))) == 0.0
        for s in result.states
    )


def test_parachute_reaches_the_designed_descent_rate() -> None:
    """A canopy sized for 7.6 m/s lands at that speed in sea-level air."""
    diameter = parachute_diameter_for_descent_rate(40.0, 7.6, 1.5)
    recovery = RecoverySystem((Parachute("main", diameter, 1.5),))
    result = simulate_descent(0.0, _state(500.0), _sea_level(), recovery)

    landing = result.states[-1].velocity.m_as("m/s")
    # Density rises 5% over the last 500 m, so it lands slightly slower
    assert -landing[0] == pytest.approx(7.6, rel=0.01)


def test_wind_carries_the_rocket_downwind() -> None:
    """Under a canopy the horizontal speed matches the wind."""
    config = IntegrationConfiguration(
        atmosphere=StandardAtmosphere1976(wind_m_s=np.array([0.0, 5.0, -2.0]))
    )
    recovery = RecoverySystem((Parachute("main", 3.0, 1.5),))
    result = simulate_descent(0.0, _state(300.0), config, recovery)

    landing = result.states[-1]
    assert landing.velocity.m_as("m/s")[1:] == pytest.approx([5.0, -2.0], abs=1e-3)
    position = landing.position.m_as("m")
    assert position[1] > 0.0 > position[2]


def test_drogue_on_delay_and_main_at_altitude() -> None:
    """The drogue opens on its delay, the main at its height, in that order."""
    recovery = RecoverySystem(
        (
            Parachute("drogue", 1.0, 2.2, deploy_delay_s=3.0),
            Parachute("main", 3.6, 2.2, deploy_altitude_m=450.0),
        ),
        body_drag_area_m2=0.01,
    )
    result = simulate_descent(
        40.0, _state(3000.0, (0.0, 50.0, 0.0)), _sea_level(), recovery
    )

    drogue, main = result.deployments
    assert drogue.name == "drogue"
    assert drogue.time_s == pytest.approx(43.0)
    # Falling for 3 s from 50 m/s sideways
    assert drogue.airspeed_m_s > 50.0
    assert main.name == "main"
    assert main.altitude_m == pytest.approx(450.0, abs=1e-3)
    assert main.time_s > drogue.time_s
    # Filling distance 8 diameters; the speed barely changes in 0.14 s
    assert drogue.inflation_time_s == pytest.approx(
        8.0 * 1.0 / drogue.airspeed_m_s, rel=0.02
    )
    assert main.peak_load_g > 1.0
    # Under the drogue alone it falls faster than under both
    altitude = np.array([s.position.m_as("m")[0] for s in result.states])
    speed = np.array([-s.velocity.m_as("m/s")[0] for s in result.states])
    under_drogue = speed[(altitude > 1000.0) & (altitude < 2000.0)]
    assert under_drogue.min() > 3 * speed[-1]


def test_main_opens_at_once_below_its_height() -> None:
    """A rocket that peaks below the main's height releases it at apogee."""
    recovery = RecoverySystem((Parachute("main", 3.0, 2.2, deploy_altitude_m=450.0),))
    result = simulate_descent(20.0, _state(300.0), _sea_level(), recovery)
    assert result.deployments[0].time_s == pytest.approx(20.0)


def test_opening_load_matches_a_hand_estimate() -> None:
    """A heavy body that barely slows feels 0.5 * rho * v**2 * CdA / m."""
    parachute = Parachute("main", 2.0, 1.5)
    config = IntegrationConfiguration(gravity=ConstantGravity(0.0))
    heavy = _state(100.0, (-20.0, 0.0, 0.0), mass_kg=1e5)
    result = simulate_descent(
        0.0, heavy, config, RecoverySystem((parachute,)), max_time_s=2.0
    )
    density = config.atmosphere.conditions(100.0).air_density
    expected = 0.5 * density * 20.0**2 * parachute.drag_area_m2 / 1e5 / _G
    # Within 0.5%: the air thickens slightly as it drops 36 m while opening
    assert result.deployments[0].peak_load_g == pytest.approx(expected, rel=5e-3)
    assert result.deployments[0].inflation_time_s == pytest.approx(0.8, rel=1e-3)


def test_opening_load_is_below_the_instant_opening_value() -> None:
    """A light rocket slows while its canopy fills, easing the opening load."""
    parachute = Parachute("main", 3.0, 2.2)
    config = _sea_level()
    result = simulate_descent(
        0.0, _state(400.0, (-25.0, 0.0, 0.0)), config, RecoverySystem((parachute,))
    )
    density = config.atmosphere.conditions(400.0).air_density
    instant = 0.5 * density * 25.0**2 * parachute.drag_area_m2 / 40.0 / _G
    assert 1.0 < result.deployments[0].peak_load_g < instant


def test_time_limit_stops_without_landing() -> None:
    """Running out of time reports that the rocket has not landed."""
    recovery = RecoverySystem((Parachute("main", 3.0, 2.2),))
    result = simulate_descent(0.0, _state(3000.0), _sea_level(), recovery, max_time_s=5)
    assert not result.landed
    assert result.times_s[-1] == pytest.approx(5.0)
    assert result.states[-1].position.m_as("m")[0] > 0.0


def test_display_attitude_hangs_nose_up_under_the_canopy() -> None:
    """After the turn the nose points up along the airflow past the rocket."""
    config = IntegrationConfiguration(
        atmosphere=LaunchSiteAtmosphere(wind_m_s=np.array([0.0, 4.0, 0.0])),
        gravity=ConstantGravity(_G),
    )
    recovery = RecoverySystem((Parachute("main", 3.0, 2.2, deploy_delay_s=1.0),))
    result = simulate_descent(0.0, _state(500.0), config, recovery)

    first = body_to_world(result.states[0].orientation)[:, 0]
    assert first == pytest.approx([1.0, 0.0, 0.0], abs=0.05)
    landing = result.states[-1]
    nose = body_to_world(landing.orientation)[:, 0]
    airflow = np.array([0.0, 4.0, 0.0]) - landing.velocity.m_as("m/s")
    assert nose == pytest.approx(airflow / np.linalg.norm(airflow), abs=1e-6)


def test_spill_hole_reduces_the_canopy_area() -> None:
    """The drag area uses the disc less the vent."""
    vented = Parachute("main", 3.0, 2.0, spill_hole_diameter_m=0.5)
    assert vented.drag_area_m2 == pytest.approx(2.0 * math.pi * (9.0 - 0.25) / 4)


def test_diameter_for_descent_rate_inverts_the_drag_equation() -> None:
    """The sized canopy's drag balances the weight at the design speed."""
    diameter = parachute_diameter_for_descent_rate(30.0, 6.0, 2.0, air_density=1.1)
    drag_area = Parachute("main", diameter, 2.0).drag_area_m2
    assert 0.5 * 1.1 * 6.0**2 * drag_area == pytest.approx(30.0 * _G)


def test_instant_canopy_opens_at_release() -> None:
    """A zero filling distance gives the full drag area at once."""
    recovery = RecoverySystem((Parachute("main", 3.0, 2.2, fill_constant=0.0),))
    result = simulate_descent(
        0.0, _state(300.0, (-20.0, 0.0, 0.0)), _sea_level(), recovery
    )
    assert result.deployments[0].inflation_time_s == pytest.approx(0.0)
    assert result.landed


def test_canopy_released_at_apogee_fills_over_its_distance() -> None:
    """From rest the canopy opens once the rocket has fallen about n * D.

    Falling from rest, 8 m takes sqrt(2 * 8 / g) = 1.28 s with no drag; the
    opening canopy slows the fall a little, so it takes slightly longer.
    """
    recovery = RecoverySystem((Parachute("drogue", 1.0, 2.2),))
    result = simulate_descent(0.0, _state(3000.0), _sea_level(), recovery)
    time = result.deployments[0].inflation_time_s
    assert time is not None
    assert math.sqrt(2 * 8.0 / _G) < time < 1.5


def test_display_attitude_turns_without_rolling() -> None:
    """A rocket lying along +Y swings nose-up about Z, keeping Z fixed."""
    rail = LaunchRail(1.0, math.pi / 2, tilt_heading=(1.0, 0.0))
    start = replace(_state(800.0), orientation=rail.orientation())
    recovery = RecoverySystem((Parachute("main", 3.0, 2.2, deploy_delay_s=0.5),))
    config = IntegrationConfiguration(gravity=ConstantGravity(_G))
    result = simulate_descent(0.0, start, config, recovery)

    first = body_to_world(result.states[0].orientation)
    assert first[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    for sample in result.states:
        to_world = body_to_world(sample.orientation)
        assert to_world[:, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    halfway = body_to_world(result.states[14].orientation)[:, 0]  # t = 1.5 s
    assert halfway == pytest.approx(
        [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0], abs=1e-6
    )
    assert body_to_world(result.states[-1].orientation)[:, 0] == pytest.approx(
        [1.0, 0.0, 0.0], abs=1e-6
    )


_SLUG_FT3 = 515.378818  # kg/m**3 per slug/ft**3
_LB = 0.45359237  # kg
_FT = 0.3048  # m


def _team_main(disreef_altitude_m: float = 2000 * _FT) -> ReefedParachute:
    """The recovery team's reefed Fruity Chutes 120 in canopy."""
    return ReefedParachute(
        "main",
        diameter_m=10 * _FT,
        drag_coefficient=2.2,
        reefed_opening_diameter_m=3.9796 * _FT,
        reefed_drag_coefficient=0.7,
        disreef_altitude_m=disreef_altitude_m,
        deploy_delay_s=1.0,
        spill_hole_diameter_m=21.12 * 0.0254,
    )


def test_reefed_canopy_reproduces_the_recovery_team_hand_calcs() -> None:
    """The team gets 110 ft/s reefed and 22.83 ft/s open for 90 lb at 4779 ft.

    Their air density there is 0.0020620996 slug/ft**3; their effective areas
    are 76.106 ft**2 open and 10.306 ft**2 reefed.
    """
    main = _team_main()
    assert main.drag_area_m2 == pytest.approx(2.2 * 76.10669 * _FT**2, rel=1e-5)
    assert main.reefed_drag_area_m2 == pytest.approx(0.7 * 10.306 * _FT**2, rel=1e-4)
    density = 0.0020620996 * _SLUG_FT3
    weight = 90 * _LB * _G

    def descent_rate_ft_s(drag_area: float) -> float:
        return math.sqrt(2 * weight / (density * drag_area)) / _FT

    assert descent_rate_ft_s(main.reefed_drag_area_m2) == pytest.approx(110.0, rel=2e-3)
    assert descent_rate_ft_s(main.drag_area_m2) == pytest.approx(22.833, rel=2e-3)


def test_reefed_canopy_flies_reefed_then_disreefs_at_its_height() -> None:
    """Reefed drag down to the cut height, the full canopy below it."""
    main = _team_main()
    recovery = RecoverySystem((main,))
    result = simulate_descent(0.0, _state(2500.0), _sea_level(), recovery)

    reefed, cut = result.deployments
    assert reefed.name == "main reefed"
    assert reefed.time_s == pytest.approx(1.0)
    assert cut.name == "main reef cut"
    assert cut.altitude_m == pytest.approx(2000 * _FT, abs=1e-3)
    # Just above the cut it falls near the reefed terminal speed there
    above = next(s for s in result.states if float(s.position.m_as("m")[0]) < 700.0)
    density = _sea_level().atmosphere.conditions(700.0).air_density
    reefed_rate = math.sqrt(2 * 40.0 * _G / (density * main.reefed_drag_area_m2))
    assert -above.velocity.m_as("m/s")[0] == pytest.approx(reefed_rate, rel=0.02)
    landing = -result.states[-1].velocity.m_as("m/s")[0]
    density = _sea_level().atmosphere.conditions(0.0).air_density
    open_rate = math.sqrt(2 * 40.0 * _G / (density * main.drag_area_m2))
    assert landing == pytest.approx(open_rate, rel=0.01)
    assert cut.peak_force_n == pytest.approx(cut.peak_load_g * _G * 40.0)


def test_reef_cut_waits_for_the_reefed_opening() -> None:
    """Starting below the cut height, the canopy still opens reefed first."""
    result = simulate_descent(
        0.0, _state(400.0), _sea_level(), RecoverySystem((_team_main(),))
    )
    reefed, cut = result.deployments
    assert reefed.time_s == pytest.approx(1.0)
    assert cut.time_s == pytest.approx(1.0)
    assert result.landed
