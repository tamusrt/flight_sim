"""Tests for the recovery architectures."""

import math

import numpy as np
import pytest

from flight_sim.__main__ import INVICTUS, MORPHEUS, get_default_properties
from flight_sim.descent import Parachute, RecoverySystem, ReefedParachute
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.flight_computer import FlightComputer, plan_recovery
from flight_sim.integration import IntegrationConfiguration, TruthConfiguration
from flight_sim.recovery_extension import EJECTION_CHARGE, frames, log_events
from flight_sim.recovery_systems import (
    BlackPowderCharge,
    Drogueless,
    DualDeploy,
    ReefedSingleSeparation,
    SingleDeploy,
)
from flight_sim.units import vector
from flight_sim.vehicle.mass_properties import MassPropertiesSI
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState
from flight_sim.visualize import TelemetryLog

_G = 9.80665
_CONFIG = IntegrationConfiguration(
    truth=TruthConfiguration(gravity=ConstantGravity(_G))
)
_COMPUTER = FlightComputer(apogee_delay_s=1.0, lockout_s=5.0, main_altitude_m=200.0)
# The 40 kg rocket the ballistic flights stand for, its CG 3 m aft of the nose tip
_APOGEE_MASS = MassPropertiesSI(
    40.0, np.array([-3.0, 0.0, 0.0]), np.diag((0.2, 80.0, 80.0))
)

_DROGUE = Parachute("drogue", diameter_m=0.9, drag_coefficient=0.8)
_MAIN = Parachute("main", diameter_m=2.4, drag_coefficient=2.2, deploy_altitude_m=100.0)
_CHARGE = BlackPowderCharge(2.0, 0.12, 0.2).ejection(
    stroke_m=0.15,
    shear_pins=2,
    pin_strength_n=200.0,
    section_mass_kg=10.0,
    section_drag_area_m2=0.01,
    other_drag_area_m2=0.01,
    cord_length_m=8.0,
)


def _ballistic(apogee_s: float) -> list[tuple[float, RocketState]]:
    """A vertical vacuum arc up to its apogee, sampled every 0.05 s."""
    samples = []
    for t in np.arange(0.0, apogee_s + 1e-9, 0.05):
        samples.append(
            (
                float(t),
                RocketState(
                    position=vector(
                        (_G * apogee_s * t - 0.5 * _G * t**2, 0.0, 0.0), "m"
                    ),
                    velocity=vector((_G * (apogee_s - t), 0.0, 0.0), "m/s"),
                ),
            )
        )
    return samples


def test_charge_pressure_matches_the_recovery_sheet() -> None:
    """2 g in an 8.5 in long, 4.8 in bay is the sheet's 24 psi, within 5%."""
    charge = BlackPowderCharge(2.0, 4.8 * 0.0254, 8.5 * 0.0254)
    assert charge.pressure_pa / 6894.757 == pytest.approx(24.09, rel=0.05)


def test_the_profiles_pick_their_architectures() -> None:
    """Sol Invictus flies one reefed canopy, Morpheus a drogue and a main."""
    assert isinstance(INVICTUS.scheme, ReefedSingleSeparation)
    assert isinstance(MORPHEUS.scheme, DualDeploy)
    assert MORPHEUS.scheme.separations == 2
    assert INVICTUS.recovery is INVICTUS.scheme.recovery


def test_morpheus_follows_srt12s_recovery_settings() -> None:
    """Canopies, avionics settings and charges are the recovery team's."""
    scheme = MORPHEUS.scheme
    assert isinstance(scheme, DualDeploy)
    drogue, main = scheme.recovery.parachutes
    assert isinstance(drogue, Parachute) and isinstance(main, Parachute)
    assert (drogue.diameter_m, drogue.drag_coefficient) == pytest.approx(
        (36 * 0.0254, 0.8)
    )
    assert (main.diameter_m, main.drag_coefficient) == pytest.approx((96 * 0.0254, 2.2))
    assert scheme.computer.main_altitude_m == pytest.approx(1500 * 0.3048)
    assert scheme.computer.apogee_delay_s == 1.0 and scheme.main_delay_s == 1.0
    assert scheme.apogee_charge is not None and scheme.main_charge is not None
    assert scheme.apogee_charge.pressure_pa / 6894.757 == pytest.approx(25.2, abs=0.2)
    assert scheme.main_charge.pressure_pa / 6894.757 == pytest.approx(23.4, abs=0.2)
    assert MORPHEUS.wind.speed.m_as("m/s") == pytest.approx(12 * 0.44704)
    assert MORPHEUS.pad_temperature_k == pytest.approx(305.93, abs=0.01)


def test_single_deploy_flies_as_plan_recovery_does() -> None:
    """The single-separation classes use the existing planner unchanged."""
    recovery = RecoverySystem((Parachute("main", 3.0, 1.5),), body_drag_area_m2=0.01)
    flight = _ballistic(12.0)
    scheme = SingleDeploy(recovery, _COMPUTER, EJECTION_CHARGE)
    plan = scheme.plan(flight, _CONFIG, _APOGEE_MASS)
    expected = plan_recovery(
        flight,
        _CONFIG,
        recovery,
        _COMPUTER,
        EJECTION_CHARGE,
        mass_kg=_APOGEE_MASS.mass,
        swing=scheme.swing(3.0),
    )
    assert plan.fire_s == expected.fire_s
    assert plan.descent.times_s[-1] == expected.descent.times_s[-1]
    assert plan.main_separation is None
    assert len(list(scheme.frames(plan))) == len(plan.descent.times_s)


def test_dual_deploy_releases_the_drogue_then_the_main_on_a_second_separation() -> None:
    """The main inflates after its command, delay and cord, below the setting."""
    scheme = DualDeploy(
        RecoverySystem((_DROGUE, _MAIN), body_drag_area_m2=0.01),
        _COMPUTER,
        _CHARGE,
        _CHARGE,
        main_delay_s=1.0,
    )
    plan = scheme.plan(_ballistic(12.0), _CONFIG, _APOGEE_MASS)
    drogue, main = plan.descent.deployments
    assert drogue.name == "drogue" and main.name == "main"
    assert drogue.time_s == pytest.approx(plan.line_stretch_s, abs=0.2)
    assert plan.main_command_s is not None and plan.main_fire_s is not None
    assert plan.main_fire_s == pytest.approx(plan.main_command_s + 1.0)
    assert plan.main_line_stretch_s is not None
    assert main.time_s == pytest.approx(plan.main_line_stretch_s, abs=0.2)
    assert plan.main_true_altitude_m is not None
    assert main.altitude_m < plan.main_true_altitude_m
    assert plan.descent.landed and plan.swing is None

    log = TelemetryLog(_log_properties(), _CONFIG)
    log_events(log, plan)
    kinds = [e["kind"] for e in log.events if e["kind"] in ("charge", "deploy")]
    assert kinds == ["charge", "deploy", "charge", "deploy"]
    ejecting = [
        f.nose_sep
        for t, f in frames(plan, scheme)
        if plan.main_fire_s <= t < plan.main_line_stretch_s
    ]
    assert ejecting and all(d > 0.0 for d in ejecting)


def test_drogueless_falls_until_the_main_separation() -> None:
    """Nothing separates at apogee, so the first canopy is the main."""
    scheme = Drogueless(
        RecoverySystem((_MAIN,), body_drag_area_m2=0.05),
        _COMPUTER,
        None,
        _CHARGE,
        main_delay_s=0.5,
    )
    plan = scheme.plan(_ballistic(12.0), _CONFIG, _APOGEE_MASS)
    assert not plan.separation.separated and math.isinf(plan.line_stretch_s)
    assert [d.name for d in plan.descent.deployments] == ["main"]
    assert plan.main_line_stretch_s is not None
    assert plan.descent.deployments[0].time_s == pytest.approx(
        plan.main_line_stretch_s, abs=0.2
    )
    assert plan.descent.landed


@pytest.mark.parametrize(
    "build",
    [
        lambda: SingleDeploy(
            RecoverySystem((ReefedParachute("m", 3.0, 2.0, 1.0, 0.7, 100.0),)),
            _COMPUTER,
            _CHARGE,
        ),
        lambda: ReefedSingleSeparation(
            RecoverySystem((Parachute("m", 3.0, 2.0),)), _COMPUTER, _CHARGE
        ),
        lambda: DualDeploy(RecoverySystem((_MAIN,)), _COMPUTER, _CHARGE, _CHARGE),
        lambda: DualDeploy(RecoverySystem((_DROGUE, _MAIN)), _COMPUTER, _CHARGE),
        lambda: Drogueless(RecoverySystem((_MAIN,)), _COMPUTER, _CHARGE, _CHARGE),
        lambda: SingleDeploy(RecoverySystem((_MAIN,)), _COMPUTER, _CHARGE),
    ],
)
def test_each_class_refuses_the_wrong_canopies_and_charges(build: object) -> None:
    """A class only builds from the parts its architecture has."""
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def _log_properties() -> RocketProperties:
    return get_default_properties()
