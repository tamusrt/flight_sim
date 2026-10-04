"""Tests for the RECO versions and FastRECO."""

import dataclasses
import math
from itertools import pairwise

import numpy as np
import pytest

from flight_sim import fast_reco, reco
from flight_sim.descent import Parachute, RecoverySystem, simulate_descent
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.environment.wind import UniformWind
from flight_sim.fast_reco import FastRECO, _fall, _opening, _time_to_fall
from flight_sim.flight_computer import FlightComputer
from flight_sim.integration import IntegrationConfiguration, TruthConfiguration
from flight_sim.reco import FullRECO, RecoveryRequest, get_reco
from flight_sim.recovery_systems import (
    BlackPowderCharge,
    Drogueless,
    DualDeploy,
    SingleDeploy,
    SingleSeparation,
)
from flight_sim.units import scalar, vector
from flight_sim.vehicle.mass_properties import MassPropertiesSI
from flight_sim.vehicle.rocket_state import RocketState
from tests.test_recovery_systems import (
    _APOGEE_MASS,
    _CHARGE,
    _COMPUTER,
    _DROGUE,
    _MAIN,
    _ballistic,
)

_G = 9.80665
_CONFIG = IntegrationConfiguration(
    truth=TruthConfiguration(
        gravity=ConstantGravity(_G),
        wind=UniformWind(speed=scalar(4.0, "m/s"), from_azimuth=scalar(270.0, "deg")),
    )
)


def _request(apogee_s: float = 12.0) -> RecoveryRequest:
    return RecoveryRequest(_ballistic(apogee_s), _CONFIG, _APOGEE_MASS)


def _dual() -> DualDeploy:
    return DualDeploy(
        RecoverySystem((_DROGUE, _MAIN), body_drag_area_m2=0.01),
        _COMPUTER,
        _CHARGE,
        _CHARGE,
        main_delay_s=1.0,
    )


def _compare(scheme: object) -> tuple[reco.RecoveryOutcome, reco.RecoveryOutcome]:
    request = _request()
    full = FullRECO(scheme).descend(request)  # type: ignore[arg-type]
    fast = FastRECO(scheme, turbulence=0.0).descend(request)  # type: ignore[arg-type]
    return full, fast


def test_versions_are_registered_by_name() -> None:
    """The past version and the fast one are both found by name."""
    assert {"full", "fast"} <= set(reco.versions())
    assert isinstance(get_reco("fast", _dual()), FastRECO)
    assert isinstance(get_reco("full", _dual()), FullRECO)
    with pytest.raises(KeyError):
        get_reco("slow", _dual())


def test_a_new_version_can_be_added_with_register() -> None:
    """Adding a version is a subclass and one decorator."""

    @reco.register
    class Dummy(reco.RECO):  # pylint: disable=unused-variable
        """A version that flies the full model, to be found by name."""

        name = "dummy-for-test"

        def descend(self, request: RecoveryRequest) -> reco.RecoveryOutcome:
            return FullRECO(self.scheme).descend(request)

    assert "dummy-for-test" in reco.versions()
    reco._VERSIONS.pop("dummy-for-test")  # pylint: disable=protected-access


def test_full_reco_is_the_scheme_plan() -> None:
    """The past version gives what ``scheme.plan`` always gave."""
    scheme = _dual()
    request = _request()
    plan = scheme.plan(request.flight, _CONFIG, _APOGEE_MASS)
    outcome = FullRECO(scheme).descend(request)
    assert outcome.plan is not None
    assert outcome.descent.times_s == plan.descent.times_s
    assert outcome.main_command_s == plan.main_command_s
    assert outcome.fire_s == plan.fire_s


def test_fast_reco_lands_where_the_full_one_does_with_the_same_loads() -> None:
    """Dual deploy: landing point, deployments and opening loads agree closely."""
    full, fast = _compare(_dual())
    assert fast.descent.landed
    drift = float(np.hypot(*full.landing_position_m[1:]))
    assert float(np.linalg.norm(fast.landing_position_m - full.landing_position_m)) < (
        0.02 * drift + 2.0
    )
    assert fast.landing_velocity_m_s[0] == pytest.approx(
        full.landing_velocity_m_s[0], abs=0.15
    )
    assert [d.name for d in fast.descent.deployments] == [
        d.name for d in full.descent.deployments
    ]
    for a, b in zip(fast.descent.deployments, full.descent.deployments, strict=True):
        assert a.altitude_m == pytest.approx(b.altitude_m, abs=6.0)
        assert a.peak_force_n == pytest.approx(b.peak_force_n, rel=0.08)
    assert fast.main_command_s == pytest.approx(full.main_command_s, abs=0.3)
    assert fast.drogue_rate_m_s() == pytest.approx(full.drogue_rate_m_s(), rel=0.03)


def test_fast_reco_handles_a_single_canopy_and_a_drogueless_fall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other schemes still agree (with the gusts off in both models)."""
    swing = SingleSeparation.swing
    monkeypatch.setattr(
        SingleSeparation,
        "swing",
        lambda self, cg: dataclasses.replace(swing(self, cg), turbulence=0.0),
    )
    single = SingleDeploy(
        RecoverySystem((Parachute("main", 3.0, 1.5),), body_drag_area_m2=0.01),
        _COMPUTER,
        _CHARGE,
    )
    drogueless = Drogueless(
        RecoverySystem((_MAIN,), body_drag_area_m2=0.05),
        _COMPUTER,
        None,
        _CHARGE,
        main_delay_s=0.5,
    )
    for scheme, tolerance in ((single, 0.08), (drogueless, 0.03)):
        full, fast = _compare(scheme)
        assert fast.descent.landed and full.descent.landed
        drift = float(np.hypot(*full.landing_position_m[1:]))
        gap = float(np.linalg.norm(fast.landing_position_m - full.landing_position_m))
        assert gap < tolerance * drift + 3.0
        assert fast.landing_velocity_m_s[0] == pytest.approx(
            full.landing_velocity_m_s[0], abs=0.4
        )


def test_a_failed_separation_leaves_the_rocket_falling_without_a_canopy() -> None:
    """If the charge cannot break the pins the drogue never opens."""
    weak = BlackPowderCharge(0.01, 0.12, 0.2).ejection(
        stroke_m=0.15,
        shear_pins=2,
        pin_strength_n=2000.0,
        section_mass_kg=10.0,
        section_drag_area_m2=0.01,
        other_drag_area_m2=0.01,
        cord_length_m=8.0,
    )
    scheme = DualDeploy(
        RecoverySystem((_DROGUE, _MAIN), body_drag_area_m2=0.01),
        _COMPUTER,
        weak,
        _CHARGE,
        main_delay_s=1.0,
    )
    full = FullRECO(scheme).descend(_request())
    fast = FastRECO(scheme).descend(_request())
    assert not fast.separated and not full.separated
    assert [d.name for d in fast.descent.deployments] == [
        d.name for d in full.descent.deployments
    ]
    assert fast.descent.landed


def test_an_opening_matches_the_integrated_one() -> None:
    """The closed-form opening gives the integrated opening's speed and load."""
    stage = _MAIN.stages()[0]
    state = RocketState(
        position=vector((800.0, 0.0, 0.0), "m"),
        velocity=vector((-35.0, 0.0, 0.0), "m/s"),
    )
    free = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(_G))
    )
    air = free.truth.atmosphere.conditions(800.0)
    solved = _opening(
        stage, 0.01, np.array([-35.0, 0.0, 0.0]), np.zeros(3), air.air_density, 40.0
    )
    descent = simulate_descent(
        0.0,
        state,
        free,
        RecoverySystem((_main_at_start(),), body_drag_area_m2=0.01),
        mass_kg=40.0,
        max_time_s=solved.elapsed_s + 1.5,
        max_step_s=0.002,
    )
    integrated = descent.deployments[0]
    assert solved.peak_force_n == pytest.approx(integrated.peak_force_n, rel=0.06)
    assert solved.filled_s == pytest.approx(integrated.inflation_time_s, rel=0.08)
    assert abs(solved.velocity[0]) < 35.0


def _main_at_start() -> Parachute:
    return Parachute("main", 2.4, 2.2, deploy_delay_s=0.0)


def test_the_fall_formula_matches_a_numerical_fall() -> None:
    """Exact drag-limited fall, from below and from above terminal velocity."""
    terminal, gravity = 8.0, _G
    for start in (0.0, 3.0, 20.0):
        speed, distance, dt = start, 0.0, 1e-4
        for _ in range(int(3.0 / dt)):
            speed += gravity * (1.0 - speed**2 / terminal**2) * dt
            distance += speed * dt
        u, d = _fall(start, terminal, gravity, 3.0)
        assert u == pytest.approx(speed, rel=2e-3)
        assert d == pytest.approx(distance, rel=2e-3)
    reach = _time_to_fall(5.0, terminal, gravity, 10.0, 5.0)
    assert _fall(5.0, terminal, gravity, reach)[1] == pytest.approx(10.0, abs=1e-6)


def test_steps_never_cross_more_than_the_guard_once_settled() -> None:
    """The guard keeps every wind layer in view: no step covers more than it."""
    outcome = FastRECO(_dual(), guard_m=20.0).descend(_request())
    heights = [float(s.position.m_as("m")[0]) for s in outcome.descent.states]
    times = outcome.descent.times_s
    # a canopy opening is one solved jump; the guard is for the steps between
    openings = [
        (d.time_s - 0.01, d.time_s + 2.0 * d.inflation_time_s + 0.1)
        for d in outcome.descent.deployments
    ]
    big = [
        a - b
        for (a, b), (t0, t1) in zip(pairwise(heights), pairwise(times), strict=True)
        if t1 - t0 > 0.3 and not any(t0 <= hi and lo <= t1 for lo, hi in openings)
    ]
    assert big and max(big) <= 1.3 * 20.0
    assert outcome.descent.states[-1].position.m_as("m")[0] == pytest.approx(0.0)


def test_fast_reco_uses_far_fewer_steps_than_the_full_model() -> None:
    """Fewer samples to compute (a short test descent; real ones save far more)."""
    full, fast = _compare(_dual())
    assert len(fast.descent.times_s) < len(full.descent.times_s) / 2
    assert math.isfinite(fast.drift_m)


def test_the_module_registers_itself() -> None:
    """Importing the module is enough to make 'fast' available."""
    assert fast_reco.FastRECO.name == "fast"
    assert isinstance(_APOGEE_MASS, MassPropertiesSI)
    assert isinstance(_COMPUTER, FlightComputer)


def test_gusts_move_the_landing_and_repeat_for_the_same_flight() -> None:
    """Gusts under the last canopy move the landing; the same flight lands the same."""
    calm = FastRECO(_dual(), turbulence=0.0).descend(_request())
    gusty = FastRECO(_dual()).descend(_request())
    again = FastRECO(_dual()).descend(_request())
    assert gusty.descent.landed
    assert (
        float(np.linalg.norm(gusty.landing_position_m - calm.landing_position_m)) > 1.0
    )
    assert np.allclose(gusty.landing_position_m, again.landing_position_m)
    assert gusty.landing_velocity_m_s[0] == pytest.approx(
        calm.landing_velocity_m_s[0], abs=0.5
    )
