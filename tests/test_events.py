"""Flight event tests."""

import pytest

from flight_sim.environment.atmosphere import VacuumAtmosphere
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.events import APOGEE, IMPACT, FlightEvent, peak_vertical_velocity
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import scalar, vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


def _coasting_state(vertical_velocity: float) -> RocketState:
    """Return a massless state rising at the given vertical velocity."""
    return RocketState(
        current_mass=scalar(0.0, "kg"),
        inertia=vector((0.1, 2.5, 2.5), "kg*m**2"),
        cg_location=vector((-2.5, 0.0, 0.0), "m"),
        velocity=vector((vertical_velocity, 0.0, 0.0), "m/s"),
    )


def test_event_crossed_when_value_falls_through_zero() -> None:
    """A step from a positive value to a non-positive one crosses the event."""
    rising = _coasting_state(50.0)
    falling = _coasting_state(-1.0)

    assert APOGEE.crossed(0.0, rising, 1.0, falling)
    assert not APOGEE.crossed(0.0, falling, 1.0, rising)
    assert not APOGEE.crossed(0.0, rising, 1.0, rising)


def test_impact_is_not_crossed_on_the_pad() -> None:
    """A rocket sitting at zero altitude has not impacted."""
    on_pad = _coasting_state(0.0)

    assert not IMPACT.crossed(0.0, on_pad, 1.0, on_pad)


def test_adaptive_step_ends_on_apogee(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A ballistic step that passes apogee is cut at apogee and reports it."""
    config = IntegrationConfiguration(gravity=ConstantGravity(9.81))

    state, dt_taken, _next_dt, hit = adaptive_step(
        0.0,
        _coasting_state(50.0),
        baseline_rocket_properties,
        config,
        scalar(10.0, "s"),
        events=(APOGEE,),
    )

    assert hit is APOGEE
    assert dt_taken.m_as("s") == pytest.approx(50.0 / 9.81, abs=1e-8)
    assert state.velocity.m_as("m/s")[0] == pytest.approx(0.0, abs=1e-7)


def test_adaptive_step_reports_no_event_when_none_is_crossed(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A step that crosses nothing runs its full length and reports None."""
    config = IntegrationConfiguration(gravity=ConstantGravity(9.81))

    _state, dt_taken, _next_dt, hit = adaptive_step(
        0.0,
        _coasting_state(50.0),
        baseline_rocket_properties,
        config,
        scalar(1.0, "s"),
        events=(APOGEE,),
    )

    assert hit is None
    assert dt_taken.m_as("s") == pytest.approx(1.0)


def test_peak_vertical_velocity_lands_where_acceleration_is_zero(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Stepping through the burn ends a step where vertical acceleration is zero."""
    config = IntegrationConfiguration(
        gravity=ConstantGravity(9.81), atmosphere=VacuumAtmosphere()
    )
    peak = peak_vertical_velocity(baseline_rocket_properties, config)
    state = _coasting_state(100.0)
    state.current_mass = scalar(20.0, "kg")

    # Thrust fades from 2 s to 4 s, so the acceleration crosses zero in between
    time = 2.0
    dt = scalar(0.5, "s")
    hit = None
    while hit is None:
        state, dt_taken, dt, hit = adaptive_step(
            time, state, baseline_rocket_properties, config, dt, events=(peak,)
        )
        time += float(dt_taken.m_as("s"))
        assert time < 4.0

    assert hit is peak
    assert peak.value(time, state) == pytest.approx(0.0, abs=1e-6)


def test_flight_event_is_immutable() -> None:
    """Events are frozen so a shared constant cannot be altered."""
    with pytest.raises(AttributeError):
        APOGEE.name = "changed"  # type: ignore[misc]


def test_custom_event() -> None:
    """Any positive-to-non-positive quantity of time and state is an event."""
    burnout = FlightEvent("burnout", lambda time, _state: 4.0 - time)

    assert burnout.crossed(3.0, _coasting_state(0.0), 5.0, _coasting_state(0.0))
