"""Mission tests."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from support import truth

from flight_sim.environment.launch_rail import LaunchRail, NoRail
from flight_sim.events import APOGEE, at_time
from flight_sim.flight_event import EventTime, FlightEvent
from flight_sim.integration import IntegrationConfiguration
from flight_sim.mission import Mission
from flight_sim.simulation import Simulation
from flight_sim.units import scalar
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.rocket_properties import RocketProperties

_RAIL = LaunchRail(
    length=scalar(5.0, "m"),
    friction_coefficient=0.0,
    elevation=scalar(85.0, "deg"),
)


def _to_apogee(properties: RocketProperties) -> Simulation:
    """Return the rocket flown off the rail to apogee in one simulation."""
    return Simulation(
        properties=properties,
        config=IntegrationConfiguration(truth=truth(launch_rail=_RAIL)),
        events=(APOGEE,),
        stop_event="apogee",
        max_time=scalar(120.0, "s"),
    )


def _off_rail(simulation: Simulation, **changes: Any) -> Simulation:
    """Return a simulation with no rail, changed by ``changes``."""
    config = replace(
        simulation.config, truth=replace(simulation.config.truth, launch_rail=NoRail())
    )
    return replace(simulation, config=config, **changes)


def test_rail_then_free_flight_matches_one_simulation(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Splitting the flight at rail exit gives the same flight."""
    whole = _to_apogee(baseline_rocket_properties)
    mission = Mission(
        segments=(replace(whole, stop_event="rail exit"), _off_rail(whole)),
        name="split",
    )

    expected = whole.run()
    flight = mission.run()

    assert list(flight.events) == ["rail exit", "apogee"]
    assert np.all(np.diff(flight.time.m_as("s")) > 0.0)
    assert flight.simulation.name == "split"
    for name in ("rail exit", "apogee"):
        assert flight.at(name).time.m_as("s") == pytest.approx(
            expected.at(name).time.m_as("s"), rel=1e-3
        )
    assert flight.summary()["apogee_m"] == pytest.approx(
        expected.summary()["apogee_m"], rel=1e-3
    )


def test_motor_lit_by_an_earlier_segment_burns_on(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A motor whose event fired in an earlier segment is lit in the next."""
    assert isinstance(baseline_rocket_properties.engine, SolidEngine)
    properties = replace(
        baseline_rocket_properties,
        engine=replace(
            baseline_rocket_properties.engine,
            ignition_time=EventTime("go", scalar(0.2, "s")),
        ),
    )
    first = _off_rail(
        _to_apogee(properties),
        events=(at_time("go", scalar(0.1, "s")),),
        stop_event="go",
    )
    second = _off_rail(
        _to_apogee(properties),
        events=(at_time("stop", scalar(0.6, "s")),),
        stop_event="stop",
    )

    flight = Mission(segments=(first, second)).run()

    time = flight.time.m_as("s")
    thrust = flight["thrust"].m_as("N")
    assert list(flight.events) == ["go", "stop"]
    assert np.all(thrust[time < 0.3] == 0.0)
    assert np.all(thrust[time > 0.3] > 0.0)


def test_later_segment_with_a_rail_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Only the first segment may start on a rail."""
    whole = _to_apogee(baseline_rocket_properties)

    with pytest.raises(ValueError, match="Segment 1 has a launch rail"):
        Mission(segments=(replace(whole, stop_event="rail exit"), whole))


def test_unknown_ignition_event_is_rejected_before_flying(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A motor waiting on an event no segment has is rejected at once."""
    assert isinstance(baseline_rocket_properties.engine, SolidEngine)
    properties = replace(
        baseline_rocket_properties,
        engine=replace(
            baseline_rocket_properties.engine, ignition_time=EventTime("go")
        ),
    )

    with pytest.raises(ValueError, match="Segment 0 ignition event 'go'"):
        Mission(segments=(_to_apogee(properties),))


def test_segment_stopping_on_the_rail_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A rocket still on the rail is not freed by the next segment."""
    whole = _to_apogee(baseline_rocket_properties)
    mission = Mission(
        segments=(
            replace(whole, events=(at_time("a", scalar(0.05, "s")),), stop_event="a"),
            _off_rail(whole),
        )
    )

    with pytest.raises(ValueError, match="still on the rail"):
        mission.run()


def test_later_stop_event_firing_early_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Segments' stop events must fire in the order the segments are flown."""
    whole = _off_rail(_to_apogee(baseline_rocket_properties))
    deploy = at_time("deploy", scalar(0.2, "s"))
    mission = Mission(
        segments=(
            replace(
                whole,
                events=(at_time("burnout", scalar(0.5, "s")), deploy),
                stop_event="burnout",
            ),
            replace(whole, events=(deploy,), stop_event="deploy"),
        )
    )

    with pytest.raises(ValueError, match=r"\['deploy'\] fired during segment 0"):
        mission.run()


def test_stop_event_already_behind_a_segment_fails_at_once(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A segment whose stop event is in the past fails before flying."""
    whole = _off_rail(_to_apogee(baseline_rocket_properties))
    mission = Mission(
        segments=(
            replace(whole, events=(at_time("a", scalar(0.5, "s")),), stop_event="a"),
            replace(whole, events=(at_time("b", scalar(0.2, "s")),), stop_event="b"),
        )
    )

    with pytest.raises(ValueError, match="Stop event 'b' is already behind"):
        mission.run()


def test_stop_event_on_a_motor_lit_earlier_is_judged_lit(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A thrust-based stop event sees the motor an earlier segment lit."""
    assert isinstance(baseline_rocket_properties.engine, SolidEngine)
    properties = replace(
        baseline_rocket_properties,
        engine=replace(
            baseline_rocket_properties.engine, ignition_time=EventTime("go")
        ),
    )
    burnout = FlightEvent(
        "burnout",
        lambda time, state: properties.engine.get_thrust(time, 101325.0) - 1.0,
    )
    first = _off_rail(
        _to_apogee(properties),
        events=(at_time("go", scalar(0.1, "s")),),
        stop_event="go",
    )
    second = _off_rail(_to_apogee(properties), events=(burnout,), stop_event="burnout")

    flight = Mission(segments=(first, second)).run()

    assert list(flight.events) == ["go", "burnout"]
    assert flight.at("burnout").time.m_as("s") > 2.0
