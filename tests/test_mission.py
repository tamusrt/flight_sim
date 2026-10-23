"""Mission tests."""

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
from support import truth

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.environment.launch_rail import LaunchRail, NoRail
from flight_sim.events import APOGEE, at_time
from flight_sim.flight_event import EventTime, FlightEvent
from flight_sim.flight_point import AeroPointSI, FlightPointSI
from flight_sim.integration import IntegrationConfiguration, flight_point_si
from flight_sim.mission import Mission
from flight_sim.physics import RocketPhysics
from flight_sim.simulation import Simulation
from flight_sim.units import scalar, vector
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
    """Splitting the flight at its events gives exactly the same flight."""
    whole = _to_apogee(baseline_rocket_properties)
    whole = replace(whole, events=(*whole.events, at_time("cut", scalar(3.0, "s"))))
    mission = Mission(
        segments=(
            replace(whole, stop_event="rail exit"),
            _off_rail(whole, stop_event="cut"),
            _off_rail(whole),
        ),
        name="split",
    )

    expected = whole.run()
    flight = mission.run()

    assert list(flight.events) == ["rail exit", "cut", "apogee"]
    assert flight.simulation.name == "split"
    # Each segment goes on with the step the last would have taken
    np.testing.assert_array_equal(flight.time.m_as("s"), expected.time.m_as("s"))
    np.testing.assert_array_equal(
        flight["state.position"].m_as("m"), expected["state.position"].m_as("m")
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


@dataclass(frozen=True)
class _NoPowerOnDrag(RocketPhysics):
    """Table aero, with no force while the motor pushes."""

    def aero(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        time: float,
        air_velocity: np.ndarray,
        conditions: AtmosphereConditions,
        to_body: np.ndarray,
        cg_location: np.ndarray,
        properties: RocketProperties,
        thrust: float,
    ) -> AeroPointSI:
        aero = super().aero(
            time, air_velocity, conditions, to_body, cg_location, properties, thrust
        )
        return aero._replace(force=aero.force * 0.0) if thrust > 0.0 else aero


def test_segment_physics_drives_only_its_segment(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A segment's own physics sets its loads, and the next goes back to default."""
    whole = _to_apogee(baseline_rocket_properties)
    boost = replace(
        whole,
        config=replace(
            whole.config,
            truth=replace(whole.config.truth, launch_rail=NoRail()),
            physics=_NoPowerOnDrag(),
        ),
        events=(at_time("switch", scalar(1.0, "s")),),
        stop_event="switch",
    )
    mission = Mission(
        segments=(replace(whole, stop_event="rail exit"), boost, _off_rail(whole))
    )

    # The zeroed force no longer matches the table's coefficients
    with pytest.warns(RuntimeWarning, match="_NoPowerOnDrag gives an aero force"):
        flight = mission.run()

    time = flight.time.m_as("s")
    drag = flight["aero.force"].m_as("N")[:, 0]
    rail_exit = flight.at("rail exit").time.m_as("s")
    assert np.all(drag[(time > rail_exit) & (time <= 1.0)] == 0.0)
    assert np.all(drag[(time > 1.0) & (time < 3.0)] != 0.0)


@dataclass(frozen=True)
class _CutOff(RocketPhysics):
    """Motor thrust until the "cut" event, then none."""

    cut_at: float = math.inf

    def after(self, event: str, point: FlightPointSI) -> RocketPhysics:
        return replace(self, cut_at=point.time) if event == "cut" else self

    def thrust(
        self, time: float, conditions: AtmosphereConditions, properties: RocketProperties
    ) -> float:
        return 0.0 if time > self.cut_at else super().thrust(time, conditions, properties)


def test_physics_hears_events_in_its_segment_and_earlier_ones(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Physics changes at an event, and a later segment starts from it."""
    whole = _off_rail(_to_apogee(baseline_rocket_properties))
    whole = replace(whole, config=replace(whole.config, physics=_CutOff()))
    first = replace(
        whole,
        events=(at_time("cut", scalar(0.5, "s")), at_time("go", scalar(1.0, "s"))),
        stop_event="go",
    )
    second = replace(
        whole, events=(at_time("stop", scalar(1.5, "s")),), stop_event="stop"
    )

    flight = Mission(segments=(first, second)).run()

    time = flight.time.m_as("s")
    thrust = flight["thrust"].m_as("N")
    assert np.all(thrust[(time > 0.0) & (time <= 0.5)] > 0.0)
    assert np.all(thrust[time > 0.5] == 0.0)
    assert first.config.physics == _CutOff()


@dataclass(frozen=True)
class _Twisted(RocketPhysics):
    """Table aero with its torque doubled."""

    def aero(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        time: float,
        air_velocity: np.ndarray,
        conditions: AtmosphereConditions,
        to_body: np.ndarray,
        cg_location: np.ndarray,
        properties: RocketProperties,
        thrust: float,
    ) -> AeroPointSI:
        aero = super().aero(
            time, air_velocity, conditions, to_body, cg_location, properties, thrust
        )
        return aero._replace(torque=aero.torque * 2.0)


def test_torque_its_coefficients_do_not_explain_warns(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Telemetry's coefficients must explain the torque as well as the force."""
    whole = _to_apogee(baseline_rocket_properties)
    config = replace(whole.config, physics=_Twisted())
    state = replace(whole.initial_state, velocity=vector((100.0, 5.0, 0.0), "m/s"))

    with pytest.warns(RuntimeWarning, match="_Twisted gives an aero torque"):
        flight_point_si(0.0, state, baseline_rocket_properties, config)


class _Lost(RocketPhysics):
    """Physics whose ``after`` forgets to return one."""

    def after(self, event: str, point: FlightPointSI) -> RocketPhysics:
        return None  # type: ignore[return-value]


def test_after_returning_no_physics_fails(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A broken ``after`` stops the run instead of flying on without physics."""
    whole = _to_apogee(baseline_rocket_properties)
    segment = replace(whole, config=replace(whole.config, physics=_Lost()))

    with pytest.raises(TypeError, match="_Lost.after.*not a RocketPhysics"):
        segment.run()


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
