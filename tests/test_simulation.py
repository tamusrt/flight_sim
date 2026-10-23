"""Simulation runner tests."""

from dataclasses import replace

import numpy as np
import pytest
from support import truth

from flight_sim.environment.gravity import ConstantGravity
from flight_sim.events import at_time
from flight_sim.flight_event import EventTime
from flight_sim.integration import IntegrationConfiguration, flight_point_si
from flight_sim.simulation import Simulation
from flight_sim.units import scalar
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.rocket_properties import RocketProperties

_WEIGHTLESS = IntegrationConfiguration(truth=truth(gravity=ConstantGravity(0.0)))


def _event_lit(properties: RocketProperties, delay_s: float) -> RocketProperties:
    """Return the rocket with its motor lit a delay after the "go" event."""
    assert isinstance(properties.engine, SolidEngine)
    engine = replace(
        properties.engine, ignition_time=EventTime("go", scalar(delay_s, "s"))
    )
    return replace(properties, engine=engine)


def test_event_ignition_lights_the_motor_after_its_delay(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The motor lights its delay after the event."""
    simulation = Simulation(
        properties=_event_lit(baseline_rocket_properties, 0.2),
        config=_WEIGHTLESS,
        events=(at_time("go", scalar(0.1, "s")), at_time("stop", scalar(0.6, "s"))),
        stop_event="stop",
        max_time=scalar(1.0, "s"),
    )

    flight = simulation.run()

    times = [flight.at(name).time.m_as("s") for name in flight.events]
    assert list(flight.events) == ["go", "stop"]
    assert times == pytest.approx([0.1, 0.6], abs=1e-6)
    time = flight.time.m_as("s")
    mass = flight["mass.mass"].m_as("kg")
    thrust = flight["thrust"].m_as("N")
    assert np.unique(mass[time <= 0.3]).size == 1
    assert np.all(thrust[time < 0.3] == 0.0)
    assert np.all(thrust[time > 0.3] > 0.0)
    assert mass[-1] < mass[0]
    assert flight["state.position"].m_as("m")[-1, 0] > 0.0
    # The unlit motor is restored for the next run
    assert simulation.properties.engine.pending_ignition is not None


def test_zero_delay_ignition_lights_at_the_event(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """With no delay the motor lights at the event itself."""
    simulation = Simulation(
        properties=_event_lit(baseline_rocket_properties, 0.0),
        config=_WEIGHTLESS,
        events=(at_time("go", scalar(0.1, "s")), at_time("stop", scalar(0.3, "s"))),
        stop_event="stop",
        max_time=scalar(1.0, "s"),
    )

    flight = simulation.run()

    assert list(flight.events) == ["go", "stop"]
    assert flight["state.position"].m_as("m")[-1, 0] > 0.0


def test_unknown_stop_event_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A stop event the flight does not have is rejected."""
    with pytest.raises(ValueError, match="Stop event 'impact'"):
        Simulation(
            properties=baseline_rocket_properties,
            config=_WEIGHTLESS,
            stop_event="impact",
            max_time=scalar(1.0, "s"),
        )


def test_no_rail_adds_no_events(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Without a rail the flight's events are only those it lists."""
    simulation = Simulation(
        properties=baseline_rocket_properties,
        config=_WEIGHTLESS,
        stop_event="stop",
        max_time=scalar(1.0, "s"),
        events=(at_time("stop", scalar(0.1, "s")),),
    )

    assert simulation.event_names() == ("stop",)
    assert list(simulation.run().events) == ["stop"]


def test_run_fails_when_the_stop_event_misses_the_max_time(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A stop event that has not fired by the max time fails the run."""
    simulation = Simulation(
        properties=_event_lit(baseline_rocket_properties, 0.0),
        config=_WEIGHTLESS,
        stop_event="stop",
        max_time=scalar(0.2, "s"),
        events=(at_time("go", scalar(1.0, "s")), at_time("stop", scalar(2.0, "s"))),
    )

    with pytest.raises(RuntimeError, match="'stop' did not fire by the max time"):
        simulation.run()
    assert simulation.properties.engine.pending_ignition is not None


def test_unknown_ignition_event_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """An ignition event the flight does not have, nor had fire, is rejected."""
    simulation = Simulation(
        properties=_event_lit(baseline_rocket_properties, 0.0),
        config=_WEIGHTLESS,
        events=(at_time("stop", scalar(1.0, "s")),),
        stop_event="stop",
        max_time=scalar(1.0, "s"),
    )

    with pytest.raises(ValueError, match="Ignition event 'go'"):
        simulation.run()


def test_already_fired_ignition_event_lights_from_its_time(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A run starting later lights a motor whose event fired before it."""
    simulation = Simulation(
        properties=_event_lit(baseline_rocket_properties, 0.2),
        config=_WEIGHTLESS,
        events=(at_time("stop", scalar(1.0, "s")),),
        stop_event="stop",
        max_time=scalar(2.0, "s"),
    )

    went = flight_point_si(
        0.4, simulation.initial_state, simulation.properties, simulation.config
    )
    flight = simulation.run(start_time=0.5, fired_events={"go": went})

    time = flight.time.m_as("s")
    thrust = flight["thrust"].m_as("N")
    assert time[0] == 0.5
    assert flight.at("stop").time.m_as("s") == pytest.approx(1.0, abs=1e-6)
    assert np.all(thrust[time < 0.6] == 0.0)
    assert np.all(thrust[time > 0.6] > 0.0)
    assert simulation.properties.engine.pending_ignition is not None


def test_engine_without_event_ignition_cannot_be_lit(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """An engine with no event ignition refuses to be lit by one."""
    engine = unpowered_rocket_properties.engine

    assert engine.pending_ignition is None
    with pytest.raises(NotImplementedError):
        engine.ignited(1.0)
