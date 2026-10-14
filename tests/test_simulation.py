"""Simulation runner tests."""

from dataclasses import replace

import numpy as np
import pytest

from flight_sim.environment.gravity import ConstantGravity
from flight_sim.events import at_time
from flight_sim.flight_event import EventTime
from flight_sim.integration import IntegrationConfiguration, TruthConfiguration
from flight_sim.simulation import Simulation
from flight_sim.units import scalar
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.rocket_properties import RocketProperties

_WEIGHTLESS = IntegrationConfiguration(
    truth=TruthConfiguration(gravity=ConstantGravity(0.0))
)


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
    )

    flight = simulation.run()

    assert list(flight.events) == ["go", "stop"]
    assert flight["state.position"].m_as("m")[-1, 0] > 0.0


def test_unknown_stop_event_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A stop event the flight does not have is rejected."""
    with pytest.raises(ValueError, match="Stop event 'impact'"):
        Simulation(properties=baseline_rocket_properties)


def test_unknown_ignition_event_is_rejected(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """An ignition event the flight does not have is rejected."""
    with pytest.raises(ValueError, match="Ignition event 'go'"):
        Simulation(
            properties=_event_lit(baseline_rocket_properties, 0.0),
            events=(at_time("stop", scalar(1.0, "s")),),
            stop_event="stop",
        )


def test_engine_without_event_ignition_cannot_be_lit(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """An engine with no event ignition refuses to be lit by one."""
    engine = unpowered_rocket_properties.engine

    assert engine.pending_ignition is None
    with pytest.raises(NotImplementedError):
        engine.ignited(1.0)
