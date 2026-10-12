"""One flight from launch to its stop event, as a sim input file describes it."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.events import at_time
from flight_sim.flight_event import FlightEvent
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import Scalar, UnitChecked, scalar
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


class FlightRecord(NamedTuple):
    """What a flight produced."""

    telemetry: pd.DataFrame  # One row per step, starting at launch
    step_lengths: list[float]  # s
    events: list[tuple[str, float]]  # Name and time in s of each event hit


@dataclass
class Simulation(UnitChecked):
    """Rocket, world and events of one flight, and the event that ends it.

    A configured launch rail holds the initial state. A motor lit by an event
    ignites its delay after the first time that event fires.
    """

    properties: RocketProperties
    config: IntegrationConfiguration = field(default_factory=IntegrationConfiguration)
    events: Sequence[FlightEvent] = ()
    initial_state: RocketState = field(default_factory=RocketState)

    # First step length the integrator tries
    initial_time_step: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(0.01, "s")
    )
    stop_event: str = "impact"
    name: str = ""

    def __post_init__(self) -> None:
        """Check that the stop and ignition events are among the flight's events.

        Raises:
            ValueError: If either names an event the flight does not have.
        """
        super().__post_init__()
        names = self.event_names()
        if self.stop_event not in names:
            raise ValueError(f"Stop event {self.stop_event!r} is not one of {names}")
        # Engine.pending_ignition returns None; SolidEngine overrides it
        ignition = self.properties.engine.pending_ignition()  # pylint: disable=assignment-from-none
        if ignition is not None and ignition.event not in names:
            raise ValueError(f"Ignition event {ignition.event!r} is not one of {names}")

    def event_names(self) -> tuple[str, ...]:
        """Return the names of the events the flight can hit, the rail's first."""
        rail = self.config.truth.launch_rail
        rail_names = (rail.exit_event.name,) if rail is not None else ()
        return (*rail_names, *(event.name for event in self.events))

    def run(self) -> FlightRecord:
        """Fly from the initial state until the stop event.

        The engine is restored afterwards, so the flight can be run again.

        Returns:
            FlightRecord: Telemetry, step lengths and events of the flight.
        """
        rail = self.config.truth.launch_rail
        state = (
            rail.mount(self.initial_state) if rail is not None else self.initial_state
        )
        engine = self.properties.engine
        ignition = engine.pending_ignition()  # pylint: disable=assignment-from-none
        events = list(self.events)
        dt = self.initial_time_step
        time = 0.0
        record = FlightRecord(pd.DataFrame(), [], [])
        rows = []
        hit: FlightEvent | None = None

        try:
            while True:
                rows.append(_telemetry_row(time, state, self.properties))
                if hit is not None:
                    print(f"{hit.name} at {time:.2f} seconds")
                    record.events.append((hit.name, time))
                    if hit.name == self.stop_event:
                        break
                    if ignition is not None and hit.name == ignition.event:
                        # Events built from the properties see the lit motor too
                        ignition_s = time + float(ignition.delay.m_as("s"))
                        self.properties.engine = engine.ignited(ignition_s)
                        events.append(at_time("ignition", scalar(ignition_s, "s")))
                        ignition = None

                state, dt_taken, dt, hit = adaptive_step(
                    time, state, self.properties, self.config, dt, events=events
                )
                time += float(dt_taken.m_as("s"))
                record.step_lengths.append(float(dt_taken.m_as("s")))
        finally:
            self.properties.engine = engine

        return record._replace(telemetry=pd.DataFrame(rows))


def _telemetry_row(
    time: float, state: RocketState, properties: RocketProperties
) -> dict[str, object]:
    """Return one telemetry row: time, state and mass, in SI units."""
    return {
        "Time (s)": time,
        "Inertial Position (m)": np.array(state.position.m_as("m")),
        "Inertial Velocity (m/s)": np.array(state.velocity.m_as("m/s")),
        "Angular Rate (rad/s)": np.array(state.angular_velocity.m_as("rad/s")),
        "Mass (kg)": properties.mass_properties(time).mass,
    }
