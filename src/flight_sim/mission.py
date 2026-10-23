"""A flight flown as a chain of simulations, each from where the last stopped."""

import time as clock
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Self

from flight_sim.config import InputSource, load_with_source
from flight_sim.environment.launch_rail import NoRail
from flight_sim.flight_point import FlightPointSI
from flight_sim.integration import IntegrationStats
from flight_sim.output import OutputConfiguration
from flight_sim.simulation import Flight, Simulation
from flight_sim.vehicle.rocket_state import RocketState


@dataclass
class Mission:
    """Segments flown in order, joined into one flight.

    Each segment is a ``Simulation`` with its own rocket, world and stop
    event. The first flies from its initial state; each later one starts
    from the state, time and fired events the one before stopped at, and so
    must use ``NoRail``. One clock runs through every segment, so thrust
    curves, ``at_time`` events and each ``max_time`` are mission times.
    Segment ``name`` and ``output`` are ignored.
    """

    segments: Sequence[Simulation]
    name: str = ""
    output: OutputConfiguration = field(default_factory=OutputConfiguration)

    source: InputSource | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Build a mission from a file of ``[[segments]]``, keeping its source.

        Raises:
            ConfigError: If the file cannot be read or built.
        """
        mission, source = load_with_source(path, cls)
        mission.source = source
        return mission

    def __post_init__(self) -> None:
        """Check the segments can follow each other.

        Raises:
            ValueError: If there are none, a later one has a rail, or a motor
                waits on an event no segment up to its own has.
        """
        if not self.segments:
            raise ValueError("A mission needs at least one segment")
        names: tuple[str, ...] = ()
        for index, segment in enumerate(self.segments):
            if index and not isinstance(segment.config.truth.launch_rail, NoRail):
                raise ValueError(
                    f"Segment {index} has a launch rail; only the first can, so"
                    ' later ones need launch_rail = { type = "flight_sim.'
                    'environment.launch_rail.NoRail" }'
                )
            names += segment.event_names()
            ignition = segment.properties.engine.pending_ignition
            if ignition is not None and ignition.event not in names:
                raise ValueError(
                    f"Segment {index} ignition event {ignition.event!r} is not"
                    f" one of {names}"
                )

    @property
    def stop_event(self) -> str:
        """Return the event that ends the mission, the last segment's."""
        return self.segments[-1].stop_event

    def run(self) -> Flight:
        """Fly every segment in turn and join them into one flight.

        Returns:
            Flight: The joined points, events and integrator work. Its
                ``simulation`` is the first segment with the mission's name,
                output and source.

        Raises:
            ValueError: If a segment stops with the rocket still on the rail,
                a later segment's stop event fires before its own, or a
                segment's stop event is already behind its start.
            RuntimeError: If a segment's stop event has not fired by its
                ``max_time``.
        """
        start = clock.perf_counter()
        points: list[FlightPointSI] = []
        events: dict[str, int] = {}
        fired: dict[str, float] = {}
        stats = IntegrationStats()
        state: RocketState | None = None
        for index, segment in enumerate(self.segments):
            if state is not None:
                if state.rail_start is not None:
                    raise ValueError(
                        f"Segment {index - 1} stopped at {points[-1].time} s with"
                        " the rocket still on the rail; stop it at 'rail exit' or"
                        " later"
                    )
                segment = replace(segment, initial_state=state)
            flight = segment.run(points[-1].time if points else 0.0, fired)
            # shortcut: only events this segment has are watched, so a later
            # stop event only its own segment has is caught when that one starts
            later = {other.stop_event for other in self.segments[index + 1 :]}
            early = (later - {segment.stop_event}) & flight.events.keys()
            if early:
                raise ValueError(
                    f"{sorted(early)} fired during segment {index}, before its stop"
                    f" event {segment.stop_event!r}; segments must stop in order"
                )
            # Each later segment's first point repeats the last one's end
            skip = 1 if points else 0
            for name, row in flight.events.items():
                events.setdefault(name, len(points) + row - skip)
                fired.setdefault(name, flight.points[row].time)
            points.extend(flight.points[skip:])
            for counter in fields(stats):
                setattr(
                    stats,
                    counter.name,
                    getattr(stats, counter.name) + getattr(flight.stats, counter.name),
                )
            state = flight.point(-1).state

        simulation = replace(self.segments[0], name=self.name, output=self.output)
        simulation.source = self.source
        return Flight(
            simulation,
            points,
            events,
            stats,
            run_time_s=clock.perf_counter() - start,
            units=self.output.units,
        )
