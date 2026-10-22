"""One flight from launch to its stop event, as a sim input file describes it."""

import re
import time as clock
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from flight_sim import output, telemetry
from flight_sim.config import InputSource, load_with_source
from flight_sim.flight_event import FlightEvent
from flight_sim.flight_point import FlightPointSI
from flight_sim.integration import (
    IntegrationConfiguration,
    IntegrationStats,
    adaptive_step,
    flight_point_si,
)
from flight_sim.output import OutputConfiguration
from flight_sim.telemetry import FlightPoint
from flight_sim.units import Scalar, UnitChecked, UnitSystem, scalar
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


@dataclass
class Simulation(UnitChecked):
    """Rocket, world and events of one flight, and the event that ends it.

    The launch rail holds the initial state. A motor lit by an event ignites
    its delay after the first time that event fires. ``from_file`` also
    keeps the input it was built from as ``source``, which ``output.write``
    copies next to the telemetry.
    """

    properties: RocketProperties
    config: IntegrationConfiguration
    stop_event: str

    # Sim time by which the stop event must fire, or the run fails
    max_time: Annotated[Scalar, "s"]

    events: Sequence[FlightEvent] = ()
    initial_state: RocketState = field(default_factory=RocketState)

    # First step length the integrator tries
    initial_time_step: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(0.01, "s")
    )
    name: str = ""
    output: OutputConfiguration = field(default_factory=OutputConfiguration)

    source: InputSource | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Build a simulation from a sim input file, keeping its source.

        Raises:
            ConfigError: If the file cannot be read or built.
        """
        simulation, source = load_with_source(path, cls)
        simulation.source = source
        return simulation

    def __post_init__(self) -> None:
        """Check that the stop and ignition events are among the flight's events.

        Raises:
            ValueError: If either names an event the flight does not have.
        """
        super().__post_init__()
        names = self.event_names()
        if self.stop_event not in names:
            raise ValueError(f"Stop event {self.stop_event!r} is not one of {names}")
        ignition = self.properties.engine.pending_ignition
        if ignition is not None and ignition.event not in names:
            raise ValueError(f"Ignition event {ignition.event!r} is not one of {names}")

    def event_names(self) -> tuple[str, ...]:
        """Return the names of the events the flight can hit, the rail's first."""
        events = (*self.config.truth.launch_rail.events, *self.events)
        return tuple(event.name for event in events)

    def run(self) -> "Flight":
        """Fly from the initial state until the stop event.

        The engine is restored afterwards, so the flight can be run again.

        Returns:
            Flight: A flight point at launch and after every step, with the
                integrator's work and the wall time the run took.

        Raises:
            RuntimeError: If the stop event has not fired by ``max_time``.
        """
        start = clock.perf_counter()
        stats = IntegrationStats()
        state = self.config.truth.launch_rail.mount(self.initial_state)
        engine = self.properties.engine
        ignition = engine.pending_ignition
        dt = self.initial_time_step
        max_time_s = float(self.max_time.m_as("s"))
        time = 0.0
        points: list[FlightPointSI] = []
        event_rows: dict[str, int] = {}
        hit: FlightEvent | None = None

        try:
            while True:
                points.append(
                    flight_point_si(time, state, self.properties, self.config)
                )
                if hit is not None:
                    event_rows.setdefault(hit.name, len(points) - 1)
                    if hit.name == self.stop_event:
                        break
                    if ignition is not None and hit.name == ignition.event:
                        # Events built from the properties see the lit motor too
                        self.properties.engine = engine.ignited(
                            time + float(ignition.delay.m_as("s"))
                        )
                        ignition = None
                if time > max_time_s:
                    raise RuntimeError(
                        f"Stop event {self.stop_event!r} did not fire by the max"
                        f" time of {max_time_s} s"
                    )

                state, dt_taken, dt, hit = adaptive_step(
                    time,
                    state,
                    self.properties,
                    self.config,
                    dt,
                    events=self.events,
                    stats=stats,
                )
                time += float(dt_taken.m_as("s"))
        finally:
            self.properties.engine = engine

        return Flight(
            self,
            points,
            event_rows,
            stats,
            run_time_s=clock.perf_counter() - start,
            units=self.output.units,
        )


@dataclass(frozen=True)
class Flight:
    """What one run of a simulation recorded: a flight point per accepted step.

    ``flight["aero.force"]`` reads a channel of ``telemetry.CHANNELS`` at
    every point, as a quantity array. ``at`` and ``point`` give one moment as
    a ``FlightPoint``, and ``to_dataframe`` lays channels out as columns.
    Channels, tables, figures and the summary are in the flight's ``units``.
    """

    simulation: Simulation
    points: Sequence[FlightPointSI]

    # Row of the first time each event fired, in firing order
    events: Mapping[str, int]

    stats: IntegrationStats = field(default_factory=IntegrationStats)

    # Wall time the run took, in seconds
    run_time_s: float = 0.0

    units: UnitSystem = "metric"

    _columns: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __len__(self) -> int:
        """Return the number of recorded points."""
        return len(self.points)

    def __getitem__(self, channel: str) -> Any:
        """Return one channel at every point, as a quantity array.

        Raises:
            KeyError: If no channel has that name.
        """
        if channel not in self._columns:
            self._columns[channel] = telemetry.column(self.points, channel, self.units)
        return self._columns[channel]

    @property
    def time(self) -> Any:
        """Return the time of every point, as a quantity array."""
        return self["time"]

    @property
    def orientations(self) -> list[Quaternion]:
        """Return the body-to-world orientation at every point."""
        return [point.state.orientation for point in self.points]

    def point(self, index: int) -> FlightPoint:
        """Return the typed view of one recorded point."""
        return FlightPoint.from_si(self.points[index])

    def at(self, moment: str | Scalar) -> FlightPoint:
        """Return the flight at an event, or at a time between recorded points.

        Args:
            moment (str | Scalar): Event name, or a time within the flight.
                Between points, the orientation is interpolated by slerp and
                everything else linearly.

        Returns:
            FlightPoint: The flight at that moment.

        Raises:
            KeyError: If the flight hit no event of that name.
            ValueError: If the time lies outside the flight.
        """
        if isinstance(moment, str):
            if moment not in self.events:
                raise KeyError(
                    f"No event {moment!r}; the flight hit {list(self.events)}"
                )
            return self.point(self.events[moment])
        return FlightPoint.from_si(self._point_at(float(moment.m_as("s"))))

    def _point_at(self, time_s: float) -> FlightPointSI:
        """Return the recorded point at a time, interpolating between points.

        Raises:
            ValueError: If the time lies outside the flight.
        """
        times = self.time.m_as("s")
        if not times[0] <= time_s <= times[-1]:
            raise ValueError(
                f"{time_s} s is outside the flight, {times[0]} to {times[-1]} s"
            )
        index = int(np.searchsorted(times, time_s, side="right")) - 1
        if index == len(times) - 1 or times[index] == time_s:
            return self.points[index]
        fraction = (time_s - times[index]) / (times[index + 1] - times[index])
        return telemetry.interpolate(
            self.points[index], self.points[index + 1], fraction
        )

    def to_dataframe(
        self,
        channels: Sequence[str] = ("*",),
        sample_interval: Scalar | None = None,
    ) -> pd.DataFrame:
        """Lay channels out as one column per component, plus an "event" column.

        Args:
            channels (Sequence[str], optional): Glob patterns of the channels
                to include, such as "aero.*". Defaults to every channel.
            sample_interval (Scalar | None, optional): Time between rows, with
                a row kept at each event and at the end. Defaults to one row
                per recorded point.

        Returns:
            pd.DataFrame: One row per recorded point, or per sample.
        """
        interval_s = (
            float(sample_interval.m_as("s")) if sample_interval is not None else None
        )
        return telemetry.table(
            self.points, self.events, channels, interval_s, self.units
        )

    def summary(self) -> dict[str, float]:
        """Return headline numbers of the flight, keyed with their units."""
        headline = {
            "flight_time": self.time[-1],
            "apogee": self["state.position"][:, 0].max(),
            "max_speed": self["derived.speed"].max(),
            "max_mach": self["aero.mach"].max(),
            "max_dynamic_pressure": self["aero.dynamic_pressure"].max(),
        }
        return {
            _with_units(name, value): float(value.magnitude)
            for name, value in headline.items()
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return the integrator's work and the wall time the run took."""
        times = self.time.m_as("s")
        steps = np.diff(times)
        result: dict[str, Any] = {
            "steps": len(steps),
            "rejected_steps": self.stats.rejected_steps,
            "rkf45_evaluations": self.stats.rkf45_evaluations,
            "event_iterations": self.stats.event_iterations,
            "min_step_s": float(steps.min()) if len(steps) else 0.0,
            "max_step_s": float(steps.max()) if len(steps) else 0.0,
            "run_time_s": self.run_time_s,
            "real_time_factor": (
                float(times[-1] - times[0]) / self.run_time_s
                if self.run_time_s > 0.0
                else 0.0
            ),
        }
        if self.simulation.source is not None:
            result["load_time_s"] = self.simulation.source.load_time_s
        return result

    def figures(self, names: Sequence[str] | None = None) -> dict[str, Figure]:
        """Draw named figures of the flight.

        Args:
            names (Sequence[str] | None, optional): Names of functions in
                ``flight_sim.plots``.
                Defaults to the simulation's ``output.plots``.

        Returns:
            dict[str, Figure]: Each figure by name.
        """
        if names is None:
            names = self.simulation.output.plots
        return {
            name: output.plot_function(name)(self, self.simulation.name)
            for name in names
        }


def _with_units(name: str, value: Any) -> str:
    """Return a key naming a quantity's units, such as "max_speed_ft_s"."""
    suffix = re.sub(r"[^a-z0-9]+", "_", f"{value.units:~}".lower()).strip("_")
    return f"{name}_{suffix}" if suffix else name
