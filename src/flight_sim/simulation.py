"""One flight from launch to its stop event, as a sim input file describes it."""

import itertools
import re
import time as clock
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from flight_sim import output
from flight_sim.config import InputSource, build_with_source, load_with_source
from flight_sim.flight_event import FlightEvent
from flight_sim.integration import (
    IntegrationConfiguration,
    IntegrationStats,
    adaptive_step,
    derivative_computation,
)
from flight_sim.output import OutputConfiguration
from flight_sim.telemetry import Channel, Recorder, resample
from flight_sim.units import Scalar, UnitChecked, UnitSystem, scalar, system_units
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

    @classmethod
    def from_table(cls, table: Mapping[str, Any], path: str | Path) -> Self:
        """Build a simulation from a resolved input table, keeping its source.

        Args:
            table (Mapping[str, Any]): The table, such as one
                ``source.table_with`` returns.
            path (str | Path): The sim input file the table came from.

        Returns:
            Self: The simulation, with ``source`` describing the table.

        Raises:
            ConfigError: If the table cannot be built.
        """
        simulation, source = build_with_source(table, path, cls)
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
            Flight: Every channel at launch and after every step, with the
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
        recorder = Recorder()
        event_rows: dict[str, int] = {}
        hit: FlightEvent | None = None

        try:
            for row in itertools.count():
                with recorder.step():
                    derivative_computation(time, state, self.properties, self.config)
                if hit is not None:
                    event_rows.setdefault(hit.name, row)
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
            recorder.channels(),
            event_rows,
            stats,
            run_time_s=clock.perf_counter() - start,
            units=self.output.units,
        )


@dataclass(frozen=True)
class Flight:
    """What one run of a simulation recorded: every channel at each step.

    ``flight["aero.force"]`` reads a channel at every step as a quantity
    array, ``at`` reads every channel at one moment, and ``to_dataframe``
    lays them out as columns. Channels, tables, figures and the summary are
    in the flight's ``units``.
    """

    simulation: Simulation

    # Recorded in the units they were recorded with, one row per step
    channels: Mapping[str, Channel]

    # Row of the first time each event fired, in firing order
    events: Mapping[str, int]

    stats: IntegrationStats = field(default_factory=IntegrationStats)

    # Wall time the run took, in seconds
    run_time_s: float = 0.0

    units: UnitSystem = "metric"

    def __getitem__(self, name: str) -> Any:
        """Return one channel at every step, as a quantity array."""
        return self.channels[name].quantity(self.units)

    def at(self, moment: str | Scalar) -> dict[str, Any]:
        """Read every channel at an event, or at a time between steps.

        Args:
            moment (str | Scalar): Event name, or a time within the flight.

        Returns:
            dict[str, Any]: Each channel's quantity at that moment, read
                between steps as its ``interp`` says.

        Raises:
            KeyError: If the flight hit no event of that name.
            ValueError: If the time lies outside the flight.
        """
        times = self.channels["time"].values
        if isinstance(moment, str):
            time_s = float(times[self.events[moment]])
        else:
            time_s = float(moment.m_as("s"))
            if not times[0] <= time_s <= times[-1]:
                raise ValueError(
                    f"{time_s} s is outside the flight, {times[0]} to {times[-1]} s"
                )
        sample = resample(self.channels, np.array([time_s]))
        return {
            name: channel.quantity(self.units)[0] for name, channel in sample.items()
        }

    def to_dataframe(self, sample_interval: Scalar | None = None) -> pd.DataFrame:
        """Lay every channel out as one column per component, plus "event".

        Args:
            sample_interval (Scalar | None, optional): Time between rows, with
                a row kept at each event and at the end. Between steps,
                channels are read as their ``interp`` says. Defaults to one
                row per step.

        Returns:
            pd.DataFrame: Columns such as "aero.force_x [N]", and "event"
                naming the events on each row.
        """
        channels, rows = self.channels, dict(self.events)
        if sample_interval is not None:
            times = channels["time"].values
            grid = np.union1d(
                np.arange(times[0], times[-1], float(sample_interval.m_as("s"))),
                np.append(times[list(rows.values())], times[-1]),
            )
            rows = {
                name: int(np.searchsorted(grid, times[row]))
                for name, row in rows.items()
            }
            channels = resample(channels, grid)
        data: dict[str, Any] = {}
        for name, channel in channels.items():
            units = system_units(channel.units, self.units) or "-"
            values = channel.quantity(self.units).magnitude
            if values.ndim == 1:
                data[f"{name} [{units}]"] = values
                continue
            flat = values.reshape(len(values), -1)
            for index, suffix in enumerate(_component_names(channel)):
                data[f"{name}_{suffix} [{units}]"] = flat[:, index]
        labels = [""] * len(channels["time"].values)
        for name, row in rows.items():
            labels[row] = f"{labels[row]}, {name}" if labels[row] else name
        data["event"] = labels
        return pd.DataFrame(data)

    def summary(self) -> dict[str, float]:
        """Return headline numbers of the flight, keyed with their units."""
        headline = {
            "flight_time": self["time"][-1],
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
        times = self.channels["time"].values
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


def _component_names(channel: Channel) -> list[str]:
    """Return column suffixes for the components of an array channel."""
    shape = channel.values.shape[1:]
    if channel.interp == "slerp":
        return ["w", "x", "y", "z"]
    if shape == (3,):
        return ["x", "y", "z"]
    if shape == (3, 3):
        return [row + col for row in "xyz" for col in "xyz"]
    return ["_".join(map(str, index)) for index in np.ndindex(shape)]


def _with_units(name: str, value: Any) -> str:
    """Return a key naming a quantity's units, such as "max_speed_ft_s"."""
    suffix = re.sub(r"[^a-z0-9]+", "_", f"{value.units:~}".lower()).strip("_")
    return f"{name}_{suffix}" if suffix else name
