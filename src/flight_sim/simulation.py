"""One flight from launch to its stop event, as a sim input file describes it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from flight_sim import output, telemetry
from flight_sim.config import load_with_input, write_toml
from flight_sim.events import at_time
from flight_sim.flight_event import FlightEvent
from flight_sim.flight_point import FlightPointSI
from flight_sim.integration import (
    IntegrationConfiguration,
    adaptive_step,
    flight_point_si,
)
from flight_sim.output import OutputConfiguration
from flight_sim.telemetry import FlightPoint
from flight_sim.units import Scalar, UnitChecked, scalar
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


@dataclass
class Simulation(UnitChecked):
    """Rocket, world and events of one flight, and the event that ends it.

    A configured launch rail holds the initial state. A motor lit by an event
    ignites its delay after the first time that event fires. ``from_file``
    also keeps the input file and its resolved table, which ``Flight.write``
    copies next to the telemetry.
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
    output: OutputConfiguration = field(default_factory=OutputConfiguration)

    source_file: Path | None = field(default=None, init=False, compare=False)
    resolved_input: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # Key paths in resolved_input that were left to their defaults
    input_defaults: frozenset[str] = field(
        default=frozenset(), init=False, repr=False, compare=False
    )

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Build a simulation from a sim input file, keeping the resolved input.

        Raises:
            ConfigError: If the file cannot be read or built.
        """
        simulation, table, defaults = load_with_input(path, cls)
        simulation.source_file = Path(path).resolve()
        simulation.resolved_input = table
        simulation.input_defaults = defaults
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
        # Engine.pending_ignition returns None; SolidEngine overrides it
        ignition = self.properties.engine.pending_ignition()  # pylint: disable=assignment-from-none
        if ignition is not None and ignition.event not in names:
            raise ValueError(f"Ignition event {ignition.event!r} is not one of {names}")

    def event_names(self) -> tuple[str, ...]:
        """Return the names of the events the flight can hit, the rail's first."""
        rail = self.config.truth.launch_rail
        rail_names = (rail.exit_event.name,) if rail is not None else ()
        return (*rail_names, *(event.name for event in self.events))

    def run(self) -> "Flight":
        """Fly from the initial state until the stop event.

        The engine is restored afterwards, so the flight can be run again.

        Returns:
            Flight: A flight point at launch and after every step.
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
        points: list[FlightPointSI] = []
        event_rows: dict[str, int] = {}
        hit: FlightEvent | None = None

        try:
            while True:
                points.append(
                    flight_point_si(time, state, self.properties, self.config)
                )
                if hit is not None:
                    print(f"{hit.name} at {time:.2f} seconds")
                    event_rows.setdefault(hit.name, len(points) - 1)
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
        finally:
            self.properties.engine = engine

        return Flight(self, points, event_rows)


@dataclass(frozen=True)
class Flight:
    """What one run of a simulation recorded: a flight point per accepted step.

    ``flight["aero.force"]`` reads a channel of ``telemetry.CHANNELS`` at
    every point, as a quantity array. ``at`` and ``point`` give one moment as
    a ``FlightPoint``, and ``to_dataframe`` lays channels out as columns.
    """

    simulation: Simulation
    points: Sequence[FlightPointSI]

    # Row of the first time each event fired, in firing order
    events: Mapping[str, int]

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
            self._columns[channel] = telemetry.column(self.points, channel)
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

    def resampled(self, interval: Scalar) -> "Flight":
        """Return the flight at a fixed spacing, keeping a row at every event.

        Args:
            interval (Scalar): Time between rows.

        Returns:
            Flight: Rows every ``interval`` from launch, plus one at each
                event and at the end.
        """
        times = self.time.m_as("s")
        event_times = times[list(self.events.values())]
        grid = np.union1d(
            np.arange(times[0], times[-1], float(interval.m_as("s"))),
            np.append(event_times, times[-1]),
        )
        events = {
            name: int(np.searchsorted(grid, times[row]))
            for name, row in self.events.items()
        }
        points = [self._point_at(float(time)) for time in grid]
        return Flight(self.simulation, points, events)

    def to_dataframe(self, channels: Sequence[str] = ("*",)) -> pd.DataFrame:
        """Lay channels out as one column per component, plus an "event" column.

        Args:
            channels (Sequence[str], optional): Glob patterns of the channels
                to include, such as "aero.*". Defaults to every channel.

        Returns:
            pd.DataFrame: One row per recorded point.
        """
        return telemetry.table(self.points, self.events, channels)

    def summary(self) -> dict[str, Any]:
        """Return headline numbers of the flight, in SI units."""
        return {
            "flight_time_s": float(self.time.m_as("s")[-1]),
            "apogee_m": float(self["state.position"].m_as("m")[:, 0].max()),
            "max_speed_m_s": float(self["derived.speed"].m_as("m/s").max()),
            "max_mach": float(self["aero.mach"].magnitude.max()),
            "max_dynamic_pressure_pa": float(
                self["aero.dynamic_pressure"].m_as("Pa").max()
            ),
        }

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

    def write(self, figures: Mapping[str, Figure] | None = None) -> Path | None:
        """Write the flight as the simulation's output table says.

        The directory gets the telemetry, "run.toml" with the summary, event
        times and source revision, "resolved_input.toml" when the simulation
        came from a file, and a PNG of each figure.

        Args:
            figures (Mapping[str, Figure] | None, optional): Figures to save.
                Defaults to drawing, saving and closing ``output.plots``.

        Returns:
            Path | None: The directory written, or None if the output table
                names none.
        """
        settings = self.simulation.output
        directory = settings.resolve_directory(self.simulation.name)
        if directory is None:
            return None
        directory.mkdir(parents=True, exist_ok=True)

        rows = (
            self.resampled(settings.sample_interval)
            if settings.sample_interval is not None
            else self
        )
        output.write_tables(
            rows.to_dataframe(settings.channels), directory, settings.formats
        )
        times = self.time.m_as("s")
        run: dict[str, Any] = {
            "name": self.simulation.name,
            "written": datetime.now().astimezone().replace(microsecond=0),
            **output.git_revision(),
            "summary": self.summary(),
            "event_times_s": {
                name: float(times[row]) for name, row in self.events.items()
            },
        }
        if self.simulation.source_file is not None:
            run["source"] = self.simulation.source_file.as_posix()
        write_toml(directory / "run.toml", run)
        resolved = self.simulation.resolved_input
        if resolved is not None:
            # Rerunning the copy writes back to this directory
            written_output = {
                **resolved.get("output", {}),
                "directory": directory.as_posix(),
            }
            write_toml(
                directory / "resolved_input.toml",
                {**resolved, "output": written_output},
                {
                    key_path: "pulled from default"
                    for key_path in self.simulation.input_defaults
                    if key_path not in ("output", "output.directory")
                },
            )

        drawn = self.figures() if figures is None else figures
        for name, figure in drawn.items():
            figure.savefig(directory / f"{name}.png", dpi=150)
        if figures is None:
            for figure in drawn.values():
                plt.close(figure)
        return directory
