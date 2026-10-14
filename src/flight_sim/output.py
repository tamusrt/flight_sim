"""Where and how a flight's telemetry, run summary and plots are written."""

import inspect
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from flight_sim import plots
from flight_sim.config import InputSource, write_toml
from flight_sim.plots import ChannelSource
from flight_sim.telemetry import channel_names
from flight_sim.units import Scalar, UnitChecked

if TYPE_CHECKING:
    from flight_sim.simulation import Flight

FORMATS = ("csv", "parquet")


@dataclass
class OutputConfiguration(UnitChecked):
    """What a run writes, and where. With no directory it writes nothing."""

    # "{name}" in it is replaced by the sim name, made safe for a file name
    directory: Path | None = None
    formats: Sequence[str] = ("csv",)

    # Glob patterns of the channels in the telemetry files
    channels: Sequence[str] = ("*",)

    # Telemetry spacing; None keeps one row per integrator step. Events keep
    # their exact rows either way.
    sample_interval: Annotated[Scalar | None, "s"] = None

    # Names of functions in flight_sim.plots, drawn and saved as PNGs
    plots: Sequence[str] = ("trajectory", "step_length")

    def __post_init__(self) -> None:
        """Check the formats, channel patterns and plot names.

        Raises:
            ValueError: If any of them names something that does not exist, or
                the sample interval is not positive.
        """
        super().__post_init__()
        unknown_formats = set(self.formats) - set(FORMATS)
        if unknown_formats:
            raise ValueError(
                f"Unknown formats {sorted(unknown_formats)}; use {FORMATS}"
            )
        known_plots = plot_names()
        unknown_plots = set(self.plots) - set(known_plots)
        if unknown_plots:
            raise ValueError(
                f"Unknown plots {sorted(unknown_plots)}; use {known_plots}"
            )
        try:
            channel_names(self.channels)
        except KeyError as error:
            raise ValueError(error.args[0]) from error
        if self.sample_interval is not None and not self.sample_interval.m_as("s") > 0:
            raise ValueError("Sample interval must be positive")

    def resolve_directory(self, name: str) -> Path | None:
        """Return the directory for a sim of this name, or None to write nothing."""
        if self.directory is None:
            return None
        return Path(str(self.directory).replace("{name}", file_name(name))).resolve()


def plot_names() -> list[str]:
    """Return the name of every plot in ``flight_sim.plots``, in file order."""
    return [
        name
        for name, value in vars(plots).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and value.__module__ == plots.__name__
    ]


def plot_function(name: str) -> Callable[[ChannelSource, str], Figure]:
    """Return the plot of ``flight_sim.plots`` with a given name.

    Raises:
        KeyError: If no plot has that name.
    """
    known_plots = plot_names()
    if name not in known_plots:
        raise KeyError(f"No plot {name!r}; use {known_plots}")
    function: Callable[[ChannelSource, str], Figure] = getattr(plots, name)
    return function


def file_name(name: str) -> str:
    """Return a sim name as a lower-case file name, such as "sol_invictus_nominal"."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "flight"


def write(flight: "Flight", figures: Mapping[str, Figure] | None = None) -> Path | None:
    """Write a flight as its simulation's output table says.

    The directory gets the telemetry, "run.toml" with the summary,
    diagnostics, event times and source revision, "resolved_input.toml"
    when the simulation came from a file, and a PNG of each figure.

    Args:
        flight (Flight): The flight to write.
        figures (Mapping[str, Figure] | None, optional): Figures to save.
            Defaults to drawing, saving and closing ``output.plots``.

    Returns:
        Path | None: The directory written, or None if the output table
            names none.
    """
    simulation = flight.simulation
    settings = simulation.output
    directory = settings.resolve_directory(simulation.name)
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)

    frame = flight.to_dataframe(settings.channels, settings.sample_interval)
    if "csv" in settings.formats:
        frame.to_csv(directory / "telemetry.csv", index=False)
    if "parquet" in settings.formats:
        frame.to_parquet(directory / "telemetry.parquet", index=False)
    write_toml(directory / "run.toml", _run_table(flight))
    if simulation.source is not None:
        _write_resolved_input(simulation.source, directory)

    drawn = flight.figures() if figures is None else figures
    for name, figure in drawn.items():
        figure.savefig(directory / f"{name}.png", dpi=150)
    if figures is None:
        for figure in drawn.values():
            plt.close(figure)
    return directory


def _run_table(flight: "Flight") -> dict[str, Any]:
    """Return the summary, diagnostics, event times and source of a run."""
    times = flight.time.m_as("s")
    run: dict[str, Any] = {
        "name": flight.simulation.name,
        "written": datetime.now().astimezone().replace(microsecond=0),
        **git_revision(),
        "summary": flight.summary(),
        "diagnostics": flight.diagnostics(),
        "event_times_s": {
            name: float(times[row]) for name, row in flight.events.items()
        },
    }
    if flight.simulation.source is not None:
        run["source"] = flight.simulation.source.path.as_posix()
    return run


def _write_resolved_input(source: InputSource, directory: Path) -> None:
    """Write the resolved input so that rerunning it writes back to ``directory``."""
    table = {
        **source.table,
        "output": {**source.table.get("output", {}), "directory": directory.as_posix()},
    }
    comments = {
        default: "pulled from default"
        for default in source.defaults
        if default not in ("output", "output.directory")
    }
    write_toml(directory / "resolved_input.toml", table, comments)


def git_revision() -> dict[str, Any]:
    """Return the commit of the flight_sim source and whether it has changes.

    Returns:
        dict[str, Any]: "commit" and "dirty", or nothing outside a git checkout.
    """
    source = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    if commit.returncode != 0:
        return {}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
