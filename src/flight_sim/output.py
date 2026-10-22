"""Where and how a run's input, telemetry, run summary and plots are written."""

import inspect
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from flight_sim import plots
from flight_sim.config import InputSource, write_toml
from flight_sim.plots import ChannelSource
from flight_sim.telemetry import channel_names
from flight_sim.units import Scalar, UnitChecked, UnitSystem

if TYPE_CHECKING:
    from flight_sim.simulation import Flight, Simulation

FORMATS = ("csv", "parquet")

# Input a run directory runs from, and the folder of its data file copies
RESOLVED_INPUT = "resolved_input.toml"
INPUT_FILES = "inputs"

# Folder of a run directory holding its figures
PLOT_FILES = "plots"

# Format of a bare "{datetime}" in an output directory, such as 2026-10-22_140327
DATETIME_FORMAT = "%Y-%m-%d_%H%M%S"


@dataclass
class OutputConfiguration(UnitChecked):
    """What a run writes, and where. With no directory it writes nothing."""

    # "{name}" in it is replaced by the sim name, made safe for a file name,
    # and "{datetime}" or "{datetime:%Y%m%d}" by the time the run starts
    directory: Path | None = None
    formats: Sequence[str] = ("csv",)

    # Units of the telemetry, plots and summary: "metric" for the SI units
    # quantities are declared in, or "imperial"
    units: UnitSystem = "metric"

    # Glob patterns of the channels in the telemetry files
    channels: Sequence[str] = ("*",)

    # Telemetry spacing; None keeps one row per integrator step. Events keep
    # their exact rows either way.
    sample_interval: Annotated[Scalar | None, "s"] = None

    # Names of functions in flight_sim.plots, drawn and saved as PNGs in
    # the directory's plots folder
    plots: Sequence[str] = ("trajectory", "step_length")

    def __post_init__(self) -> None:
        """Check the directory, formats, unit system, channels and plot names.

        Raises:
            ValueError: If any of them names something that does not exist, or
                the sample interval is not positive.
        """
        super().__post_init__()
        if self.directory is not None:
            try:
                _fill(str(self.directory), "", datetime.now())
            except (AttributeError, IndexError, KeyError, ValueError) as error:
                raise ValueError(
                    f"Output directory {self.directory} has a bad placeholder"
                    f" ({error}); use {{name}}, {{datetime}} or {{datetime:%Y%m%d}}"
                ) from error
        unknown_formats = set(self.formats) - set(FORMATS)
        if unknown_formats:
            raise ValueError(
                f"Unknown formats {sorted(unknown_formats)}; use {FORMATS}"
            )
        if self.units not in get_args(UnitSystem):
            raise ValueError(
                f"Unknown units {self.units!r}; use {get_args(UnitSystem)}"
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

    def resolve_directory(self, name: str, time: datetime) -> Path | None:
        """Return the directory for a sim of this name started at a time.

        Returns:
            Path | None: The directory, or None to write nothing.
        """
        if self.directory is None:
            return None
        return Path(_fill(str(self.directory), name, time)).resolve()


class _StartTime:
    """Time a run starts, as "{datetime}" or "{datetime:<strftime format>}" puts it."""

    def __init__(self, time: datetime) -> None:
        self.time = time

    def __format__(self, spec: str) -> str:
        return self.time.strftime(spec or DATETIME_FORMAT)


def _fill(template: str, name: str, time: datetime) -> str:
    """Replace the "{name}" and "{datetime}" placeholders in a directory."""
    return template.format_map({"name": file_name(name), "datetime": _StartTime(time)})


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


def prepare(simulation: "Simulation", *, overwrite: bool = False) -> Path:
    """Create a sim's run directory and write the input it runs from.

    The directory gets RESOLVED_INPUT, which names a copy of each data file
    the sim reads under INPUT_FILES and the directory itself as the output
    directory, all relative to itself, so the directory reruns anywhere.

    Args:
        simulation (Simulation): Sim read from a file, naming an output
            directory.
        overwrite (bool, optional): Write into a directory that already
            exists. Defaults to refusing.

    Returns:
        Path: The run directory.

    Raises:
        ValueError: If the sim did not come from a file or names no directory.
        FileExistsError: If the directory exists and ``overwrite`` is not set.
    """
    if simulation.source is None:
        raise ValueError("Only a sim read from a file can be prepared")
    directory = simulation.output.resolve_directory(simulation.name, datetime.now())
    if directory is None:
        raise ValueError("The sim names no output directory")
    if directory.exists() and not overwrite:
        raise FileExistsError(f"{directory} already exists")
    directory.mkdir(parents=True, exist_ok=True)
    _write_input(simulation.source, directory)
    return directory


def write(flight: "Flight", figures: Mapping[str, Figure] | None = None) -> Path | None:
    """Write a flight as its simulation's output table says.

    The directory gets the telemetry, "run.toml" with the summary,
    diagnostics, event times and source revision, and a PNG of each figure
    under PLOT_FILES, replacing any figures already there.
    A simulation read from a file also gets its input written there, as
    ``prepare`` writes it. The telemetry, figures and summary are in the
    flight's ``units``.

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
    directory = settings.resolve_directory(simulation.name, datetime.now())
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
        _write_input(simulation.source, directory)

    drawn = flight.figures() if figures is None else figures
    plot_directory = directory / PLOT_FILES
    shutil.rmtree(plot_directory, ignore_errors=True)
    if drawn:
        plot_directory.mkdir()
    for name, figure in drawn.items():
        figure.savefig(plot_directory / f"{name}.png", dpi=150)
    if figures is None:
        for figure in drawn.values():
            plt.close(figure)
    return directory


def _run_table(flight: "Flight") -> dict[str, Any]:
    """Return the summary, diagnostics, event times and source of a run."""
    times = flight.time.m_as("s")
    run: dict[str, Any] = {
        "name": flight.simulation.name,
        "units": flight.units,
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


def _write_input(source: InputSource, directory: Path) -> None:
    """Copy a sim's data files into a run directory and write the input naming them.

    Data files of the same name from different places get numbered copies.
    Input read from the directory's own RESOLVED_INPUT is already there.
    """
    if source.path == directory / RESOLVED_INPUT:
        return
    inputs = directory / INPUT_FILES
    shutil.rmtree(inputs, ignore_errors=True)
    copies: dict[Path, str] = {}
    relative_paths: dict[str, str] = {}
    for where, path in source.paths.items():
        if where == "output.directory":
            continue
        if path not in copies:
            name, number = path.name, 1
            while name in copies.values():
                number += 1
                name = f"{path.stem}_{number}{path.suffix}"
            copies[path] = name
            inputs.mkdir(exist_ok=True)
            shutil.copy2(path, inputs / copies[path])
        relative_paths[where] = f"{INPUT_FILES}/{copies[path]}"

    table = source.table_with(relative_paths)
    table["output"] = {**table.get("output", {}), "directory": "."}
    comments = {
        default: "pulled from default"
        for default in source.defaults
        if default not in ("output", "output.directory")
    }
    write_toml(directory / RESOLVED_INPUT, table, comments)


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
