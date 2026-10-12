"""Where and how a flight's telemetry, run summary and plots are written."""

import inspect
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import pandas as pd  # type: ignore[import-untyped]
import tomli_w
from matplotlib.figure import Figure

from flight_sim import plots
from flight_sim.plots import ChannelSource
from flight_sim.telemetry import channel_names
from flight_sim.units import Scalar, UnitChecked

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
    sample_interval: Annotated[Scalar, "s"] | None = None

    # Names of functions in flight_sim.plots, drawn and saved as PNGs
    plots: Sequence[str] = field(default_factory=lambda: ("trajectory", "step_length"))

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
        unknown_plots = set(self.plots) - set(plot_names())
        if unknown_plots:
            raise ValueError(
                f"Unknown plots {sorted(unknown_plots)}; use {plot_names()}"
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
    if name not in plot_names():
        raise KeyError(f"No plot {name!r}; use {plot_names()}")
    function: Callable[[ChannelSource, str], Figure] = getattr(plots, name)
    return function


def file_name(name: str) -> str:
    """Return a sim name as a lower-case file name, such as "sol_invictus_nominal"."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "flight"


def write_tables(frame: pd.DataFrame, directory: Path, formats: Sequence[str]) -> None:
    """Write telemetry as "telemetry.<format>" in each format.

    Raises:
        ImportError: If parquet is asked for without pyarrow or fastparquet
            installed.
    """
    if "csv" in formats:
        frame.to_csv(directory / "telemetry.csv", index=False)
    if "parquet" in formats:
        frame.to_parquet(directory / "telemetry.parquet", index=False)


def write_toml(path: Path, table: Mapping[str, Any]) -> None:
    """Write a table as a TOML file."""
    path.write_text(tomli_w.dumps(table), encoding="utf-8")


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
