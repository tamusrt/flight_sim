"""Output table, resampling, file writing and plot tests."""

import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from flight_sim import output
from flight_sim.config import ConfigError, write_toml
from flight_sim.output import OutputConfiguration, file_name
from flight_sim.simulation import Flight, Simulation
from flight_sim.units import scalar
from flight_sim.vehicle.engine import SolidEngine

_VEHICLE = Path("configs/vehicles/sol_invictus.toml").resolve().as_posix()


def _sim_file(directory: Path, output_table: str = "") -> Path:
    """Write a short, weightless sim file whose events are fixed times."""
    path = directory / "short.toml"
    path.write_text(
        f"""
        name = "Short Hop"
        stop_event = "end"
        properties = {{ include = "{_VEHICLE}" }}
        initial_state = {{ velocity = {{ value = [50, 1, 0], units = "m/s" }} }}

        [config.truth.gravity]
        type = "flight_sim.environment.gravity.ConstantGravity"
        magnitude_m_s2 = 0.0

        [[events]]
        type = "flight_sim.events.at_time"
        name = "half"
        time = "25 ms"

        [[events]]
        type = "flight_sim.events.at_time"
        name = "end"
        time = "50 ms"

        {output_table}
        """,
        encoding="utf-8",
    )
    return path


def _same_engine(a: Simulation, b: Simulation) -> bool:
    """Return whether two sims have the same thrust curve and grain."""
    engine_a, engine_b = a.properties.engine, b.properties.engine
    assert isinstance(engine_a, SolidEngine) and isinstance(engine_b, SolidEngine)
    return (
        np.array_equal(engine_a.thrusts, engine_b.thrusts)
        and np.array_equal(engine_a.times, engine_b.times)
        and engine_a.grain.mass == engine_b.grain.mass
        and np.array_equal(
            engine_a.grain.cg_location.m_as("m"), engine_b.grain.cg_location.m_as("m")
        )
    )


@pytest.fixture(name="flight")
def _flight(tmp_path: Path) -> Flight:
    """Provides a short flight loaded from a file, writing under tmp_path/out."""
    table = """
        [output]
        directory = "out/{name}"
        channels = ["time", "aero.*"]
        sample_interval = "10 ms"
        plots = ["trajectory"]
    """
    return Simulation.from_file(_sim_file(tmp_path, table)).run()


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"formats": ["xlsx"]}, "Unknown formats"),
        ({"plots": ["polar"]}, "Unknown plots"),
        ({"channels": ["drag"]}, "No channel matches"),
        ({"sample_interval": scalar(0.0, "s")}, "must be positive"),
    ],
)
def test_output_table_is_checked(table: dict[str, Any], message: str) -> None:
    """Unknown formats, plots and channels, and a zero interval, are rejected."""
    with pytest.raises(ValueError, match=message):
        OutputConfiguration(**table)


def test_bad_output_table_fails_at_load(tmp_path: Path) -> None:
    """A bad output table is reported when the sim file loads."""
    path = _sim_file(tmp_path, '[output]\nplots = ["polar"]')

    with pytest.raises(ConfigError, match=r"output in .*short\.toml: Unknown plots"):
        Simulation.from_file(path)


def test_directory_names_the_sim() -> None:
    """The directory's "{name}" becomes the sim name as a file name."""
    settings = OutputConfiguration(directory=Path("out/{name}"))

    assert (
        file_name("Sol Invictus - 20 mph west wind") == "sol_invictus_20_mph_west_wind"
    )
    assert file_name("!!!") == "flight"
    assert settings.resolve_directory("Short Hop") == Path("out/short_hop").resolve()
    assert OutputConfiguration().resolve_directory("Short Hop") is None


def test_sampled_table_keeps_events_on_their_rows(flight: Flight) -> None:
    """Rows fall every interval, plus exactly at each event and the end."""
    frame = flight.to_dataframe(
        ["time", "state.orientation.*", "derived.on_rail"], scalar(10.0, "ms")
    )
    times = frame["time [s]"].to_numpy()

    assert times == pytest.approx([0.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05])
    assert frame["event"].tolist()[3:] == ["half", "", "", "end"]
    assert times[3] == flight.at("half").time.m_as("s")
    assert times[-1] == flight.at("end").time.m_as("s")
    orientation = frame[[f"state.orientation.q_{axis} [-]" for axis in "wxyz"]]
    assert np.linalg.norm(orientation.to_numpy(), axis=1) == pytest.approx(1.0)
    assert set(frame["derived.on_rail [-]"]) <= {0.0, 1.0}


def test_write_lays_out_the_run(flight: Flight, tmp_path: Path) -> None:
    """The directory gets telemetry, the run summary, the input and the plots."""
    directory = output.write(flight)

    assert directory == (tmp_path / "out" / "short_hop").resolve()
    frame = pd.read_csv(directory / "telemetry.csv")
    assert frame.columns[0] == "time [s]"
    assert all(c.startswith(("time", "aero.", "event")) for c in frame.columns)
    assert len(frame) == 7
    assert frame["event"].dropna().tolist() == ["half", "end"]
    assert (directory / "trajectory.png").is_file()
    assert plt.get_fignums() == []

    run = tomllib.loads((directory / "run.toml").read_text(encoding="utf-8"))
    assert run["name"] == "Short Hop"
    assert run["event_times_s"]["end"] == pytest.approx(0.05)
    assert run["summary"]["max_mach"] > 0.1
    assert run["source"].endswith("short.toml")
    diagnostics = run["diagnostics"]
    assert diagnostics["steps"] == len(flight) - 1
    assert diagnostics["rkf45_evaluations"] > 0
    assert diagnostics["rkf45_evaluations"] % 6 == 0
    assert diagnostics["run_time_s"] > 0.0
    assert diagnostics["load_time_s"] > 0.0


def test_resolved_input_reruns_the_same_sim(flight: Flight) -> None:
    """The copied input rebuilds the sim, writing back to the same place."""
    directory = output.write(flight)
    assert directory is not None

    rerun = Simulation.from_file(directory / "resolved_input.toml")

    assert rerun.name == flight.simulation.name
    assert rerun.output.resolve_directory(rerun.name) == directory
    assert _same_engine(rerun, flight.simulation)
    assert rerun.event_names() == flight.simulation.event_names()


def test_nominal_resolved_input_is_absolute_and_rebuilds(tmp_path: Path) -> None:
    """The nominal sim's resolved input names its data files absolutely."""
    simulation = Simulation.from_file("configs/sims/sol_invictus_nominal.toml")
    assert simulation.source is not None
    resolved = simulation.source.table
    aero_file = Path(resolved["properties"]["aero_table"]["filepath"])

    write_toml(tmp_path / "copy.toml", resolved)
    rebuilt = Simulation.from_file(tmp_path / "copy.toml")

    assert aero_file.is_absolute() and aero_file.is_file()
    assert "include" not in resolved["config"]["truth"]
    assert rebuilt.config.truth.launch_rail == simulation.config.truth.launch_rail
    assert _same_engine(rebuilt, simulation)


def test_write_without_directory_writes_nothing(tmp_path: Path) -> None:
    """A sim whose output table names no directory writes nothing."""
    flight = Simulation.from_file(_sim_file(tmp_path)).run()

    assert output.write(flight) is None


def test_write_saves_given_figures_and_leaves_them_open(flight: Flight) -> None:
    """Figures passed in are saved and left for the caller to show."""
    figures = flight.figures(["step_length"])

    directory = output.write(flight, figures)

    assert directory is not None
    assert (directory / "step_length.png").is_file()
    assert not (directory / "trajectory.png").exists()
    assert plt.fignum_exists(figures["step_length"].number)
    plt.close("all")


def test_parquet_holds_the_csv_table(flight: Flight) -> None:
    """Both formats hold the same telemetry."""
    simulation = flight.simulation
    simulation.output = replace(simulation.output, formats=["csv", "parquet"])

    directory = output.write(flight, figures={})

    assert directory is not None
    csv = pd.read_csv(directory / "telemetry.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "telemetry.parquet"), csv)


def test_every_plot_draws(flight: Flight) -> None:
    """Every named plot draws a figure of the flight."""
    figures = flight.figures(output.plot_names())

    assert list(figures) == [
        "trajectory",
        "attitude",
        "aero",
        "propulsion",
        "step_length",
    ]
    assert all(figure.get_axes() for figure in figures.values())
    plt.close("all")


def test_git_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkout's commit is found; without git there is nothing to report."""
    revision = output.git_revision()
    assert len(revision["commit"]) == 40
    assert isinstance(revision["dirty"], bool)

    def failed(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 128, "", "not a git repository")

    monkeypatch.setattr(subprocess, "run", failed)
    assert not output.git_revision()

    def missing(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert not output.git_revision()


@pytest.mark.parametrize("name", ["_figure", "Figure", "ChannelSource", "plt"])
def test_only_public_plot_functions_are_plots(name: str) -> None:
    """Private helpers, imports and classes in the plots module are not plots."""
    assert name not in output.plot_names()
    with pytest.raises(KeyError, match="No plot"):
        output.plot_function(name)


def test_write_toml_comments_keys_and_comments_out_none(tmp_path: Path) -> None:
    """Comments land on their key paths; None is written commented out."""
    path = tmp_path / "out.toml"
    write_toml(
        path,
        {
            "name": "hop",
            "rail": {"length": "17 ft", "start": None},
            "events": [{"ref": "flight_sim.events.APOGEE"}],
        },
        {"rail.length": "pulled from default", "events[0]": "first event"},
    )
    text = path.read_text(encoding="utf-8")

    assert 'length = "17 ft"  # pulled from default' in text
    assert "# start = None" in text
    assert "[[events]]  # first event" in text
    assert tomllib.loads(text) == {
        "name": "hop",
        "rail": {"length": "17 ft"},
        "events": [{"ref": "flight_sim.events.APOGEE"}],
    }
