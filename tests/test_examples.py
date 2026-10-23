"""Example sim tests."""

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from flight_sim import output
from flight_sim.mission import Mission
from flight_sim.simulation import Simulation

_EXAMPLES = sorted(Path("examples").glob("*/sim.toml"))
_MISSIONS = sorted(Path("examples").glob("*/mission.toml"))


def test_there_are_examples() -> None:
    """The examples folder holds at least one sim."""
    assert _EXAMPLES


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda path: path.parent.name)
def test_example_flies_to_its_stop_event(path: Path) -> None:
    """Each example builds from its own folder and flies to its stop event."""
    simulation = Simulation.from_file(path)

    flight = simulation.run()

    assert list(flight.events)[-1] == simulation.stop_event
    assert simulation.source is not None
    folder = path.parent.resolve()
    files = [Path(text) for text in _strings(simulation.source.table)]
    read = [file for file in files if file.is_file()]
    assert read and all(file.is_relative_to(folder) for file in read)


@pytest.mark.parametrize("path", _MISSIONS, ids=lambda path: path.parent.name)
def test_example_mission_flies_every_segment(path: Path, tmp_path: Path) -> None:
    """Each example mission flies its segments' events from its run directory."""
    original = Mission.from_file(path)
    original.output = replace(original.output, directory=tmp_path / "run")
    resolved = output.prepare(original) / output.RESOLVED_INPUT
    mission = Mission.from_file(resolved)

    flight = mission.run()

    stops = [segment.stop_event for segment in mission.segments]
    assert [name for name in flight.events if name in stops] == stops
    assert flight.simulation.name == mission.name
    assert stops == [segment.stop_event for segment in original.segments]
    # Segment output is ignored, so none is written to name machine paths
    written = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert all("output" not in segment for segment in written["segments"])


def _strings(node: object) -> list[str]:
    """Return every string in a table."""
    if isinstance(node, dict):
        return [text for item in node.values() for text in _strings(item)]
    if isinstance(node, list):
        return [text for item in node for text in _strings(item)]
    return [node] if isinstance(node, str) else []
