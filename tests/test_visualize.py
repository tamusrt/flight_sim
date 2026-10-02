"""Tests for the telemetry log and the viewer page."""

import json
from pathlib import Path

from flight_sim.__main__ import get_default_state
from flight_sim.integration import IntegrationConfiguration
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.visualize import TelemetryLog, write_viewer


def test_viewer_page_embeds_the_logged_flight(
    baseline_rocket_properties: RocketProperties, tmp_path: Path
) -> None:
    """Each sample is logged and the page carries it as JSON."""
    log = TelemetryLog(baseline_rocket_properties, IntegrationConfiguration())
    state = get_default_state()
    log.record(0.0, state)
    log.record(0.5, state)
    log.events = [("drogue release", 0.25)]

    page = write_viewer(log, tmp_path / "flight.html", open_browser=False)

    text = page.read_text(encoding="utf-8")
    assert "<!--TELEMETRY-->" not in text
    start = text.index("window.TELEMETRY=") + len("window.TELEMETRY=")
    data = json.loads(text[start : text.index(";window.TELEMETRY_NAME")])
    assert data["t"] == [0.0, 0.5]
    assert data["quat"][0] == [1.0, 0.0, 0.0, 0.0]
    assert len(data["pos"]) == len(data["vel"]) == len(data["thrust"]) == 2
    assert data["mach"] == [0.0, 0.0]
    assert data["events"] == [["drogue release", 0.25]]
