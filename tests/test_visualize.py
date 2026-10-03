"""Tests for the telemetry log and the viewer page."""

import json
from pathlib import Path

import pytest

from flight_sim.__main__ import (
    INVICTUS,
    LAUNCH_RAIL,
    RECOVERY,
    get_default_state,
    get_profile_properties,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.visualize import RecoveryFrame, TelemetryLog, write_viewer


def test_viewer_page_embeds_the_logged_flight(tmp_path: Path) -> None:
    """Each sample is logged and the page carries it as JSON."""
    log = TelemetryLog(get_profile_properties(INVICTUS), IntegrationConfiguration())
    state = get_default_state()
    log.record(0.0, state)
    log.record(0.5, state)
    log.add_event("deploy", "Main reefed", 0.25, inflation=0.1)
    log.describe_rail(LAUNCH_RAIL)
    log.describe_recovery(RECOVERY)

    page = write_viewer(log, tmp_path / "flight.html", open_browser=False)

    text = page.read_text(encoding="utf-8")
    assert "<!--TELEMETRY-->" not in text
    start = text.index("window.TELEMETRY=") + len("window.TELEMETRY=")
    data = json.loads(text[start : text.index(";window.TELEMETRY_NAME")])
    assert data["t"] == [0.0, 0.5]
    assert data["quat"][0] == [1.0, 0.0, 0.0, 0.0]
    assert len(data["pos"]) == len(data["vel"]) == len(data["thrust"]) == 2
    assert data["mach"] == [0.0, 0.0]
    assert data["q"] == [0.0, 0.0]
    # Burning at both samples, so propellant flows out of the nozzle
    assert all(m > 0.0 for m in data["mdot"])
    # The CG at ignition is the profile's, then moves forward as the grain burns
    assert data["cg"][0] == pytest.approx(3.298)
    assert 3.109 < data["cg"][1] < 3.298
    assert len(data["cp"]) == 2
    assert data["events"] == [
        {"kind": "deploy", "name": "Main reefed", "t": 0.25, "inflation": 0.1}
    ]
    assert data["rail"]["length"] == 10.0
    assert data["recovery"]["cord_to_body"] == 364 * 0.0254


def test_recovery_frame_is_logged_beside_the_state(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A sample without a frame has no line; one with a frame carries it."""
    log = TelemetryLog(baseline_rocket_properties, IntegrationConfiguration())
    state = get_default_state()
    log.record(0.0, state)
    log.record(
        1.0,
        state,
        RecoveryFrame(
            nose_dir=[0.0, 0.0, 1.0],
            line=[-1.0, 0.0, 0.0],
            swing_deg=12.0,
            drag_fraction=0.5,
        ),
    )
    assert log.rows["line"] == [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert log.rows["nose_dir"][1] == [0.0, 0.0, 1.0]
    assert log.rows["swing"] == [0.0, 12.0]
    assert log.rows["drag"] == [1.0, 0.5]


def test_viewer_page_names_the_openrocket_run_to_load(tmp_path: Path) -> None:
    """With an OpenRocket simulation named, the page offers to load its path."""
    log = TelemetryLog(get_profile_properties(INVICTUS), IntegrationConfiguration())
    log.record(0.0, get_default_state())
    source = {"ork": "2027_OR.ork", "sim": "average"}

    with_or = write_viewer(
        log, tmp_path / "a.html", open_browser=False, openrocket=source
    )
    without = write_viewer(log, tmp_path / "b.html", open_browser=False)

    text = with_or.read_text(encoding="utf-8")
    start = text.index("window.OPENROCKET_SOURCE=") + len("window.OPENROCKET_SOURCE=")
    assert json.loads(text[start : text.index(";</script>", start)]) == source
    assert "window.OPENROCKET_SOURCE=" not in without.read_text(encoding="utf-8")
    assert 'id="orBtn"' in text
