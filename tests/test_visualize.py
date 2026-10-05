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


def _script_body(text: str, name: str) -> str:
    """The JSON a page assigns to ``window.<name>`` (up to its closing ``;``)."""
    start = text.index(f"window.{name}=") + len(f"window.{name}=")
    ends = [text.find(mark, start) for mark in (";</script>", ";window.")]
    return text[start : min(end for end in ends if end >= 0)]


def test_names_cannot_end_the_script_early(tmp_path: Path) -> None:
    """``</script>`` and ``<!--`` in an event, OpenRocket or .ork name are escaped."""
    log = TelemetryLog(get_profile_properties(INVICTUS), IntegrationConfiguration())
    log.record(0.0, get_default_state())
    nasty = "</script><script>alert(1)</script><!-- </SCRIPT>"
    log.add_event("charge", nasty, 0.0)
    log.vehicle = {"segments": [{"name": nasty}]}
    page = write_viewer(
        log,
        tmp_path / "flight.html",
        open_browser=False,
        real={"events": [{"name": nasty}]},
        openrocket={"ork": nasty + ".ork", "sim": nasty},
    )
    text = page.read_text(encoding="utf-8")
    # the data goes in three scripts, and no name starts another
    assert text.count("<script>window.") == 3
    for tag in ("</script><script>alert", "<!-- ", "</SCRIPT>"):
        assert tag not in text
    # and the data is unchanged when read back
    data = json.loads(_script_body(text, "TELEMETRY"))
    assert data["events"][0]["name"] == nasty
    assert data["vehicle"]["segments"][0]["name"] == nasty
    assert (
        json.loads(_script_body(text, "REAL_TELEMETRY"))["events"][0]["name"] == nasty
    )
    source = json.loads(_script_body(text, "OPENROCKET_SOURCE"))
    assert source == {"ork": nasty + ".ork", "sim": nasty}


def test_not_a_number_is_written_as_null(tmp_path: Path) -> None:
    """Any NaN or infinity in the data becomes null, so the JSON stays valid."""
    log = TelemetryLog(get_profile_properties(INVICTUS), IntegrationConfiguration())
    log.record(0.0, get_default_state())
    log.rows["cg"][0] = float("nan")
    log.rows["q"][0] = float("inf")
    log.add_event("charge", "x", 0.0, speed=float("nan"))
    text = log.to_json()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["cg"] == [None] and data["q"] == [None]
    assert data["events"][0]["speed"] is None

    page = write_viewer(
        log,
        tmp_path / "f.html",
        open_browser=False,
        real={"mass": [float("nan")]},
    )
    html = page.read_text(encoding="utf-8")
    assert json.loads(_script_body(html, "REAL_TELEMETRY")) == {"mass": [None]}


def test_a_failed_sample_leaves_no_half_written_row(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Every column is added together, or none is."""
    log = TelemetryLog(baseline_rocket_properties, IntegrationConfiguration())
    log.record(0.0, get_default_state())

    def broken(_mach: float) -> float:
        raise RuntimeError("no centre of pressure")

    log._centre_of_pressure = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        log.record(1.0, get_default_state())
    assert {len(column) for column in log.rows.values()} == {1}


def test_the_page_is_written_into_a_new_folder(tmp_path: Path) -> None:
    log = TelemetryLog(get_profile_properties(INVICTUS), IntegrationConfiguration())
    log.record(0.0, get_default_state())
    page = write_viewer(log, tmp_path / "a" / "b" / "flight.html", open_browser=False)
    assert page.is_file() and page.read_text(encoding="utf-8").startswith(
        '<!doctype html><html lang="en">'
    )


def test_a_canopy_that_is_not_reefed_is_still_drawn() -> None:
    """The scene gets the canopy of any recovery system, not only a reefed one."""
    from flight_sim.descent import Parachute, RecoverySystem

    log = TelemetryLog.__new__(TelemetryLog)
    log.recovery = None
    log.describe_recovery(
        RecoverySystem(
            (
                Parachute("drogue", diameter_m=1.0, drag_coefficient=1.5),
                Parachute("main", diameter_m=2.5, drag_coefficient=1.5),
            )
        )
    )
    assert log.recovery is not None
    assert log.recovery["diameter"] == 2.5
    assert log.recovery["reefed_opening"] == 2.5
