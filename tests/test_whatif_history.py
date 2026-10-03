# ruff: noqa: E501
# pylint: disable=line-too-long
"""Tests for OpenRocket's numbers on the predictions page: from the History site, or from the design file."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ork_fixture import write_aero_csv, write_eng, write_ork

from flight_sim.whatif.build import build_data
from flight_sim.whatif.history import from_history, load_history

DESIGN = "aero_modeling/IREC/OR/t.ork"
SIM_WITH_COMMA = "calm, windy [85%]"


def _flight(scale: float = 1.0) -> dict[str, Any]:
    """A climb to apogee at 40 s and a slow fall, as the History site stores it (SI, some samples undefined)."""
    t = [round(i * 0.05, 3) for i in range(1301)]
    alt = [
        scale * (80 * x - x * x) if x <= 40 else scale * (1600 - 20 * (x - 40))
        for x in t
    ]
    speed = [scale * (80 - 2 * x) if x <= 40 else 20.0 for x in t]
    cols = {
        "time": t,
        "altitude": alt,
        "velocity_total": speed,
        "acceleration_total": [-9.80665 * 2.0] * len(t),
        "mach_number": [v / 340 for v in speed],
        "stability": [None if x < 0.5 else 2.0 + 0.01 * x for x in t],
        "drag_coeff": [0.45] * len(t),
        "dynamic_pressure": [0.5 * 1.2 * v * v for v in speed],
    }
    events = {
        "LAUNCHROD": [0.5],
        "BURNOUT": [10.0],
        "APOGEE": [40.0],
        "GROUND_HIT": [65.0],
    }
    return {"short": "abc1234", "date": "2027-01-03", "cols": cols, "events": events}


def _row(apogee: float) -> dict[str, Any]:
    m = {
        "apogee": apogee,
        "max_mach": 0.9,
        "max_dynamic_pressure_kpa": 12.5,
        "stability_off_rod_cal": 1.9,
        "min_stability_cal": 1.7,
        "max_stability_cal": 2.4,
    }
    return {"short": "abc1234", "date": "2027-01-03", "ok": True, "m": m}


def _site(root: Path, sims: dict[str, dict[str, Any]]) -> Path:
    """A History site folder holding the given simulations of one design."""
    (root / "flights").mkdir(parents=True)
    designs: dict[str, Any] = {DESIGN: {}}
    flights = {}
    for i, (sim, flight) in enumerate(sims.items()):
        designs[DESIGN][sim] = [
            {"short": "0000001", "date": "2027-01-01", "ok": False, "m": {}},
            _row(flight["cols"]["altitude"][800]),
        ]
        name = f"flights/{i}.js"
        flights[f"{DESIGN}|{sim}"] = name
        payload = json.dumps({"versions": [flight]}, separators=(",", ":"))
        (root / name).write_text(
            f"window.__flight({json.dumps(f'{DESIGN}|{sim}')}, {payload});\n",
            encoding="utf-8",
        )
    (root / "data.json").write_text(
        json.dumps({"designs": designs, "flights": flights}), encoding="utf-8"
    )
    return root


def test_the_history_site_is_read_by_design_file_name(tmp_path: Path) -> None:
    """Every simulation of the design is found, even one whose name has a comma in it."""
    site = _site(tmp_path / "site", {"calm": _flight(), SIM_WITH_COMMA: _flight(1.1)})
    found = load_history(site, "t.ork")
    assert set(found) == {"calm", SIM_WITH_COMMA}
    assert (
        found["calm"]["short"] == "abc1234"
        and found["calm"]["flight"]["cols"]["time"][0] == 0.0
    )
    assert not load_history(site, "other.ork")
    assert not load_history(tmp_path / "nothing", "t.ork")


def test_a_broken_flight_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One unreadable simulation leaves the others alone."""
    site = _site(tmp_path / "site", {"calm": _flight(), "windy": _flight()})
    (site / "flights" / "1.js").write_text("not a flight file", encoding="utf-8")
    assert set(load_history(site, "t.ork")) == {"calm"}


def test_history_numbers_and_lines(tmp_path: Path) -> None:
    """The climb is cut at apogee and thinned; the History table's numbers win."""
    site = _site(tmp_path / "site", {"calm": _flight()})
    entry = load_history(site, "t.ork")["calm"]
    result = from_history(entry)
    assert (
        result is not None
        and result["source"] == "history"
        and "abc1234" in result["label"]
    )
    m = result["m"]
    assert m["apogeeT"] == 40.0 and m["machMax"] == pytest.approx(
        0.9
    )  # the History table's value, not the series'
    assert m["qMax"] == pytest.approx(12500.0)  # kPa on the History page, Pa here
    assert (
        m["marginRail"] == pytest.approx(1.9)
        and m["marginLo"] == pytest.approx(1.7)
        and m["marginHi"] == pytest.approx(2.4)
    )
    assert m["vmax"] == pytest.approx(80.0) and m["accMax"] == pytest.approx(
        -2.0, abs=0.01
    )
    assert m["railV"] == pytest.approx(80 - 2 * 0.5, abs=0.1)
    t = np.array(result["t"])
    assert (
        t[0] == 0.0 and t[-1] == pytest.approx(40.0) and 100 < len(t) < 300
    )  # about a quarter second apart
    assert np.diff(t).min() >= 0.2 and max(result["alt"]) == pytest.approx(
        1600.0, rel=0.01
    )
    assert (
        len(result["alt"])
        == len(result["v"])
        == len(result["mach"])
        == len(result["acc"])
        == len(result["margin"])
        == len(result["cd"])
        == len(t)
    )
    assert (
        result["margin"][0] is None and result["margin"][-1] is not None
    )  # undefined samples stay undefined


def test_a_flight_without_the_basic_columns_is_not_used() -> None:
    """Nothing is drawn from a flight that lacks altitude, speed, Mach or stability."""
    flight = _flight()
    del flight["cols"]["stability"]
    assert (
        from_history({"flight": flight, "metrics": {}, "short": "x", "date": "d"})
        is None
    )


def test_the_build_uses_history_where_it_has_the_simulation(tmp_path: Path) -> None:
    """History numbers for 'calm'; a design with no History flights falls back to the file's own saved run."""
    ork = write_ork(tmp_path / "t.ork")
    aero = write_aero_csv(
        tmp_path / "aero.csv"
    )  # not "t.csv": loading t.eng writes that next to it
    motor = write_eng(tmp_path / "t.eng")
    site = _site(tmp_path / "site", {"calm": _flight()})
    data = build_data(
        ork,
        aero,
        motor,
        "Test rocket",
        "calm",
        history_site=site,
        history_motor="IGNIS.rse",
    )
    assert (
        data["openrocket"]["calm"]["source"] == "history"
        and data["openrocketMotor"] == "IGNIS.rse"
    )
    json.dumps(data)
    plain = build_data(
        ork, aero, motor, "Test rocket", "calm", history_site=tmp_path / "no_such_site"
    )
    assert (
        plain["openrocket"]["calm"]["source"] == "file"
        and plain["openrocketMotor"] == ""
    )
