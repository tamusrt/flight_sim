# ruff: noqa: E501
# pylint: disable=line-too-long
"""Tests for the predictions page: its data, its page and its flight model."""

import argparse
import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from ork_fixture import write_aero_csv, write_cdx, write_eng, write_ork, write_rse

from flight_sim.ork import load_ork
from flight_sim.whatif import build
from flight_sim.whatif.build import (
    build_data,
    read_rasaero_geometry,
    read_thrust,
    reduce_aero,
    slug,
    viewer_paths,
    write_page,
    write_viewers,
)

_ENGINE = Path(__file__).parents[1] / "src" / "flight_sim" / "whatif" / "engine.js"


@pytest.fixture(name="data")
def fixture_data(tmp_path: Path) -> dict[str, Any]:
    """Page data built from the synthetic design."""
    return build_data(
        write_ork(tmp_path / "t.ork"),
        write_aero_csv(tmp_path / "t.csv"),
        write_eng(tmp_path / "t.eng"),
        "Test rocket",
        "calm",
    )


def test_aero_is_reduced_to_roll_angle_zero(tmp_path: Path) -> None:
    """Aero is reduced to roll angle zero."""
    aero = reduce_aero(write_aero_csv(tmp_path / "t.csv"), 0.1)
    assert aero["alpha"][0] == 0 and aero["alpha"][-1] == 30
    assert len(aero["mach"]) == len(aero["ca"]) == len(aero["cn"]) == len(aero["xcp"])
    row = aero["alpha"].index(4)
    assert aero["ca"][0][row] == pytest.approx(0.458)  # CA is -Cx
    assert aero["cn"][0][row] == pytest.approx(0.8)  # CN is -Cz
    assert aero["xcp"][0][row] == pytest.approx(1.9)  # CMy * L / Cz
    assert aero["xcp"][0][0] == aero["xcp"][0][1]  # no force at zero alpha


def test_thrust_curve_is_read(tmp_path: Path) -> None:
    """Thrust curve is read."""
    pairs = read_thrust(write_eng(tmp_path / "t.eng"))
    assert pairs[0] == [0.0, 0.0] and pairs[-1] == [4.0, 0.0]


def test_data_holds_the_design_and_its_references(data: dict[str, Any]) -> None:
    """Data holds the design and its references."""
    assert data["defaultSim"] == "calm"
    base = data["base"]
    assert base["fins"]["count"] == 4
    assert base["motor"]["prop"] == pytest.approx(4.0)
    assert data["presets"]["calm"]["railAngleDeg"] == pytest.approx(5.0)
    saved = data["openrocket"]["calm"]
    assert saved["source"] == "file" and saved["m"]["apogee"] == pytest.approx(3000.0)
    assert saved["m"]["railV"] == pytest.approx(25.0) and saved["t"][0] == 0.0
    assert data["sixdof"]["calm"]["apogee"] > 100.0
    json.dumps(data)  # must serialise


def test_data_holds_the_descent_and_the_6dof_landing(data: dict[str, Any]) -> None:
    """The original descent is in the data, and the 6-DOF run ends on the ground."""
    recovery = data["base"]["recovery"]
    assert "Sol Invictus" in recovery["label"] and recovery["bodyCdA"] > 0.0
    assert [s["name"] for s in recovery["stages"]] == ["main reefed", "main reef cut"]
    assert recovery["stages"][1]["after"] == "main reefed"
    landing = data["sixdof"]["calm"]["landing"]
    assert landing["landed"] and landing["t"] > data["sixdof"]["calm"]["apogeeT"]
    assert 0.0 < landing["vVert"] <= landing["v"] and landing["peakG"] > 0.0
    assert [d["name"] for d in landing["deployments"]] == [
        "main reefed",
        "main reef cut",
    ]
    # the 6-DOF record carries on down to the ground
    assert data["sixdof"]["calm"]["alt"][-1] == pytest.approx(0.0, abs=1.0)


def test_jarvis_is_the_6dof_flight_with_every_figure_the_page_shows(
    data: dict[str, Any],
) -> None:
    """The figures and lines of Jarvis all come from the one detailed flight."""
    jarvis = data["sixdof"]["calm"]
    for key in (
        "apogee", "apogeeT", "vmax", "machMax", "qMax", "accMax", "marginLo",
        "marginHi", "burnoutAlt", "aoaMax", "drift", "mass0", "mass1", "cg0", "cp",
    ):  # fmt: skip
        assert isinstance(jarvis[key], float | int), key
    assert jarvis["mass0"] >= jarvis["mass1"] > 0.0
    assert jarvis["marginLo"] <= jarvis["marginHi"]
    assert jarvis["railV"] > 0.0 and jarvis["marginRail"] is not None
    assert 0.0 < jarvis["burnoutAlt"] <= jarvis["apogee"]
    assert jarvis["machMax"] > 0.0 and jarvis["qMax"] > 0.0
    # a row at most every tenth of a second on the way up, and every line has one value a row
    times = jarvis["t"]
    assert times[0] == 0.0 and all(b > a for a, b in pairwise(times))
    slots = [int(t / 0.1 + 1e-6) for t in times[:20]]  # one row per tenth of a second
    assert all(b > a for a, b in pairwise(slots))
    lines = ("alt", "v", "mach", "acc", "marginCal")
    assert all(len(jarvis[key]) == len(times) for key in lines)
    climb = [a for a in jarvis["alt"] if a is not None]
    assert max(climb) == pytest.approx(jarvis["apogee"], rel=0.01, abs=0.5)
    assert 0.0 < jarvis["apogeeT"] <= times[-1]
    # the way down has a height and a speed only
    assert jarvis["mach"][-1] is None and jarvis["marginCal"][-1] is None
    assert jarvis["alt"][-1] == pytest.approx(0.0, abs=1.0)
    json.dumps(jarvis)  # no NaN gets into the page


def test_burnout_is_where_the_thrust_ends(tmp_path: Path) -> None:
    """A motor that burns out on the way up loses its propellant by then."""
    motor = tmp_path / "short.eng"
    motor.write_text(
        "T 100 1000 0 1.0 10.0 X\n0.0 0\n0.01 800\n0.99 800\n1.0 0\n",
        encoding="utf-8",
    )
    jarvis = build_data(
        write_ork(tmp_path / "t.ork"),
        write_aero_csv(tmp_path / "t.csv"),
        motor,
        "Test rocket",
        "calm",
    )["sixdof"]["calm"]
    assert jarvis["mass0"] - jarvis["mass1"] == pytest.approx(1.0, abs=0.05)
    assert 0.0 < jarvis["burnoutAlt"] < jarvis["apogee"]


def test_every_saved_simulation_gets_its_own_vision_page(data: dict[str, Any]) -> None:
    """The page data says where Vision plays each saved condition."""
    assert data["viewers"] == {"calm": "viewer/calm/index.html"}


@pytest.mark.parametrize(
    ("name", "folder"),
    [
        ("average", "average"),
        ("No wind (worst)", "no-wind-worst"),
        ("  Best / 10 m/s ", "best-10-m-s"),
        ("???", "sim"),
    ],
)
def test_a_name_becomes_a_folder_name(name: str, folder: str) -> None:
    """Folder names are lower case letters and digits with single dashes."""
    assert slug(name) == folder


def test_names_that_make_the_same_folder_are_told_apart() -> None:
    """Two conditions never share a Vision page."""
    paths = viewer_paths(["Worst", "worst", "worst!", "Best"])
    assert paths == {
        "Worst": "viewer/worst/index.html",
        "worst": "viewer/worst-2/index.html",
        "worst!": "viewer/worst-3/index.html",
        "Best": "viewer/best/index.html",
    }


def test_every_condition_is_written_and_listed_for_the_history_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One viewer per condition, the default one also at ``viewer/index.html``."""
    flown: list[str] = []

    def fake(_args: argparse.Namespace, target: Path, sim: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"flight of {sim}", encoding="utf-8")
        flown.append(sim)

    monkeypatch.setattr(build, "_write_one_viewer", fake)
    paths = viewer_paths(["average", "Worst case", "best"])
    args = argparse.Namespace(ork=str(tmp_path / "Design 1.ork"))
    first = write_viewers(args, tmp_path, paths, "Worst case")
    assert sorted(flown) == ["Worst case", "average", "best"]
    assert first == tmp_path / "viewer" / "index.html"
    assert first.read_text(encoding="utf-8") == "flight of Worst case"
    assert (tmp_path / "viewer" / "best" / "index.html").is_file()
    manifest = json.loads((tmp_path / "viewer" / "sims.json").read_text("utf-8"))
    assert manifest == {
        "ork": "Design 1.ork",
        "default": "Worst case",
        "sims": {
            "average": "average/index.html",
            "Worst case": "worst-case/index.html",
            "best": "best/index.html",
        },
    }


def test_rasaero_geometry_is_read_in_metres(tmp_path: Path) -> None:
    """The ``.CDX1`` inches become the page's change keys, tube difference stretched in."""
    ork = load_ork(write_ork(tmp_path / "t.ork"))
    geometry = read_rasaero_geometry(write_cdx(tmp_path / "t.CDX1"), ork)
    assert geometry["finCount"] == 3
    assert geometry["finRoot"] == pytest.approx(0.3048)
    assert geometry["finSpan"] == pytest.approx(0.127)
    assert geometry["finSweep"] == pytest.approx(0.2032)
    assert geometry["finThick"] == pytest.approx(0.00508)
    assert geometry["noseLength"] == pytest.approx(0.5, abs=1e-4)
    assert geometry["tailLength"] == pytest.approx(0.06096)
    assert geometry["tailAft"] == pytest.approx(0.5 * 2.8 * 0.0254)
    # the CDX1 tubes total 60 in = 1.524 m against the .ork's 1.5 m
    assert geometry["bodyLength"] == pytest.approx(ork.body_length_m + 0.024)


def test_rasaero_file_without_a_fin_is_rejected(tmp_path: Path) -> None:
    """A ``.CDX1`` that has no fin set cannot be the geometry of the table."""
    ork = load_ork(write_ork(tmp_path / "t.ork"))
    bare = tmp_path / "bare.CDX1"
    bare.write_text(
        "<RASAeroDocument><RocketDesign><NoseCone><Length>10</Length></NoseCone>"
        "</RocketDesign></RASAeroDocument>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nose cone and a fin set"):
        read_rasaero_geometry(bare, ork)


def test_reference_geometry_goes_into_the_page_data(tmp_path: Path) -> None:
    """With a ``.CDX1`` the data carries the geometry the table is for."""
    data = build_data(
        write_ork(tmp_path / "t.ork"),
        write_aero_csv(tmp_path / "t.csv"),
        write_eng(tmp_path / "t.eng"),
        "Test rocket",
        "calm",
        rasaero=write_cdx(tmp_path / "t.CDX1"),
    )
    assert data["refFile"] == "t.CDX1"
    assert data["refChange"]["finCount"] == 3
    json.dumps(data)


def test_rse_motor_is_flown_and_described(tmp_path: Path) -> None:
    """An ``.rse`` works like an ``.eng``; its masses and the saved ones are both in the data."""
    data = build_data(
        write_ork(tmp_path / "t.ork"),
        write_aero_csv(tmp_path / "t.csv"),
        write_rse(tmp_path / "m.rse"),
        "Test rocket",
        "calm",
        motor_note="newest of 2 in Thrust Curves",
    )
    motor = data["motorFile"]
    assert motor["note"] == "newest of 2 in Thrust Curves" and motor["burnS"] == 4.0
    assert motor["propKg"] == pytest.approx(3.5) and motor[
        "savedPropKg"
    ] == pytest.approx(4.0)
    assert data["base"]["motor"]["mass"] == pytest.approx(9.0)
    assert data["sixdof"]["calm"]["apogee"] > 100.0


def test_page_is_one_file_with_the_data_and_engine(
    tmp_path: Path, data: dict[str, Any]
) -> None:
    """Page is one file with the data and engine."""
    html = write_page(data, tmp_path / "out").read_text(encoding="utf-8")
    assert "Test rocket" in html
    assert "const DATA = {" in html and "/*DATA*/" not in html
    assert "WhatIf" in html and "/*ENGINE*/" not in html
    assert 'type="range"' not in html and "'range'" not in html  # no sliders
    assert 'id="updatecsv"' in html and "/api/status" in html
    assert 'id="weather"' in html and "climbOnly" in html  # apogee in every weather


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_engine_flies_the_design_and_responds_to_geometry(
    tmp_path: Path, data: dict[str, Any]
) -> None:
    """Engine flies the design and responds to geometry."""
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    script = tmp_path / "run.js"
    script.write_text(
        f"""
const W = require({json.dumps(str(_ENGINE))});
const d = require({json.dumps(str(tmp_path / "data.json"))});
const b = d.base, f = b.fins;
const c0 = {{finCount: f.count, finRoot: f.root, finTip: f.tip, finSpan: f.span, finSweep: f.sweep,
  finThick: f.thick, noseLength: b.noseLength, bodyLength: b.bodyLength, tailLength: b.tailLength,
  tailAft: b.tailAftRadius}};
b.geometry = W.applyChange(b, c0);
const p = d.presets.calm, cond = Object.assign({{leanIntoWind: true}}, p);
const base = W.fly(b, c0, cond).sum;
const wide = Object.assign({{}}, c0, {{finSpan: f.span * 1.3}});
const big = W.fly(b, wide, cond).sum;
const longer = Object.assign({{}}, c0, {{noseLength: b.noseLength + 0.2}});
const nose = W.fly(b, longer, cond).sum;
const bigOut = {{m: big.marginRail, apogee: big.apogee, finMass: big.finMass}};
const full = W.fly(b, c0, cond, {{every: 0.1}});
const none = W.fly(b, c0, cond, {{descent: false}});
b.refGeometry = W.applyChange(b, wide);  // a table made for bigger fins than the design
const lighter = W.fly(b, c0, cond).sum.apogee;
delete b.refGeometry;
console.log(JSON.stringify({{base, big: bigOut, lighter, land: full.sum.land, lastAlt: full.out.d.alt[full.out.d.alt.length - 1],
  window: [base.marginMin, base.marginLo, base.marginHi], ascent: full.out.nAscent, nT: full.out.t.length, noLand: none.sum.land === undefined, noD: none.out.d === undefined, noApogee: none.sum.apogee,
  nose: {{cg: nose.cg0, cp: nose.cpBarrowman, m: nose.marginRail}},
  mass: W.massAt(b.geometry, 0).m}}));
""",
        encoding="utf-8",
    )
    out = json.loads(
        subprocess.run(
            ["node", str(script)], check=True, capture_output=True, text=True
        ).stdout
    )
    base = out["base"]
    assert base["apogee"] > 100.0 and base["apogeeT"] > 1.0
    low, window_low, window_high = out[
        "window"
    ]  # the stability window of the History page
    assert low <= window_low <= window_high
    # airframe from the component model plus the motor file's 10 kg
    assert out["mass"] == pytest.approx(
        load_ork(tmp_path / "t.ork").airframe_mass_kg + 10.0, rel=1e-6
    )
    assert out["big"]["m"] > base["marginRail"]  # bigger fins: more stable
    assert out["big"]["apogee"] < base["apogee"]  # ...and more drag
    assert out["big"]["finMass"] > base["finMass"]
    # A table made for bigger fins than the design over-states its drag, and the model takes that out
    assert out["lighter"] > base["apogee"]
    # A longer nose moves the CG and the CP aft together, leaving the margin nearly alone
    assert out["nose"]["cg"] > base["cg0"] and out["nose"]["cp"] > base["cpBarrowman"]
    assert out["nose"]["m"] == pytest.approx(base["marginRail"], abs=0.35)
    # The descent ends on the ground. The rate it settles at depends only on the mass and the
    # canopy, so it must match the 6-DOF sim's (the two ascents of this toy design differ)
    land, six = out["land"], data["sixdof"]["calm"]["landing"]
    assert land["landed"] and out["lastAlt"] == 0.0 and land["t"] > base["apogeeT"]
    assert land["vVert"] == pytest.approx(six["vVert"], rel=0.05)
    assert land["peakG"] > 0.0 and [d["name"] for d in land["deployments"]] == [
        "main reefed",
        "main reef cut",
    ]
    assert (
        out["ascent"] == out["nT"] and out["noLand"] and out["noD"]
    )  # descent can be switched off
    # the climb-only run the weather table uses reaches the same apogee as the full run
    assert out["noApogee"] == pytest.approx(base["apogee"])
