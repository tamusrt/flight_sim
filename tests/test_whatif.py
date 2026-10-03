# ruff: noqa: E501
# pylint: disable=line-too-long
"""Tests for the predictions page: its data, its page and its flight model."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from ork_fixture import write_aero_csv, write_cdx, write_eng, write_ork, write_rse

from flight_sim.ork import load_ork
from flight_sim.whatif.build import (
    build_data,
    read_rasaero_geometry,
    read_thrust,
    reduce_aero,
    write_page,
)

_ENGINE = Path(__file__).parents[1] / "src" / "flight_sim" / "whatif" / "engine.js"


@pytest.fixture(name="data")
def fixture_data(tmp_path: Path) -> dict[str, object]:
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


def test_data_holds_the_design_and_its_references(data: dict[str, object]) -> None:
    """Data holds the design and its references."""
    assert data["defaultSim"] == "calm"
    base = data["base"]
    assert base["fins"]["count"] == 4
    assert base["motor"]["prop"] == pytest.approx(4.0)
    assert data["presets"]["calm"]["railAngleDeg"] == pytest.approx(5.0)
    assert data["openrocket"]["calm"]["summary"]["maxaltitude"] == pytest.approx(3000.0)
    assert data["sixdof"]["calm"]["apogee"] > 100.0
    json.dumps(data)  # must serialise


def test_data_holds_the_descent_and_the_6dof_landing(data: dict[str, object]) -> None:
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
    tmp_path: Path, data: dict[str, object]
) -> None:
    """Page is one file with the data and engine."""
    html = write_page(data, tmp_path / "out").read_text(encoding="utf-8")
    assert "Test rocket" in html
    assert "const DATA = {" in html and "/*DATA*/" not in html
    assert "WhatIf" in html and "/*ENGINE*/" not in html
    assert 'type="range"' not in html and "'range'" not in html  # no sliders
    assert 'id="updatecsv"' in html and "/api/status" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_engine_flies_the_design_and_responds_to_geometry(
    tmp_path: Path, data: dict[str, object]
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
  ascent: full.out.nAscent, nT: full.out.t.length, noLand: none.sum.land === undefined, noD: none.out.d === undefined,
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
