"""Build the predictions page and the 3D viewer for one OpenRocket design.

Usage::

    python -m flight_sim.whatif.build --ork D.ork --aero D_aero.csv \\
        --motor D.eng|D.rse --out site/predictions [--name "SRT14"] [--sim average]

The page (``index.html``) shows the predictions of the committed design from a
quick flight model; ``viewer/index.html`` is the full 6-DOF flight in 3D, flown
from the design as committed. Both come from the same ``.ork``, RASAero CSV and
thrust curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from flight_sim.__main__ import (
    apply_mass_properties,
    get_default_config,
    get_default_properties,
    get_launch_state,
)
from flight_sim.events import APOGEE, rail_exit
from flight_sim.integration import adaptive_step
from flight_sim.motor_file import load_motor
from flight_sim.ork import OrkRocket, SavedSim, load_ork, saved_motor
from flight_sim.ork_profile import motor_with_file, pad_pressure_pa, profile_from_ork
from flight_sim.units import scalar

_HERE = Path(__file__).parent
_ALPHAS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)


def reduce_aero(csv_path: str | Path, reference_length_m: float) -> dict[str, Any]:
    """Cut the RASAero CSV down to what the page needs, at roll angle zero.

    Returns ``mach``, ``alpha`` (degrees) and, per Mach and alpha, the axial
    coefficient ``ca``, the normal coefficient ``cn`` and the centre of
    pressure ``xcp`` in metres aft of the nose.
    """
    table = pd.read_csv(csv_path)
    table = table[(table["Phi"] == 0) & (table["Alpha"].isin(_ALPHAS))]
    machs = sorted(table["Mach"].unique())
    ca, cn, xcp = [], [], []
    for mach in machs:
        rows = table[table["Mach"] == mach].set_index("Alpha").reindex(_ALPHAS)
        cz = rows["Cz"].to_numpy(float)
        cm = rows["CMy"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            arm = np.where(np.abs(cz) > 1e-9, cm * reference_length_m / cz, np.nan)
        arm[0] = arm[1]  # no normal force at zero angle of attack
        ca.append([round(-v, 4) for v in rows["Cx"]])
        cn.append([round(-v, 4) for v in cz])
        xcp.append([round(float(v), 4) for v in arm])
    return {
        "mach": [round(float(m), 3) for m in machs],
        "alpha": list(_ALPHAS),
        "ca": ca,
        "cn": cn,
        "xcp": xcp,
    }


def read_thrust(motor_path: str | Path) -> list[list[float]]:
    """Time and thrust pairs of an ``.eng`` or ``.rse`` motor file."""
    return load_motor(motor_path).thrust


def _base(
    ork: OrkRocket, aero: dict[str, Any], motor: Any, thrust: list[list[float]]
) -> dict[str, Any]:
    fins = ork.fins
    return {
        "diameter": ork.reference_diameter_m,
        "noseShape": ork.nose_shape,
        "noseLength": ork.nose_length_m,
        "bodyStart": ork.body_start_x_m,
        "bodyLength": ork.body_length_m,
        "bodyWall": ork.body_wall_m,
        "bodyDensity": ork.body_density_kg_m3,
        "tailLength": ork.tail_length_m,
        "tailForeRadius": ork.tail_fore_radius_m,
        "tailAftRadius": ork.tail_aft_radius_m,
        "tailWall": ork.tail_wall_m,
        "tailDensity": ork.tail_density_kg_m3,
        "length": ork.length_m,
        "items": [
            {
                "name": i.name,
                "kind": i.kind,
                "mass": i.mass_kg,
                "x": i.x_m,
                "len": i.length_m,
            }
            for i in ork.items
        ],
        "fins": {
            "count": fins.count,
            "root": fins.root_chord_m,
            "tip": fins.tip_chord_m,
            "span": fins.span_m,
            "sweep": fins.sweep_m,
            "thick": fins.thickness_m,
            "teX": fins.root_trailing_edge_x_m,
            "density": fins.density_kg_m3,
            "airfoil": fins.airfoil,
        },
        "motor": {"mass": motor.mass_kg, "prop": motor.propellant_kg, "thrust": thrust},
        "aero": aero,
    }


_IN = 0.0254


def read_rasaero_geometry(cdx_path: Path, ork: OrkRocket) -> dict[str, float]:
    """The geometry a RASAero ``.CDX1`` file describes, as the page's change keys.

    The aero CSV is whatever RASAero made from that file, so this is the
    geometry the table is for, which may lag behind the ``.ork``.
    """
    root = ET.parse(cdx_path).getroot()
    design = root.find("RocketDesign")
    if design is None:
        raise ValueError(f"{cdx_path}: no RocketDesign in the file")
    nose = design.find("NoseCone")
    fin = design.find(".//Fin")
    tail = design.find("BoatTail")
    tubes = [float(t.findtext("Length", "0")) for t in design.findall("BodyTube")]
    if nose is None or fin is None:
        raise ValueError(f"{cdx_path}: needs a nose cone and a fin set")
    # The .ork splits the same stretch of tube differently, so the difference in
    # their total length goes into the one tube the page stretches.
    ork_tubes = ork.length_m - ork.nose_length_m - ork.tail_length_m
    stretch = sum(tubes) * _IN - ork_tubes
    return {
        "finCount": float(fin.findtext("Count", "4")),
        "finRoot": float(fin.findtext("Chord", "0")) * _IN,
        "finTip": float(fin.findtext("TipChord", "0")) * _IN,
        "finSpan": float(fin.findtext("Span", "0")) * _IN,
        "finSweep": float(fin.findtext("SweepDistance", "0")) * _IN,
        "finThick": float(fin.findtext("Thickness", "0")) * _IN,
        "noseLength": float(nose.findtext("Length", "0")) * _IN,
        "bodyLength": ork.body_length_m + stretch,
        "tailLength": float(tail.findtext("Length", "0")) * _IN
        if tail is not None
        else 0.0,
        "tailAft": 0.5 * float(tail.findtext("RearDiameter", "0")) * _IN
        if tail is not None
        else 0.5 * ork.reference_diameter_m,
    }


def _preset(sim: SavedSim) -> dict[str, float]:

    c = sim.conditions
    pad_k = c.get("basetemperature", 288.15)
    return {
        "railAngleDeg": c["launchrodangle"],
        "rodLength": c["launchrodlength"],
        "windMs": c["windaverage"],
        "padK": pad_k,
        "padPa": pad_pressure_pa(
            c.get("basepressure", 101325.0), pad_k, c["launchaltitude"]
        ),
        "padAlt": c["launchaltitude"],
    }


def _or_series(sim: SavedSim) -> dict[str, Any]:
    s = sim.series
    t = s["Time"]
    keep = (t <= sim.summary["timetoapogee"]) & (np.arange(len(t)) % 10 == 0)
    cal = s["Stability margin calibers"]
    return {
        "summary": sim.summary,
        "t": [round(float(v), 2) for v in t[keep]],
        "alt": [round(float(v), 1) for v in s["Altitude"][keep]],
        "v": [round(float(v), 1) for v in s["Total velocity"][keep]],
        "mach": [round(float(v), 3) for v in s["Mach number"][keep]],
        "acc": [round(float(v) / 9.80665, 2) for v in s["Total acceleration"][keep]],
        "margin": [None if np.isnan(v) else round(float(v), 2) for v in cal[keep]],
    }


def _six_dof(profile: Any) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Fly the 6-DOF sim to apogee and keep a coarse record of it."""
    properties = get_default_properties(profile)
    state = get_launch_state(profile)
    config = get_default_config(profile)
    events = (rail_exit(profile.rail), APOGEE)
    dt = scalar(0.01, "s")
    time = 0.0
    rows = []
    next_sample = 0.0
    while True:
        velocity = state.velocity.m_as("m/s")
        if time >= next_sample:
            next_sample += 0.5
            rows.append(
                (
                    time,
                    float(state.position.m_as("m")[0]),
                    float(np.linalg.norm(velocity)),
                )
            )
        if time > 400:
            break
        state, taken, dt, hit = adaptive_step(
            time, state, properties, config, dt, events=events
        )
        time += float(taken.m_as("s"))
        apply_mass_properties(state, profile)
        if hit is APOGEE:
            rows.append(
                (
                    time,
                    float(state.position.m_as("m")[0]),
                    float(np.linalg.norm(state.velocity.m_as("m/s"))),
                )
            )
            break
    data = np.array(rows)
    top = int(data[:, 1].argmax())
    return {
        "t": [round(float(v), 2) for v in data[:, 0]],
        "alt": [round(float(v), 1) for v in data[:, 1]],
        "v": [round(float(v), 1) for v in data[:, 2]],
        "apogee": round(float(data[top, 1]), 1),
        "apogeeT": round(float(data[top, 0]), 2),
        "vmax": round(float(data[:, 2].max()), 1),
    }


def build_data(  # pylint: disable=too-many-locals
    ork_path: Path,
    aero_csv: Path,
    motor_path: Path,
    name: str,
    sim: str | None,
    *,
    rasaero: Path | None = None,
    motor_note: str = "",
) -> dict[str, Any]:
    """Everything the page needs, as one JSON-able dict."""

    ork = load_ork(ork_path)
    default_sim = sim if sim in ork.sims else next(iter(ork.sims))
    motor_file = load_motor(motor_path)
    saved = saved_motor(ork, default_sim)
    motor = motor_with_file(saved, motor_file)
    aero = reduce_aero(aero_csv, ork.reference_diameter_m)
    sims = {n: _preset(s) for n, s in ork.sims.items()}
    six = {
        n: _six_dof(
            profile_from_ork(ork_path, str(aero_csv), str(motor_path), sim=n, name=name)
        )
        for n in ork.sims
    }
    return {
        "name": name,
        "design": ork.name,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "files": {
            "ork": ork_path.name,
            "orkHash": hashlib.sha256(ork_path.read_bytes()).hexdigest()[:10],
            "aero": aero_csv.name,
            "motor": motor_path.name,
        },
        "motorFile": {
            "name": motor_file.name,
            "note": motor_note,
            "propKg": motor_file.propellant_kg,
            "totalKg": motor_file.total_kg,
            "burnS": motor_file.burn_s,
            "impulse": motor_file.impulse_ns,
            "peakN": motor_file.peak_n,
            "savedPropKg": saved.propellant_kg,
            "savedTotalKg": saved.mass_kg,
        },
        "refChange": None if rasaero is None else read_rasaero_geometry(rasaero, ork),
        "refFile": None if rasaero is None else rasaero.name,
        "defaultSim": default_sim,
        "base": _base(ork, aero, motor, motor_file.thrust),
        "presets": sims,
        "openrocket": {n: _or_series(s) for n, s in ork.sims.items()},
        "sixdof": six,
    }


def write_page(data: dict[str, Any], out_dir: Path) -> Path:
    """Write ``index.html`` with the data and the engine inlined."""
    template = (_HERE / "page.html").read_text(encoding="utf-8")
    engine = (_HERE / "engine.js").read_text(encoding="utf-8")
    page = template.replace("/*ENGINE*/", engine).replace(
        "/*DATA*/null", json.dumps(data, separators=(",", ":"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def write_viewer(args: argparse.Namespace, out_dir: Path, sim: str) -> Path:
    """Fly the 6-DOF sim to apogee and write the 3D viewer page."""
    target = out_dir / "viewer" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "flight_sim.visual_run", "--apogee", "--no-open",
        "--ork", args.ork, "--aero", args.aero, "--motor", args.motor,
        "--ork-sim", sim, "--name", args.name, "--output", str(target),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    return target


def main(argv: list[str] | None = None) -> None:
    """Command line entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ork", required=True)
    parser.add_argument("--aero", required=True)
    parser.add_argument("--motor", required=True, help="the .eng or .rse file")
    parser.add_argument("--motor-note", default="", help="how the motor was chosen")
    parser.add_argument("--out", default=None)
    parser.add_argument("--name", default="SRT14")
    parser.add_argument(
        "--sim", default=None, help="saved OpenRocket sim for the default launch"
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--rasaero",
        default=None,
        help="the RASAero .CDX1 the aero CSV was made from (default: same as the .ork)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        parser.error("--out is required")
    out = Path(args.out)
    data = build_data(
        Path(args.ork),
        Path(args.aero),
        Path(args.motor),
        args.name,
        args.sim,
        rasaero=None if args.rasaero is None else Path(args.rasaero),
        motor_note=args.motor_note,
    )
    print(f"predictions page: {write_page(data, out)}")
    if not args.no_viewer:
        print(f"3D viewer: {write_viewer(args, out, data['defaultSim'])}")


if __name__ == "__main__":
    main()
