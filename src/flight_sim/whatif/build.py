"""Build the predictions page and the 3D viewer for one OpenRocket design.

Usage::

    python -m flight_sim.whatif.build --ork D.ork --aero D_aero.csv \\
        --motor D.eng|D.rse --out site/predictions [--name "SRT14"] [--sim average]

The page (``index.html``) shows the predictions of the committed design from a
quick flight model, up to the apogee and down to the ground under the original
(Sol Invictus) descent; ``viewer/index.html`` is the full 6-DOF flight in 3D,
flown from the design as committed. Both come from the same ``.ork``, RASAero CSV and
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
from flight_sim.descent import DescentResult, ReefedParachute, simulate_descent
from flight_sim.events import APOGEE, rail_exit
from flight_sim.integration import adaptive_step
from flight_sim.motor_file import load_motor
from flight_sim.ork import OrkRocket, SavedSim, load_ork, saved_motor
from flight_sim.ork_profile import motor_with_file, pad_pressure_pa, profile_from_ork
from flight_sim.units import scalar
from flight_sim.whatif.history import from_history, from_saved, load_history

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


def _row(time: float, state: Any) -> tuple[float, float, float]:
    """Time, height above the pad and speed of one sample."""
    return (
        time,
        float(state.position.m_as("m")[0]),
        float(np.linalg.norm(state.velocity.m_as("m/s"))),
    )


def _landing(descent: DescentResult, mass_kg: float) -> dict[str, Any]:
    """What the descent ends with: when, how fast, how far and how hard."""
    last = descent.states[-1]
    position = last.position.m_as("m")
    speed = float(np.linalg.norm(last.velocity.m_as("m/s")))
    return {
        "landed": descent.landed,
        "t": round(descent.times_s[-1], 1),
        "v": round(speed, 2),
        "vVert": round(abs(float(last.velocity.m_as("m/s")[0])), 2),
        "drift": round(float(np.hypot(position[1], position[2])), 1),
        "energy": round(0.5 * mass_kg * speed**2, 1),
        "peakG": round(
            max((d.peak_load_g for d in descent.deployments), default=0.0), 2
        ),
        "deployments": [
            {"name": d.name, "t": round(d.time_s, 2), "alt": round(d.altitude_m, 1)}
            for d in descent.deployments
        ],
    }


def _six_dof(profile: Any) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Fly the 6-DOF sim to apogee, then down under the descent (coarse record)."""
    properties = get_default_properties(profile)
    state = get_launch_state(profile)
    config = get_default_config(profile)
    events = (rail_exit(profile.rail), APOGEE)
    dt = scalar(0.01, "s")
    time = 0.0
    rows = []
    next_sample = 0.0
    while True:
        if time >= next_sample:
            next_sample += 0.5
            rows.append(_row(time, state))
        if time > 400:
            break
        state, taken, dt, hit = adaptive_step(
            time, state, properties, config, dt, events=events
        )
        time += float(taken.m_as("s"))
        apply_mass_properties(state, profile)
        if hit is APOGEE:
            rows.append(_row(time, state))
            break
    data = np.array(rows)
    top = int(data[:, 1].argmax())
    result: dict[str, Any] = {
        "apogee": round(float(data[top, 1]), 1),
        "apogeeT": round(float(data[top, 0]), 2),
        "vmax": round(float(data[:, 2].max()), 1),
    }
    if profile.scheme is not None and hit is APOGEE:
        mass = float(state.current_mass.m_as("kg"))
        descent = simulate_descent(time, state, config, profile.recovery)
        # one sample a second of the way down (the descent is sampled every 0.1 s)
        rows += [
            _row(t, s)
            for i, (t, s) in enumerate(
                zip(descent.times_s, descent.states, strict=True)
            )
            if i % 10 == 9 or i == len(descent.times_s) - 1
        ]
        result["landing"] = _landing(descent, mass)
        data = np.array(rows)
    result["t"] = [round(float(v), 2) for v in data[:, 0]]
    result["alt"] = [round(float(v), 1) for v in data[:, 1]]
    result["v"] = [round(float(v), 1) for v in data[:, 2]]
    return result


def _recovery_data(profile: Any) -> dict[str, Any] | None:
    """The descent the profile flies, as the page's model needs it."""
    scheme = profile.scheme
    if scheme is None:
        return None
    recovery = scheme.recovery
    first = recovery.parachutes[0]
    text = "Sol Invictus recovery as a stand-in: one canopy on a single separation"
    if isinstance(first, ReefedParachute):
        inches = first.diameter_m / 0.0254
        text = (
            f"Sol Invictus recovery as a stand-in: {inches:.0f} in reefed main, "
            f"out {first.deploy_delay_s:g} s after apogee, "
            f"reef cut at {first.disreef_altitude_m / 0.3048:,.0f} ft"
        )
    return {
        "label": text,
        "bodyCdA": recovery.body_drag_area_m2,
        "stages": [
            {
                "name": stage.name,
                "cda": stage.drag_area_m2,
                "fill": stage.fill_distance_m,
                "delay": stage.deploy_delay_s,
                "alt": stage.deploy_altitude_m,
                "after": stage.after,
            }
            for stage in recovery.stages
        ],
    }


def build_data(  # pylint: disable=too-many-locals,too-many-arguments
    ork_path: Path,
    aero_csv: Path,
    motor_path: Path,
    name: str,
    sim: str | None,
    *,
    rasaero: Path | None = None,
    motor_note: str = "",
    history_site: Path | None = None,
    history_motor: str = "",
) -> dict[str, Any]:
    """Everything the page needs, as one JSON-able dict.

    OpenRocket's numbers come from the History site in ``history_site`` when it has
    this design, and otherwise from the results saved in the design file.
    """

    ork = load_ork(ork_path)
    default_sim = sim if sim in ork.sims else next(iter(ork.sims))
    motor_file = load_motor(motor_path)
    saved = saved_motor(ork, default_sim)
    motor = motor_with_file(saved, motor_file)
    aero = reduce_aero(aero_csv, ork.reference_diameter_m)
    sims = {n: _preset(s) for n, s in ork.sims.items()}
    profiles = {
        n: profile_from_ork(ork_path, str(aero_csv), str(motor_path), sim=n, name=name)
        for n in ork.sims
    }
    six = {n: _six_dof(profile) for n, profile in profiles.items()}
    base = _base(ork, aero, motor, motor_file.thrust)
    base["recovery"] = _recovery_data(profiles[default_sim])
    history = {} if history_site is None else load_history(history_site, ork_path.name)
    openrocket = {
        n: (from_history(history[n]) if n in history else None) or from_saved(s)
        for n, s in ork.sims.items()
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
        "base": base,
        "presets": sims,
        "openrocket": openrocket,
        "openrocketMotor": history_motor,
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
    """Fly the 6-DOF sim, down to the ground, and write the 3D viewer page."""
    target = out_dir / "viewer" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "flight_sim.visual_run", "--no-open",
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
    parser.add_argument(
        "--history-site",
        default=None,
        help="the History site folder (data.json, flights/) to take OpenRocket from",
    )
    parser.add_argument(
        "--history-motor",
        default="",
        help="motor file the History runs use (warns when Jarvis flies another)",
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
        history_site=None if args.history_site is None else Path(args.history_site),
        history_motor=args.history_motor,
    )
    print(f"predictions page: {write_page(data, out)}")
    if not args.no_viewer:
        print(f"3D viewer: {write_viewer(args, out, data['defaultSim'])}")


if __name__ == "__main__":
    main()
