"""Build the predictions page and the 3D viewer for one OpenRocket design.

Usage::

    python -m flight_sim.whatif.build --ork D.ork --aero D_aero.csv \\
        --motor D.eng|D.rse --out site/predictions [--name "SRT14"] [--sim average]

The page (``index.html``) shows the predictions of the committed design: Jarvis is
the full 6-DOF flight, up to the apogee and down to the ground under the original
(Sol Invictus) descent, and ``viewer/<name>/index.html`` plays each saved simulation's
flight in 3D (Vision). The RASAero line is a quick flight model on the RASAero table.
All come from the same ``.ork``, RASAero CSV and thrust curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim import recovery_extension
from flight_sim.__main__ import (
    get_default_config,
    get_launch_state,
    get_profile_properties,
)
from flight_sim.descent import DescentResult, ReefedParachute, air_at
from flight_sim.events import APOGEE, IMPACT, peak_vertical_velocity
from flight_sim.integration import adaptive_step
from flight_sim.motor_file import load_motor
from flight_sim.ork import OrkRocket, SavedSim, load_ork, saved_motor
from flight_sim.ork_profile import motor_with_file, pad_pressure_pa, profile_from_ork
from flight_sim.units import scalar
from flight_sim.utilities.dcm import aero_angles, body_to_world
from flight_sim.visualize import TelemetryLog, static_centre_of_pressure
from flight_sim.whatif import script_json
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
    # RASAero's "Distance from the base of the tube": from the aft end of the
    # fins' tube to their leading edge
    fin_from_base = (
        {"finFromBase": float(fin.findtext("Location", "0")) * _IN}
        if fin.find("Location") is not None
        else {}
    )
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
        **fin_from_base,
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
        **_stages(descent),
    }


def _stages(descent: DescentResult) -> dict[str, Any]:
    """Drogue descent rate and main deployment altitude, for the IREC checks.

    With two or more canopy events, the drogue rate is the vertical speed just before
    the last opens (the drogue has settled by then); the main opens at the last
    one's altitude. With one, there is no drogue and the main opens at the first.
    """
    deps = descent.deployments
    if not deps:
        return {"drogueV": None, "mainAlt": None}
    if len(deps) < 2:
        return {"drogueV": None, "mainAlt": round(deps[0].altitude_m, 1)}
    t_main = deps[-1].time_s
    before = [
        abs(float(s.velocity.m_as("m/s")[0]))
        for t, s in zip(descent.times_s, descent.states, strict=True)
        if t < t_main - 0.5
    ]
    return {
        "drogueV": round(before[-1], 2) if before else None,
        "mainAlt": round(deps[-1].altitude_m, 1),
    }


# A row of the climb every tenth of a second, and one of the way down every second
_CLIMB_STEP_S = 0.1
# The History page measures stability and angle of attack above this speed
_FAST_M_S = 30.0
_G0 = 9.80665


def _rounded(value: float, digits: int) -> float | None:
    """A number rounded, or None for NaN (which JSON cannot hold)."""
    return None if math.isnan(value) else round(float(value), digits)


class _Climb:  # pylint: disable=too-many-instance-attributes
    """The climb of the 6-DOF flight: a row every 0.1 s, and the figures of the flight.

    Everything the page shows for Jarvis is taken from here, so it is the flight
    that Vision plays back.
    """

    def __init__(self, properties: Any, config: Any) -> None:
        self.properties = properties
        self.config = config
        self.diameter = float(properties.aero_table.reference_length_m)
        self.log = TelemetryLog(properties, config)
        self.velocities: list[np.ndarray] = []
        self.rows: list[
            tuple[float, float, float, float, float]
        ] = []  # t, altitude, speed, Mach, margin
        self.q_max = 0.0
        self.aoa_max = 0.0
        self.fast_margins: list[float] = []
        self.rail_speed: float | None = None
        self.rail_margin: float | None = None
        self.burnout: tuple[float, float, float] | None = None  # time, altitude, mass
        self.drift = 0.0

    def margin(self, state: Any, time: float) -> float:
        """Stability in calibers: (centre of pressure - CG) over the diameter."""
        position = float(state.position.m_as("m")[0])
        air, wind = air_at(self.config, position)
        speed = float(np.linalg.norm(state.velocity.m_as("m/s") - wind))
        cp = static_centre_of_pressure(self.properties, speed / air.speed_of_sound)
        cg = -float(self.properties.mass_properties(time).cg_location[0])
        return (cp - cg) / self.diameter

    def add(self, time: float, state: Any) -> None:
        """Record the state at a time."""
        self.log.record(time, state)
        position = state.position.m_as("m")
        velocity = state.velocity.m_as("m/s")
        speed = float(np.linalg.norm(velocity))
        mach = float(self.log.rows["mach"][-1])
        margin = self.margin(state, time)
        self.velocities.append(velocity)
        self.rows.append((time, float(position[0]), speed, mach, margin))
        self.q_max = max(self.q_max, float(self.log.rows["q"][-1]))
        if speed > _FAST_M_S:
            _, wind = air_at(self.config, float(position[0]))
            airspeed_body = body_to_world(state.orientation).T @ (velocity - wind)
            self.aoa_max = max(
                self.aoa_max, math.degrees(aero_angles(airspeed_body)[0])
            )
            if not math.isnan(margin):
                self.fast_margins.append(margin)
        burnt_out = time > 0.5 and self.properties.engine.get_thrust(time) <= 0.0
        if self.burnout is None and burnt_out:
            mass = self.properties.mass_properties(time).mass
            self.burnout = (time, float(position[0]), mass)
        self.drift = float(np.hypot(position[1], position[2]))

    def accelerations(self) -> list[float]:
        """The acceleration in g at each row, from the change in velocity."""
        out = [0.0]
        for i in range(1, len(self.rows)):
            step = self.rows[i][0] - self.rows[i - 1][0]
            change = float(np.linalg.norm(self.velocities[i] - self.velocities[i - 1]))
            out.append(change / step / _G0 if step > 1e-9 else out[-1])
        return out

    def figures(self) -> dict[str, Any]:
        """The figures of the flight the page's tables show."""
        data = np.array([row[:4] for row in self.rows])
        accelerations = self.accelerations()
        margins = [m for m in (r[4] for r in self.rows) if not math.isnan(m)]
        window = self.fast_margins or margins
        mass0 = self.properties.mass_properties(0.0)
        last = self.rows[-1]
        burnout = self.burnout or (
            last[0],
            last[1],
            self.properties.mass_properties(last[0]).mass,
        )  # still burning when the climb ends
        top = int(np.argmax(data[:, 1]))
        later = [
            a for a, row in zip(accelerations, self.rows, strict=True) if row[0] > 0.2
        ]
        return {
            "apogee": round(float(data[top, 1]), 1),
            "apogeeT": round(float(data[top, 0]), 2),
            "vmax": round(float(data[:, 2].max()), 1),
            "machMax": round(float(data[:, 3].max()), 3),
            "qMax": round(self.q_max, 1),
            "accMax": round(max(later, default=0.0), 2),
            "railV": None if self.rail_speed is None else round(self.rail_speed, 1),
            "marginRail": None
            if self.rail_margin is None
            else round(self.rail_margin, 3),
            "marginLo": round(min(window), 3) if window else None,
            "marginHi": round(max(window), 3) if window else None,
            "burnoutAlt": round(burnout[1], 1),
            "aoaMax": round(self.aoa_max, 1),
            "drift": round(self.drift, 1),
            "mass0": round(mass0.mass, 3),
            "mass1": round(burnout[2], 3),
            "cg0": round(-float(mass0.cg_location[0]), 3),
            "cp": round(static_centre_of_pressure(self.properties, 0.3), 3),
        }

    def series(self) -> dict[str, list[Any]]:
        """The rows as lists for the page."""
        return {
            "t": [round(r[0], 3) for r in self.rows],
            "alt": [round(r[1], 1) for r in self.rows],
            "v": [round(r[2], 1) for r in self.rows],
            "mach": [_rounded(r[3], 3) for r in self.rows],
            "acc": [round(a, 2) for a in self.accelerations()],
            "marginCal": [_rounded(r[4], 3) for r in self.rows],
        }


def _six_dof(profile: Any) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Fly the 6-DOF sim to apogee, then down under the descent.

    This is Jarvis: every number and line the page shows for it comes from this flight.
    The flight is the one VISION replays: the same climb (every step kept, the same
    events) and the same full recovery model (``recovery_extension``: the flight
    computer, the ejection, the tumble, the shock cord and the swing under the canopy),
    so the page's descent numbers are the numbers of the flight in VISION.
    """
    properties = get_profile_properties(profile)
    state = get_launch_state(profile)
    config = get_default_config(profile)
    climb = _Climb(properties, config)
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)
    flight: list[tuple[float, Any]] = []
    dt = scalar(0.01, "s")
    time = 0.0
    next_sample = 0.0
    hit = None
    while True:
        flight.append((time, state))
        if hit is APOGEE or hit is IMPACT:
            break
        if time >= next_sample - 1e-9:
            climb.add(time, state)
            while next_sample <= time + 1e-9:  # one row per 0.1 s, however big the step
                next_sample += _CLIMB_STEP_S
        if time > 400:
            break
        state, taken, dt, hit = adaptive_step(
            time, state, properties, config, dt, events=events
        )
        time += float(taken.m_as("s"))
        if hit is not None and hit.name == "rail exit":
            climb.rail_speed = float(np.linalg.norm(state.velocity.m_as("m/s")))
            climb.rail_margin = climb.margin(state, time)
        if hit is APOGEE:
            climb.add(time, state)
    result: dict[str, Any] = climb.figures()
    series = climb.series()
    if profile.scheme is not None and hit is APOGEE:
        mass = properties.mass_properties(time).mass
        plan = recovery_extension.plan_full_recovery(
            flight, config, profile.scheme, properties.mass_properties(time), properties
        )
        descent = plan.descent
        # about one sample a second of the way down, and the last one
        down = []
        next_t = float(descent.times_s[0]) + 1.0
        for i, (t, s) in enumerate(zip(descent.times_s, descent.states, strict=True)):
            if t >= next_t or i == len(descent.times_s) - 1:
                down.append(_row(t, s))
                next_t = float(t) + 1.0
        result["landing"] = _landing(descent, mass)
        series["t"] += [round(float(r[0]), 2) for r in down]
        series["alt"] += [round(float(r[1]), 1) for r in down]
        series["v"] += [round(float(r[2]), 1) for r in down]
        for key in (
            "mach",
            "acc",
            "marginCal",
        ):  # the way down has only height and speed
            series[key] += [None] * len(down)
    result.update(series)
    return result


def _recovery_data(profile: Any) -> dict[str, Any] | None:
    """The descent the profile flies, as the page's model needs it."""
    scheme = profile.scheme
    if scheme is None:
        return None
    recovery = scheme.recovery
    first = recovery.parachutes[0]
    text = "Sol Invictus recovery (placeholder): one parachute, single separation"
    if isinstance(first, ReefedParachute):
        inches = first.diameter_m / 0.0254
        text = (
            f"Sol Invictus recovery (placeholder): {inches:.0f} in reefed main, "
            f"deployed {first.deploy_delay_s:g} s after apogee, "
            f"disreefed at {first.disreef_altitude_m / 0.3048:,.0f} ft"
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


def slug(name: str) -> str:
    """A name as a folder name: lower case letters and digits, with single dashes."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "sim"


def viewer_paths(sims: list[str]) -> dict[str, str]:
    """Where each saved simulation's 3D flight is, relative to the folder of the page.

    Two names that make the same folder name are told apart by a number.
    """
    paths: dict[str, str] = {}
    used: set[str] = set()
    for sim in sims:
        folder = base = slug(sim)
        number = 2
        while folder in used:
            folder = f"{base}-{number}"
            number += 1
        used.add(folder)
        paths[sim] = f"viewer/{folder}/index.html"
    return paths


_FT = 0.3048


def read_rasaero_results(path: Path, sims: list[str]) -> list[dict[str, Any]]:
    """RASAero II's own flight results, typed in from its Flight window.

    The file (JSON) has a ``runs`` list; each run has a ``label``, the ``sim`` it
    matches in the OpenRocket design (or null), and ``apogee_ft``,
    ``max_velocity_ft_s`` and ``time_to_apogee_s`` as RASAero shows them. They are
    returned in metres and seconds, with ``sim`` set to None when the design has no
    launch condition of that name.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for run in data.get("runs", []):
        sim = run.get("sim")
        runs.append(
            {
                "label": str(run.get("label", "")),
                "sim": sim if sim in sims else None,
                "simAsked": sim,
                "apogee": run["apogee_ft"] * _FT,
                "vmax": (
                    None
                    if run.get("max_velocity_ft_s") is None
                    else run["max_velocity_ft_s"] * _FT
                ),
                "apogeeT": run.get("time_to_apogee_s"),
                "railDeg": run.get("rail_deg"),
                "windMph": run.get("wind_mph"),
            }
        )
    return runs


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
    rasaero_results: Path | None = None,
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
        "viewers": viewer_paths(list(ork.sims)),
        "rasaero": (
            None
            if rasaero_results is None
            else {
                "file": rasaero_results.name,
                "runs": read_rasaero_results(rasaero_results, list(ork.sims)),
            }
        ),
    }


def write_page(data: dict[str, Any], out_dir: Path) -> Path:
    """Write ``index.html`` with the data and the engine inlined."""
    template = (_HERE / "page.html").read_text(encoding="utf-8")
    engine = (_HERE / "engine.js").read_text(encoding="utf-8")
    page = template.replace("/*ENGINE*/", engine).replace(
        "/*DATA*/null", script_json(data)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def _write_one_viewer(args: argparse.Namespace, target: Path, sim: str) -> None:
    """Fly one saved simulation to the ground and write its 3D viewer page."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "flight_sim.visual_run", "--no-open",
        "--ork", args.ork, "--aero", args.aero, "--motor", args.motor,
        "--ork-sim", sim, "--name", f"{args.name} · {sim}", "--output", str(target),
    ]  # fmt: skip
    subprocess.run(command, check=True)


def write_viewers(
    args: argparse.Namespace, out_dir: Path, paths: dict[str, str], default: str
) -> Path:
    """Write the 3D viewer of every saved simulation, for Vision to follow.

    ``viewer/<name>/index.html`` is one simulation's flight;
    ``viewer/index.html`` is the default simulation's, as it always was;
    ``viewer/sims.json`` lists them for the History page.
    """
    fly_viewers(args, out_dir, paths)
    return finish_viewers(args, out_dir, paths, default)


def fly_viewers(args: argparse.Namespace, out_dir: Path, paths: dict[str, str]) -> None:
    """Fly every saved simulation and write its 3D viewer page (in parallel)."""
    names = list(paths)
    with ThreadPoolExecutor(max_workers=min(len(names), 4)) as pool:
        list(
            pool.map(
                lambda sim: _write_one_viewer(args, out_dir / paths[sim], sim), names
            )
        )


def finish_viewers(
    args: argparse.Namespace, out_dir: Path, paths: dict[str, str], default: str
) -> Path:
    """Copy the default simulation's viewer to ``viewer/index.html``; list them all."""
    first = out_dir / "viewer" / "index.html"
    shutil.copyfile(out_dir / paths[default], first)
    manifest = {
        "ork": Path(args.ork).name,
        "default": default,
        "sims": {sim: path.removeprefix("viewer/") for sim, path in paths.items()},
    }
    (out_dir / "viewer" / "sims.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return first


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
        "--rasaero-results",
        default=None,
        help="RASAero II's flight results (JSON), shown next to Jarvis and OpenRocket",
    )
    parser.add_argument(
        "--rasaero",
        default=None,
        help="the RASAero .CDX1 the aero CSV was made from (default: same as the .ork)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        parser.error("--out is required")
    out = Path(args.out)
    # VISION's flights run in their own processes while Jarvis flies the same flights
    # here for the page, so the two share the computer's cores instead of taking turns
    flying = ThreadPoolExecutor(max_workers=1)
    viewers_done = (
        None
        if args.no_viewer
        else flying.submit(
            fly_viewers, args, out, viewer_paths(list(load_ork(args.ork).sims))
        )
    )
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
        rasaero_results=(
            None if args.rasaero_results is None else Path(args.rasaero_results)
        ),
    )
    print(f"predictions page: {write_page(data, out)}")
    if viewers_done is not None:
        viewers_done.result()  # a viewer that failed stops the build here, as before
        shown = finish_viewers(args, out, data["viewers"], data["defaultSim"])
        print(f"3D viewers: {shown.parent} ({len(data['viewers'])} flights)")
    flying.shutdown()


if __name__ == "__main__":
    main()
