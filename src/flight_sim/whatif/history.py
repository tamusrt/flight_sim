"""OpenRocket results for the predictions page.

They come from the History site, or from the design file when there is none.

The History site (``tools/openrocket/or_ci.py history --site``) runs OpenRocket on
every committed design. It writes ``data.json`` plus one ``flights/<id>.js`` file per
simulation. The predictions page puts Jarvis next to those same numbers, so the two
pages never disagree. When there is no History site (a build on a laptop), the results
saved inside the OpenRocket file are used instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from flight_sim.ork import SavedSim

G0 = 9.80665
_RAIL_MIN_SPEED = 30.0  # as on the History page: stability is judged above this speed
_STEP_S = 0.25  # one sample every quarter second is plenty for a line on a chart
_FLIGHT_JS = re.compile(r"^window\.__flight\((.*)\);\s*$", re.DOTALL)


def _clean(values: Any) -> np.ndarray:
    """A float array with undefined samples as NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _thin(t: np.ndarray, stop: int) -> np.ndarray:
    """Indexes of about one sample per quarter second up to ``stop`` (inclusive)."""
    keep = [0]
    for i in range(1, stop + 1):
        if t[i] - t[keep[-1]] >= _STEP_S:
            keep.append(i)
    if keep[-1] != stop:
        keep.append(stop)
    return np.array(keep)


def _peak(values: np.ndarray) -> float:
    """Largest defined value, or NaN when none is defined."""
    values = values[~np.isnan(values)]
    return float(values.max()) if len(values) else float("nan")


def _round(values: np.ndarray, digits: int) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), digits) for v in values]


def climb(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    t: np.ndarray,
    alt: np.ndarray,
    speed: np.ndarray,
    mach: np.ndarray,
    acc: np.ndarray,
    cal: np.ndarray,
    drag: np.ndarray,
    q: np.ndarray,
    apogee_t: float,
    rail_t: float,
) -> dict[str, Any]:
    """The climb to apogee as chart lines and as the numbers of the results table.

    Arrays are at the sample rate of the source; ``acc`` is in m/s2, ``q`` in Pa.
    """
    stop = int(np.searchsorted(t, apogee_t, side="right")) - 1
    stop = max(1, min(stop, len(t) - 1))
    first = int(np.searchsorted(t, rail_t, side="left"))
    window = np.arange(first, stop + 1)
    window = window[(speed[window] > _RAIL_MIN_SPEED) & ~np.isnan(cal[window])]
    defined = np.nonzero(~np.isnan(cal[first : stop + 1]))[0]
    keep = _thin(t, stop)
    numbers = {
        "apogee": _peak(alt[: stop + 1]),
        "apogeeT": float(apogee_t),
        "vmax": _peak(speed[: stop + 1]),
        "machMax": _peak(mach[: stop + 1]),
        "accMax": _peak(acc[: stop + 1]) / G0,
        "qMax": _peak(q[: stop + 1]),
        "railV": float(speed[min(first, stop)]),  # speed as the rail is left
        "marginRail": float(cal[first + defined[0]]) if len(defined) else float("nan"),
        "marginLo": float(cal[window].min()) if len(window) else float("nan"),
        "marginHi": float(cal[window].max()) if len(window) else float("nan"),
    }
    return {
        "m": {k: (None if np.isnan(v) else round(v, 4)) for k, v in numbers.items()},
        "t": _round(t[keep], 2),
        "alt": _round(alt[keep], 1),
        "v": _round(speed[keep], 1),
        "mach": _round(mach[keep], 3),
        "acc": _round(acc[keep] / G0, 2),
        "margin": _round(cal[keep], 2),
        "cd": _round(drag[keep], 3),
    }


def from_history(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The page's OpenRocket entry from one History flight (newest version), or None."""
    cols = entry["flight"]["cols"]
    events = entry["flight"]["events"]
    needed = ("time", "altitude", "velocity_total", "mach_number", "stability")
    if any(k not in cols for k in needed):
        return None
    t = _clean(cols["time"])
    alt = _clean(cols["altitude"])
    n = len(t)
    nothing = np.full(n, np.nan)

    def col(key: str) -> np.ndarray:
        return _clean(cols[key]) if key in cols else nothing

    apogee_t = (events.get("APOGEE") or [float(t[int(np.nanargmax(alt))])])[0]
    rail_t = (events.get("LAUNCHROD") or [0.0])[0]
    result = climb(
        t,
        alt,
        col("velocity_total"),
        col("mach_number"),
        col("acceleration_total"),
        col("stability"),
        col("drag_coeff"),
        col("dynamic_pressure"),
        apogee_t,
        rail_t,
    )
    history = entry["metrics"]
    # the numbers the History table shows win over the ones worked out here
    for key, name, factor in (
        ("apogee", "apogee", 1.0),
        ("machMax", "max_mach", 1.0),
        ("qMax", "max_dynamic_pressure_kpa", 1000.0),
        ("marginRail", "stability_off_rod_cal", 1.0),
        ("marginLo", "min_stability_cal", 1.0),
        ("marginHi", "max_stability_cal", 1.0),
    ):
        if history.get(name) is not None:
            result["m"][key] = round(history[name] * factor, 4)
    result["source"] = "history"
    result["label"] = (
        "OpenRocket results from the History tab "
        f"(design commit {entry['short']}, {entry['date']})"
    )
    return result


def from_saved(sim: SavedSim) -> dict[str, Any]:
    """The page's OpenRocket entry from the results saved in the OpenRocket file."""
    s = sim.series
    nothing = np.full_like(s["Time"], np.nan)
    result = climb(
        s["Time"],
        s["Altitude"],
        s["Total velocity"],
        s["Mach number"],
        s["Total acceleration"],
        s["Stability margin calibers"],
        s.get("Drag coefficient", nothing),
        0.5 * s.get("Air density", nothing) * s["Total velocity"] ** 2,
        float(sim.summary["timetoapogee"]),
        _rail_time(sim),
    )
    saved = sim.summary  # the numbers OpenRocket stored win over the ones worked out
    for key, name, factor in (
        ("apogee", "maxaltitude", 1.0),
        ("apogeeT", "timetoapogee", 1.0),
        ("vmax", "maxvelocity", 1.0),
        ("machMax", "maxmach", 1.0),
        ("accMax", "maxacceleration", 1.0 / G0),
        ("railV", "launchrodvelocity", 1.0),
    ):
        if name in saved:
            result["m"][key] = round(float(saved[name]) * factor, 4)
    result["source"] = "file"
    result["label"] = "OpenRocket results saved in the design file"
    return result


def _rail_time(sim: SavedSim) -> float:
    """When the saved run first reaches the speed it left the launch rod at."""
    s = sim.series
    reached = np.nonzero(s["Total velocity"] >= sim.summary["launchrodvelocity"])[0]
    return float(s["Time"][reached[0]]) if len(reached) else 0.0


def _read_flight(path: Path) -> list[dict[str, Any]]:
    """The versions in a ``flights/<id>.js`` file.

    The file is ``window.__flight(id, {versions: [...]});``.
    """
    match = _FLIGHT_JS.match(path.read_text(encoding="utf-8").strip())
    if match is None:
        raise ValueError(f"{path}: not a History flight file")
    body = match.group(1)
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(body)  # the simulation id comes first
    payload, _ = decoder.raw_decode(body[end:].lstrip(" ,"))
    return list(payload["versions"])


def load_history(site: Path, ork_name: str) -> dict[str, dict[str, Any]]:
    """OpenRocket flights of one design from a History site folder: {simulation: entry}.

    The design is found by its file name. An entry holds the newest good version's
    flight columns, events and the numbers of the History table. A missing or
    unreadable site gives an empty result, so the page falls back to the design file.
    """
    try:
        data = json.loads((site / "data.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys = [k for k in data.get("designs", {}) if Path(k).name == ork_name]
    if not keys:
        return {}
    design = keys[0]
    found: dict[str, dict[str, Any]] = {}
    for sim, rows in data["designs"][design].items():
        good = [r for r in rows if r.get("ok")]
        script = (data.get("flights") or {}).get(f"{design}|{sim}")
        if not good or not script:
            continue
        try:
            versions = _read_flight(site / script)
        except (OSError, ValueError, AttributeError, KeyError):
            continue
        if not versions:
            continue
        newest = good[-1]
        flight = versions[0]
        found[sim] = {
            "flight": flight,
            "metrics": newest["m"],
            "short": flight.get("short", newest["short"]),
            "date": flight.get("date", newest["date"]),
        }
    return found
