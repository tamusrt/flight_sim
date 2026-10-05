"""Run EDITH for one rocket of the dynamics site and add it to that rocket's pages.

    python -m flight_sim.whatif.edith_site --ork ... --aero ... --motor ... --out site/predictions

This runs after the JARVIS predictions page and VISION are built (``flight_sim.whatif.build``)
and adds to them:

* ``<out>/edith/index.html``: the EDITH page, with every chance and picture
  (EDITH's results are only on its own page, not on the JARVIS page);
* the cloud of simulated flights in the VISION page of the default launch condition,
  with a switch to show where the flights peaked or came down.

EDITH is the team's Monte Carlo simulation: it flies the rocket many times with the wind,
weather and motor strength varied a little each time, and counts what happens. It takes a
few minutes, so the result is kept (``--cache``) and reused until the design, the aero
table, the motor, the site settings or the simulator code EDITH runs change (a change to
a page or to VISION alone does not run it again). Whatever goes wrong here
leaves the JARVIS pages as they were: this program exits with an error and changes nothing.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import pandas  # type: ignore
import scipy  # type: ignore

import flight_sim
from flight_sim.edith import alerts
from flight_sim.edith.batch import Settings, run_batch
from flight_sim.edith.inputs import SiteConfig
from flight_sim.edith.run import RocketSpec
from flight_sim.ork import load_ork
from flight_sim.whatif import script_json

_HERE = Path(__file__).parent
_CLOUD_MARK = "<!--EDITH-->"
_CLOUD_END_MARK = "<!--/EDITH-->"
_ELLIPSE_POINTS = 72
# The landing circles VISION draws: the share of landings inside each
CIRCLE_SHARES = (0.25, 0.50, 0.75, 0.90)

# Where EDITH's flights start: the code they run is these modules and everything they import
_ENTRY = ("flight_sim.edith.batch", "flight_sim.edith.run", "flight_sim.fast_reco")


def _module_file(root: Path, name: str) -> Path | None:
    """The file of a ``flight_sim`` module (a package's ``__init__.py``), or None."""
    parts = name.split(".")[1:]
    path = root.joinpath(*parts)
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _imports(path: Path) -> set[str]:
    """Every ``flight_sim`` module a file imports, anywhere in it (also inside functions)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("flight_sim"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "flight_sim"
        ):
            found.add(node.module or "")
            found.update(f"{node.module}.{a.name}" for a in node.names)  # a submodule
    return found


def physics_files() -> list[Path]:
    """The simulator files EDITH's flights run: its entry modules and all they import."""
    root = Path(flight_sim.__file__).parent
    todo, seen = list(_ENTRY), set()
    files: set[Path] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_file(root, name)
        if path is None:
            continue
        files.add(path)
        todo.extend(_imports(path) - seen)
    return sorted(files)


def code_digest() -> str:
    """A short fingerprint of the simulator code EDITH runs, so changing it reruns EDITH.

    Only the modules EDITH's flights import count: changing the pages, VISION or anything
    else EDITH does not use keeps the kept result.
    """
    root = Path(flight_sim.__file__).parent
    digest = hashlib.sha256()
    for path in physics_files():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_key(args: argparse.Namespace, site: SiteConfig, settings: Settings) -> str:
    """What the result depends on: the design files, the settings and the code."""
    digest = hashlib.sha256()
    for name in ("ork", "aero", "motor"):
        digest.update(Path(getattr(args, name)).read_bytes())
    digest.update(str(args.sim).encode())
    digest.update(json.dumps(asdict(site), sort_keys=True).encode())
    keep = {
        k: v
        for k, v in asdict(settings).items()
        if k not in ("time_limit_s", "workers")
    }
    digest.update(json.dumps(keep, sort_keys=True).encode())
    digest.update(code_digest().encode())
    # The numbers also depend on the libraries that do the maths and on Python itself
    # (a new version of numpy or scipy can change a result in the last digits).
    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
    }
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return digest.hexdigest()[:20]


def _read_cache(cache: Path) -> dict[str, Any] | None:
    """The kept report, or None when the file cannot be used (it is then deleted).

    A file cut short (a build stopped while saving, or a half-copied cache) must not
    break every later build: it counts as a miss and the batch is run again.
    """
    try:
        report = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or "stopped_because" not in report:
            raise ValueError("not an EDITH report")
        return report
    except (OSError, ValueError) as error:
        print(
            f"EDITH: the kept result {cache.name} cannot be read ({error}); "
            "running EDITH again",
            flush=True,
        )
        cache.unlink(missing_ok=True)
        return None


def _write_cache(cache: Path, report: dict[str, Any]) -> None:
    """Keep the report as the only kept result, written in one step.

    The report is written to a temporary file in the same folder and then renamed to
    its name, so a reader (or a cache saved by the build system at that moment) sees
    either the whole file or none of it.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=cache.parent, prefix=".edith-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(report, file)
        for old in cache.parent.glob("edith-*.json"):
            old.unlink()
        os.replace(temporary, cache)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def ellipse(report: dict[str, Any]) -> list[list[float]]:
    """Points (east, north in metres) round the 90% landing ellipse; empty without one."""
    foot = report.get("footprint")
    if not foot:
        return []
    cx = foot["mean_east_m"]["estimate"]
    cy = foot["mean_north_m"]["estimate"]
    a = foot["ellipse90_semi_major_m"]
    b = foot["ellipse90_semi_minor_m"]
    bearing = math.radians(foot["ellipse90_major_axis_bearing_deg"])
    major = (math.sin(bearing), math.cos(bearing))  # (east, north)
    minor = (math.cos(bearing), -math.sin(bearing))
    points = []
    for i in range(_ELLIPSE_POINTS):
        t = 2.0 * math.pi * i / _ELLIPSE_POINTS
        u, v = a * math.cos(t), b * math.sin(t)
        points.append(
            [
                round(cx + u * major[0] + v * minor[0], 1),
                round(cy + u * major[1] + v * minor[1], 1),
            ]
        )
    return points


def page_data(
    report: dict[str, Any],
    name: str,
    sim: str,
    cached: bool,
    accepted: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Everything the EDITH page needs, as one JSON-able dict.

    ``accepted``: IREC recommendations the team has looked at and accepted (check ids,
    such as ``stability_static_max``); the page shows them greyed, marked accepted.
    """
    shown = {k: v for k, v in report.items() if k != "footprint"}
    foot = dict(report["footprint"] or {})
    foot.pop("points_east_north_m", None)  # the cloud has the landing points
    shown["footprint"] = foot or None
    return {
        "name": name,
        "sim": sim,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "cached": cached,
        "report": shown,
        "alerts": alerts.build(report),
        "ellipse": ellipse(report),
        "accepted": list(accepted),
    }


def write_page(data: dict[str, Any], out_dir: Path) -> Path:
    """Write ``edith/index.html`` with the data inlined."""
    template = (_HERE / "edith_page.html").read_text(encoding="utf-8")
    page = template.replace("/*DATA*/null", script_json(data))
    target = out_dir / "edith" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return target


def _viewer_files(out_dir: Path) -> list[Path]:
    """The VISION pages of the default launch condition (the flights EDITH is centred on)."""
    manifest = out_dir / "viewer" / "sims.json"
    files = [out_dir / "viewer" / "index.html"]
    if manifest.is_file():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        default = info.get("sims", {}).get(info.get("default", ""))
        if default:
            files.append(out_dir / "viewer" / default)
    return [f for f in dict.fromkeys(files) if f.is_file()]


def landing_circles(cloud: dict[str, Any]) -> dict[str, Any] | None:
    """Circles round the average landing point holding 25, 50, 75 and 90% of landings.

    Each radius is the distance from the average landing point that that share of
    the landings is inside (a percentile of the distances).
    """
    points = [p for p in cloud["landing"] if p]
    if len(points) < 4:
        return None
    east = sum(p[0] for p in points) / len(points)
    north = sum(p[1] for p in points) / len(points)
    distances = sorted(math.hypot(p[0] - east, p[1] - north) for p in points)

    def share(q: float) -> float:
        at = q * (len(distances) - 1)
        low = math.floor(at)
        high = min(low + 1, len(distances) - 1)
        return distances[low] + (distances[high] - distances[low]) * (at - low)

    return {
        "centre": [round(east, 1), round(north, 1)],
        "rings": [{"share": q, "radius_m": round(share(q), 1)} for q in CIRCLE_SHARES],
        "landings": len(points),
    }


def cloud_data(data: dict[str, Any]) -> dict[str, Any]:
    """The flights as VISION draws them."""
    report = data["report"]
    cloud, rng = report["cloud"], report["apogee_range"]
    return {
        "circles": landing_circles(cloud),
        "n": cloud["n"],
        "apogee": cloud["apogee"],
        "landing": cloud["landing"],
        "status": cloud["status"],
        "range": cloud["range"],
        "target_m": rng["target_m"],
        "low_m": rng["low_m"],
        "high_m": rng["high_m"],
        "ellipse": data["ellipse"],
    }


# A VISION page's cloud script, with or without the marker around it (pages patched by an earlier version have no marker)
_CLOUD_BLOCK = re.compile(
    r"(?:<!--EDITH-->)?<script>window\.EDITH_CLOUD=.*?</script>(?:<!--/EDITH-->)?"
    r"|<!--EDITH-->",
    re.DOTALL,
)


def patch_viewers(out_dir: Path, cloud: dict[str, Any]) -> int:
    """Give the VISION pages of the default launch condition the cloud of flights.

    Safe to run again: a page that already has a cloud gets the new one in
    its place, so no run leaves an old cloud behind. The markers stay in the
    page so the next run can find the cloud.
    """
    done = 0
    for path in _viewer_files(out_dir):
        text = path.read_text(encoding="utf-8")
        if not _CLOUD_BLOCK.search(text):
            continue
        # viewer/index.html and viewer/<name>/index.html are one folder apart
        link = (
            "../edith/index.html"
            if path.parent.name == "viewer"
            else "../../edith/index.html"
        )
        data = script_json({**cloud, "link": link})  # "</" and "<!--" are escaped inside
        block = (
            f"{_CLOUD_MARK}<script>window.EDITH_CLOUD={data};</script>{_CLOUD_END_MARK}"
        )
        # the new text is a re.sub template, so its backslashes are doubled
        patched = _CLOUD_BLOCK.sub(block.replace("\\", "\\\\"), text)
        path.write_text(patched, encoding="utf-8")
        done += 1
    return done


def run(args: argparse.Namespace) -> dict[str, Any]:
    """The batch's report: from the cache when nothing it depends on has changed."""
    site = SiteConfig.load(args.site) if args.site else SiteConfig()
    settings = Settings(
        round_size=args.round_size,
        max_rounds=args.max_rounds,
        time_limit_s=args.minutes * 60.0,
        workers=args.workers,
        surrogate=False,  # every climb in full: real paths and charts, about a second each
    )
    key = cache_key(args, site, settings)
    cache = Path(args.cache) / f"edith-{key}.json" if args.cache else None
    kept = _read_cache(cache) if cache is not None and cache.is_file() else None
    if cache is not None and kept is not None:
        print(
            f"EDITH: reusing the result kept for these inputs ({cache.name})",
            flush=True,
        )
        if args.name:  # the name is not part of the key: show the current one
            kept["rocket"] = args.name
        return {"report": kept, "cached": True}
    spec = RocketSpec("ork", args.ork, args.aero, args.motor, args.sim, args.name)
    started = time.perf_counter()
    report = run_batch(spec, site, settings)
    print(
        f"EDITH: {report['runs']} flights in {time.perf_counter() - started:.0f} s",
        flush=True,
    )
    # only a batch cut short by the time limit is not kept; one that flew all its
    # planned rounds is complete, however long it took
    finished = report.get("stopped_for") != "time_limit"
    if cache is not None and finished:
        _write_cache(cache, report)
    return {"report": report, "cached": False}


def main(argv: list[str] | None = None) -> None:
    """Command line entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ork", required=True)
    parser.add_argument("--aero", required=True)
    parser.add_argument("--motor", required=True)
    parser.add_argument("--sim", default=None)
    parser.add_argument("--name", default="SRT14")
    parser.add_argument("--out", required=True, help="the rocket's predictions folder")
    parser.add_argument(
        "--site", default=None, help="site settings (JSON); placeholders if left out"
    )
    parser.add_argument(
        "--cache", default=None, help="folder that keeps the result between builds"
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=10.0,
        help="stop starting new rounds after this long",
    )
    parser.add_argument("--round-size", type=int, default=64)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--accepted",
        default="",
        help="IREC recommendations the team accepts, as check ids separated by commas "
        "(shown greyed on the page; does not rerun EDITH)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not (out / "index.html").is_file():
        raise SystemExit(
            f"{out}: build the JARVIS predictions page first (flight_sim.whatif.build)"
        )
    # the launch condition the page calls the default; EDITH is centred on it
    sims = load_ork(args.ork).sims
    args.sim = args.sim if args.sim in sims else next(iter(sims))
    result = run(args)
    accepted = tuple(a.strip() for a in args.accepted.split(",") if a.strip())
    data = page_data(result["report"], args.name, args.sim, result["cached"], accepted)
    page = write_page(data, out)
    viewers = patch_viewers(out, cloud_data(data))
    print(
        f"EDITH page: {page} (VISION pages with the flights: {viewers})",
        flush=True,
    )


if __name__ == "__main__":
    main()
