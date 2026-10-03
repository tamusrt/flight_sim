"""Jarvis's quick model on every committed version of a design, for the History tab.

The History tab draws OpenRocket's numbers for each commit of a design file. This
flies the same commits with Jarvis, so the tab can draw Jarvis next to OpenRocket.
Each commit is flown with:

* its own design file, read back from git;
* the RASAero table and ``.CDX1`` as they were at that commit (the nearest ones when
  there were none yet), so the quick model corrects for a changed rocket as it does
  on the page;
* today's motor, so the lines show the design changing and nothing else.

The flying is done by ``commits_runner.js``, which runs the very same ``engine.js`` the
predictions page runs in the browser.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flight_sim.motor_file import load_motor
from flight_sim.ork import load_ork, saved_motor
from flight_sim.ork_profile import motor_with_file
from flight_sim.whatif.build import (
    _base,
    _preset,
    read_rasaero_geometry,
    read_thrust,
    reduce_aero,
)

_RUNNER = Path(__file__).with_name("commits_runner.js")
_TIMEOUT_S = 600


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=False, timeout=120
    )


def _toplevel(folder: Path) -> Path | None:
    """The git repository a folder is in, or None."""
    try:
        done = _git(folder, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.decode("utf-8", "replace").strip()
    return Path(text) if done.returncode == 0 and text else None


def _show(repo: Path, sha: str, rel: str) -> bytes | None:
    """The file as it was at that commit, or None when it did not exist."""
    done = _git(repo, "show", f"{sha}:{rel}")
    return done.stdout if done.returncode == 0 else None


def _version_at(repo: Path, sha: str, rel: str) -> str | None:
    """The commit that last changed the file by ``sha``, else the first that had it."""
    last = _git(repo, "log", "-1", "--format=%H", sha, "--", rel)
    text = last.stdout.decode().strip()
    if last.returncode == 0 and text:
        return text
    first = _git(repo, "log", "--reverse", "--format=%H", "--", rel)
    lines = first.stdout.decode().split()
    return lines[0] if first.returncode == 0 and lines else None


def history_commits(site: Path, ork_name: str) -> tuple[str, dict[str, list[str]]]:
    """The design's key in the History site and, per simulation, the commits it has."""
    try:
        data = json.loads((site / "data.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", {}
    keys = [k for k in data.get("designs", {}) if Path(k).name == ork_name]
    if not keys:
        return "", {}
    design = data["designs"][keys[0]]
    return keys[0], {
        sim: [r["sha"] for r in rows if r.get("sha")] for sim, rows in design.items()
    }


class _Files:
    """The RASAero table and ``.CDX1`` of a commit, written out once each."""

    def __init__(
        self, repo: Path | None, tmp: Path, aero: Path, rasaero: Path | None
    ) -> None:
        self.repo, self.tmp, self.aero, self.rasaero = repo, tmp, aero, rasaero
        self._done: dict[tuple[str, str], Path | None] = {}

    def _rel(self, path: Path | None) -> str | None:
        if path is None or self.repo is None:
            return None
        try:
            return path.resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return None

    def _file(self, sha: str, path: Path | None) -> Path | None:
        rel = self._rel(path)
        version = (
            None
            if rel is None or self.repo is None
            else _version_at(self.repo, sha, rel)
        )
        if path is None:
            return None
        key = (version or "", path.name)
        if key not in self._done:
            blob = None
            if version and self.repo is not None and rel is not None:
                blob = _show(self.repo, version, rel)
            if blob is None:  # not in git: the file as it is now
                self._done[key] = path if path.is_file() else None
            else:
                target = self.tmp / f"{(version or '')[:10]}_{path.name}"
                target.write_bytes(blob)
                self._done[key] = target
        return self._done[key]

    def at(self, sha: str) -> tuple[Path | None, Path | None]:
        """The aero table and the ``.CDX1`` (None when there is none) for a commit."""
        return self._file(sha, self.aero), self._file(sha, self.rasaero)


def _run_node(job: dict[str, Any]) -> dict[str, Any]:
    done = subprocess.run(
        ["node", str(_RUNNER)],
        input=json.dumps(job).encode(),
        capture_output=True,
        check=True,
        timeout=_TIMEOUT_S,
    )
    return json.loads(done.stdout)  # type: ignore[no-any-return]


def jarvis_by_commit(  # pylint: disable=too-many-locals,too-many-arguments
    ork_path: Path,
    aero_csv: Path,
    motor_path: Path,
    history_site: Path,
    *,
    rasaero: Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any] | None:
    """Jarvis's six History numbers for every commit of the design, per simulation.

    Returns None when the History site does not have this design. Commits whose design
    file cannot be read (or has nothing to fly) are left out; ``skipped`` counts them.
    """
    key, wanted = history_commits(history_site, ork_path.name)
    if not key:
        return None
    repo = repo or _toplevel(ork_path.parent)
    motor_file = load_motor(motor_path)
    thrust = read_thrust(motor_path)
    shas = list(dict.fromkeys(s for shas in wanted.values() for s in shas))
    runs: list[dict[str, Any]] = []
    skipped = 0
    tables: dict[tuple[str, float], Any] = {}
    with tempfile.TemporaryDirectory(prefix="jarvis_commits_") as folder:
        tmp = Path(folder)
        files = _Files(repo, tmp, aero_csv, rasaero)
        for sha in shas:
            blob = _show(repo, sha, key) if repo is not None else None
            table, cdx = files.at(sha)
            if blob is None or table is None:
                skipped += 1
                continue
            target = tmp / f"{sha[:10]}.ork"
            target.write_bytes(blob)
            try:
                ork = load_ork(target)
                saved = saved_motor(ork, next(iter(ork.sims)))
                motor = motor_with_file(saved, motor_file)
                cache = (
                    table.name + str(table.stat().st_size),
                    ork.reference_diameter_m,
                )
                if cache not in tables:
                    tables[cache] = reduce_aero(table, ork.reference_diameter_m)
                ref = None
                if cdx is not None:
                    try:
                        ref = read_rasaero_geometry(cdx, ork)
                    except (ValueError, OSError):
                        ref = None
                conds = {
                    n: _preset(s)
                    for n, s in ork.sims.items()
                    if sha in wanted.get(n, [])
                }
                base = _base(ork, tables[cache], motor, thrust)
            except Exception:  # pylint: disable=broad-exception-caught
                skipped += 1  # an old design file this reader cannot make sense of
                continue
            if conds:
                runs.append({"id": sha, "base": base, "ref": ref, "conds": conds})
        flown = _run_node({"runs": runs}) if runs else {}
    sims: dict[str, dict[str, Any]] = {}
    for sha, per_sim in flown.items():
        if "error" in per_sim:
            skipped += 1
            continue
        for sim, values in per_sim.items():
            sims.setdefault(sim, {})[sha] = values
    return {
        "design": key,
        "motor": motor_path.name,
        "skipped": skipped,
        "note": (
            "Each point flies that version's design file with the RASAero table as it "
            f"was then and today's motor ({motor_path.name})"
        ),
        "sims": sims,
    }


def main(argv: list[str] | None = None) -> int:
    """Command line entry: write ``--out``, a JSON file, for the History design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ork", required=True)
    parser.add_argument("--aero", required=True)
    parser.add_argument("--motor", required=True, help="the .eng or .rse file")
    parser.add_argument("--history-site", required=True, help="the History site folder")
    parser.add_argument("--rasaero", default=None, help="the RASAero .CDX1")
    parser.add_argument("--out", required=True, help="the JSON file to write")
    args = parser.parse_args(argv)
    try:
        result = jarvis_by_commit(
            Path(args.ork),
            Path(args.aero),
            Path(args.motor),
            Path(args.history_site),
            rasaero=None if args.rasaero is None else Path(args.rasaero),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"Jarvis by commit skipped: {error}")
        return 1
    if result is None:
        print("Jarvis by commit skipped: the History site does not have this design")
        return 1
    Path(args.out).write_text(
        json.dumps(result, separators=(",", ":")), encoding="utf-8"
    )
    flown = sum(len(v) for v in result["sims"].values())
    print(f"Jarvis by commit: {flown} points, {result['skipped']} commits skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
