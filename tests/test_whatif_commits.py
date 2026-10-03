# ruff: noqa: E501
# pylint: disable=line-too-long
"""Tests for Jarvis flown on every commit of a design (the History tab's extra line)."""

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from ork_fixture import write_aero_csv, write_cdx, write_eng, write_ork

from flight_sim.whatif.commits import history_commits, jarvis_by_commit, main

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or shutil.which("git") is None,
    reason="node and git are needed",
)

_KEY = "aero_modeling/D.ork"
_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@t",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@t",
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, env=_ENV, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _widen_fins(ork: Path) -> None:
    """Edit the design: much taller fins, so it is more stable and drags more."""
    with zipfile.ZipFile(ork) as archive:
        text = archive.read("rocket.ork").decode("utf-8")
    with zipfile.ZipFile(ork, "w") as archive:
        archive.writestr(
            "rocket.ork", text.replace("<height>0.12</height>", "<height>0.2</height>")
        )


def _history(site: Path, shas: list[str]) -> None:
    rows = [
        {"short": s[:7], "sha": s, "t": 1_800_000_000 + i, "ok": True, "m": {}}
        for i, s in enumerate(shas)
    ]
    site.mkdir()
    (site / "data.json").write_text(json.dumps({"designs": {_KEY: {"calm": rows}}}))


@pytest.fixture(name="repo")
def fixture_repo(tmp_path: Path) -> dict[str, object]:
    """A git repository with the design committed twice (second time with bigger fins)."""
    folder = tmp_path / "aero_modeling"
    folder.mkdir()
    ork = write_ork(folder / "D.ork")
    aero = write_aero_csv(folder / "aero.csv")
    cdx = write_cdx(folder / "rasaero.CDX1")
    motor = write_eng(folder / "m.eng")
    _git(tmp_path, "init", "-q")
    first = _commit(tmp_path, "first")
    _widen_fins(ork)
    second = _commit(tmp_path, "bigger fins")
    site = tmp_path / "site"
    _history(site, [first, second])
    return {
        "root": tmp_path,
        "ork": ork,
        "aero": aero,
        "cdx": cdx,
        "motor": motor,
        "site": site,
        "shas": [first, second],
    }


def test_history_commits_finds_the_design_by_its_file_name(
    repo: dict[str, object],
) -> None:
    """The design is found in the History site by name, with its commits per simulation."""
    key, commits = history_commits(repo["site"], "D.ork")
    assert key == _KEY and commits == {"calm": repo["shas"]}
    assert history_commits(repo["site"], "other.ork") == ("", {})
    assert history_commits(repo["root"] / "nothing", "D.ork") == ("", {})


def test_every_commit_is_flown_with_its_own_design(repo: dict[str, object]) -> None:
    """Bigger fins in the second commit: more stable and less apogee, as the page would say."""
    result = jarvis_by_commit(
        repo["ork"], repo["aero"], repo["motor"], repo["site"], rasaero=repo["cdx"]
    )
    assert result["design"] == _KEY and result["skipped"] == 0
    first, second = (result["sims"]["calm"][sha] for sha in repo["shas"])
    assert set(first) == {
        "apogee",
        "max_mach",
        "max_dynamic_pressure_kpa",
        "stability_off_rod_cal",
        "min_stability_cal",
        "max_stability_cal",
        "stability_off_rod_pct",
        "min_stability_pct",
        "max_stability_pct",
    }
    assert first["apogee"] > 100.0 and first["max_mach"] > 0.1
    assert second["stability_off_rod_cal"] > first["stability_off_rod_cal"]
    assert second["apogee"] != first["apogee"]
    json.dumps(result)


def test_a_commit_that_cannot_be_read_is_skipped_and_counted(
    repo: dict[str, object],
) -> None:
    """A garbled design file in the history leaves a gap, not a failed build."""
    repo["ork"].write_bytes(b"not a zip")
    third = _commit(repo["root"], "broken")
    _history(repo["root"] / "site2", [*repo["shas"], third])
    result = jarvis_by_commit(
        repo["ork"],
        repo["aero"],
        repo["motor"],
        repo["root"] / "site2",
        rasaero=repo["cdx"],
    )
    assert result["skipped"] == 1 and set(result["sims"]["calm"]) == set(repo["shas"])


def test_a_design_the_history_does_not_have_gives_nothing(
    repo: dict[str, object],
) -> None:
    """No History for the design: nothing to draw."""
    empty = repo["root"] / "empty_site"
    empty.mkdir()
    (empty / "data.json").write_text(json.dumps({"designs": {}}))
    assert jarvis_by_commit(repo["ork"], repo["aero"], repo["motor"], empty) is None


def test_the_command_writes_the_json(
    repo: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m flight_sim.whatif.commits`` writes the file the History page loads."""
    out = repo["root"] / "by_commit.json"
    code = main(
        [
            "--ork", str(repo["ork"]), "--aero", str(repo["aero"]), "--motor", str(repo["motor"]),
            "--history-site", str(repo["site"]), "--rasaero", str(repo["cdx"]), "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0 and "2 points" in capsys.readouterr().out
    assert set(json.loads(out.read_text())["sims"]["calm"]) == set(repo["shas"])
    assert main(["--ork", "x.ork", "--aero", "a", "--motor", "m", "--history-site", str(repo["root"] / "nope"), "--out", str(out)]) == 1  # fmt: skip
