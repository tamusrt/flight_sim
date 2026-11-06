"""EDITH parallel tests: flying run directories in flight_sim processes."""

from pathlib import Path

from edith.config import EdithConfig
from edith.parallel import CAMPAIGN_LOG, LOG, Status, fly, run
from edith.runner import prepare
from edith.sampler import sample


def _first_directory(campaign: Path) -> Path:
    """Prepare and return the campaign's first run directory only."""
    config = EdithConfig.from_file(campaign)
    return next(prepare(config, sample(config)))


def test_fly_reports_a_timeout(campaign: Path) -> None:
    """A trial that outlives the timeout is stopped and reported as TIMEOUT."""
    directory = _first_directory(campaign)

    result = fly(directory, 0.01)

    assert (result.status, result.returncode) == (Status.TIMEOUT, None)
    assert (directory / LOG).is_file()


def test_fly_reports_a_failed_run(tmp_path: Path) -> None:
    """A directory flight_sim cannot run is reported as FAILED, with its log."""
    directory = tmp_path / "empty"
    directory.mkdir()

    result = fly(directory, None)

    assert result.status is Status.FAILED
    assert result.returncode not in (0, None)
    assert "Cannot build" in (directory / LOG).read_text(encoding="utf-8")


def test_run_flies_every_directory_and_logs_each(campaign: Path) -> None:
    """Three trials on three workers all fly, write their results and get logged."""
    config = EdithConfig.from_file(campaign)
    directories = list(prepare(config, sample(config)))
    log = config.output_dir / CAMPAIGN_LOG

    results = run(directories, 3, None, log)

    assert [result.directory for result in results] == directories
    assert {result.status for result in results} == {Status.OK}
    for directory in directories:
        assert (directory / "run.toml").is_file()
        assert (directory / "plots").is_dir()
        assert "Wrote" in (directory / LOG).read_text(encoding="utf-8")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert sorted(line.split()[0] for line in lines) == ["0000", "0001", "0002"]
    assert all(" OK " in line for line in lines)
