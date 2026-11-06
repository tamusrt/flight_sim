"""EDITH parallel tests: flying run directories in persistent worker processes."""

from itertools import islice
from pathlib import Path

from edith.config import EdithConfig
from edith.parallel import CAMPAIGN_LOG, WORKER_LOGS, Status, run
from edith.runner import prepare
from edith.sampler import sample
from edith.worker import LOG, fly_directory


def _prepared(campaign: Path, count: int) -> tuple[EdithConfig, list[Path]]:
    """Prepare and return the campaign's first ``count`` run directories."""
    config = EdithConfig.from_file(campaign)
    return config, list(islice(prepare(config, sample(config)), count))


def test_fly_directory_flies_a_prepared_run_and_logs_it(campaign: Path) -> None:
    """In this process, a prepared directory flies, writes its results and logs."""
    _, (directory,) = _prepared(campaign, 1)

    status = fly_directory(directory)

    assert status.startswith("OK ")
    assert (directory / "run.toml").is_file()
    assert (directory / "plots").is_dir()
    assert "Wrote" in (directory / LOG).read_text(encoding="utf-8")


def test_fly_directory_reports_a_failure_with_its_log(tmp_path: Path) -> None:
    """A directory with no input fails with the reason, and the traceback logged."""
    directory = tmp_path / "empty"
    directory.mkdir()

    status = fly_directory(directory)

    assert status.startswith("FAILED ConfigError: ")
    assert "resolved_input.toml" in status
    assert "Traceback" in (directory / LOG).read_text(encoding="utf-8")


def test_run_restarts_a_worker_after_a_timeout(campaign: Path) -> None:
    """A trial past the timeout is TIMEOUT, and the next trial still gets a worker."""
    config, directories = _prepared(campaign, 2)

    results = run(directories, 1, 0.05, config.output_dir / CAMPAIGN_LOG)

    assert [result.status for result in results] == [Status.TIMEOUT, Status.TIMEOUT]
    assert all(result.seconds < 5.0 for result in results)


def test_run_flies_every_directory_and_logs_each(campaign: Path) -> None:
    """Workers fly their shares, carry on past a failed trial, and log every one."""
    config, directories = _prepared(campaign, 3)
    empty = config.output_dir / "runs" / "empty"
    empty.mkdir()
    log = config.output_dir / CAMPAIGN_LOG

    results = run([empty, *directories], 3, None, log)

    assert [result.directory for result in results] == [empty, *directories]
    assert [result.status for result in results] == [Status.FAILED, *[Status.OK] * 3]
    assert results[0].reason.startswith("ConfigError: ")
    for directory in directories:
        assert (directory / "run.toml").is_file()
        assert (directory / "plots").is_dir()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert sorted(line.split()[0] for line in lines) == [
        "0000",
        "0001",
        "0002",
        "empty",
    ]
    assert sum(" OK " in line for line in lines) == 3
    assert any(
        line.startswith("empty FAILED") and "ConfigError" in line for line in lines
    )
    assert sorted(
        path.name for path in (config.output_dir / WORKER_LOGS).iterdir()
    ) == [
        "0.log",
        "1.log",
        "2.log",
    ]
