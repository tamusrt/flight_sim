"""EDITH parallel tests: flying run directories in persistent worker processes."""

import time
from itertools import islice
from pathlib import Path
from typing import Any, cast

import xarray as xr
import zarr

from edith.config import EdithConfig
from edith.parallel import CAMPAIGN_LOG, WORKER_LOGS, Status, Workers, run
from edith.runner import prepare
from edith.worker import LOG, fly_directory


def _prepared(campaign: Path, count: int) -> tuple[EdithConfig, list[Path]]:
    """Prepare and return the campaign's first ``count`` run directories."""
    config = EdithConfig.from_file(campaign)
    trials = islice(prepare(config), count)
    return config, [trial.folder for trial in trials]


def test_fly_directory_flies_a_prepared_run_and_logs_it(campaign: Path) -> None:
    """In this process, a prepared directory flies, writes its results and logs."""
    _, (directory,) = _prepared(campaign, 1)

    status = fly_directory(directory)

    assert status.startswith("OK ")
    assert (directory / "run.toml").is_file()
    assert (directory / "plots").is_dir()
    assert "Wrote" in (directory / LOG).read_text(encoding="utf-8")
    store = zarr.open_group(directory.parents[1] / "results.zarr")
    group = cast(Any, store[f"runs/{directory.name}"])
    telemetry = cast(Any, group["telemetry"])
    assert telemetry.shape[0] > 0
    assert cast(str, group.attrs["job_output"]).startswith("rail exit")
    dataset = xr.open_zarr(
        directory.parents[1] / "results.zarr" / "runs" / directory.name,
        consolidated=False,
    )
    assert dataset["telemetry"].dims == ("time", "channel")


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
    assert sorted(
        path.name
        for path in (config.output_dir / "results.zarr" / "runs").iterdir()
        if path.is_dir()
    ) == ["0000", "0001", "0002"]
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


def test_workers_start_at_once_and_say_when_they_are_ready(tmp_path: Path) -> None:
    """Started together, the processes come to answer READY while nothing flies."""
    with Workers(2, tmp_path / "workers") as workers:
        deadline = time.perf_counter() + 60.0
        while not workers.settled() and time.perf_counter() < deadline:
            time.sleep(0.1)

        assert (workers.count, workers.ready(), workers.settled()) == (2, 2, True)
