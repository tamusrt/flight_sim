"""Fly prepared run directories in parallel, one flight_sim process per trial.

Each worker thread owns a fixed share of the directories and flies them one
at a time with ``flight_sim run``, which writes the trial's results and plots
into its directory; the process's output goes to ``LOG`` there. A process
that fails or outlives the timeout leaves the other trials untouched.
"""

import os
import subprocess
import sys
import threading
import time as clock
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tqdm import tqdm  # type: ignore[import-untyped]

# Output of flight_sim run, in each run directory
LOG = "flight_sim.log"

# One line per trial as it finishes, in the campaign's output directory
CAMPAIGN_LOG = "edith.log"


class Status(StrEnum):
    """How a trial ended."""

    OK = "OK"
    FAILED = "FAILED"  # flight_sim run exited with an error
    TIMEOUT = "TIMEOUT"  # Stopped at the trial timeout


@dataclass(frozen=True)
class Result:
    """One trial's outcome."""

    directory: Path
    status: Status
    seconds: float  # Wall time of the flight_sim process
    returncode: int | None  # None on timeout


def default_workers() -> int:
    """Return the worker count when the campaign names none: the CPUs less one."""
    return max(1, (os.cpu_count() or 2) - 1)


def run(
    directories: Sequence[Path], workers: int, timeout_s: float | None, log: Path
) -> list[Result]:
    """Fly run directories in parallel, showing a progress bar per worker.

    Worker ``w`` flies directories ``w``, ``w + workers``, and so on, so each
    bar counts the trials that worker has flown over those assigned to it; a
    last bar counts the campaign's.

    Args:
        directories (Sequence[Path]): Prepared run directories.
        workers (int): Worker threads, each flying one process at a time.
        timeout_s (float | None): Seconds a trial may run before it is
            stopped as TIMEOUT, or None for no limit.
        log (Path): Campaign log, appended with a line per trial as it ends.

    Returns:
        list[Result]: One per directory, in the directories' order.
    """
    workers = min(workers, len(directories)) or 1
    assignments = [directories[worker::workers] for worker in range(workers)]
    lock = threading.Lock()
    with (
        log.open("a", encoding="utf-8") as file,
        tqdm(
            total=len(directories),
            position=workers,
            desc="all",
            unit="trial",
            disable=None,
        ) as total,
    ):

        def report(result: Result) -> None:
            with lock:
                file.write(_log_line(result) + "\n")
                file.flush()
                total.update()

        with ThreadPoolExecutor(workers) as pool:
            futures = [
                pool.submit(_worker, number, assigned, timeout_s, report)
                for number, assigned in enumerate(assignments)
            ]
            by_worker = [future.result() for future in futures]
    results: list[Result | None] = [None] * len(directories)
    for worker, worker_results in enumerate(by_worker):
        results[worker::workers] = worker_results
    return [result for result in results if result is not None]


def fly(directory: Path, timeout_s: float | None) -> Result:
    """Fly one run directory in a flight_sim process, writing its output to LOG.

    Args:
        directory (Path): A prepared run directory.
        timeout_s (float | None): Seconds before the process is stopped, or
            None for no limit.

    Returns:
        Result: The trial's status, wall time and exit code.
    """
    command = [sys.executable, "-m", "flight_sim", "run", str(directory), "--no-show"]
    env = {**os.environ, "MPLBACKEND": "Agg"}  # Draw the plots without a window
    start = clock.perf_counter()
    with (directory / LOG).open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Result(directory, Status.TIMEOUT, clock.perf_counter() - start, None)
    status = Status.OK if completed.returncode == 0 else Status.FAILED
    return Result(directory, status, clock.perf_counter() - start, completed.returncode)


def _worker(
    number: int,
    directories: Sequence[Path],
    timeout_s: float | None,
    report: Callable[[Result], None],
) -> list[Result]:
    """Fly a worker's directories in turn, reporting each, behind its own bar."""
    results = []
    with tqdm(
        total=len(directories),
        position=number,
        desc=f"worker {number}",
        unit="trial",
        disable=None,
    ) as bar:
        for directory in directories:
            result = fly(directory, timeout_s)
            report(result)
            results.append(result)
            bar.update()
    return results


def _log_line(result: Result) -> str:
    """Return a trial's line of the campaign log."""
    line = f"{result.directory.name} {result.status} {result.seconds:.1f} s"
    if result.status is Status.FAILED:
        line += f", exit {result.returncode}; see {result.directory.name}/{LOG}"
    return line
