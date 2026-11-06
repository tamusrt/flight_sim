"""Fly prepared run directories in parallel, one persistent process per worker.

Each worker thread owns a fixed share of the directories and one
``edith.worker`` process, which imports flight_sim once and flies them one at
a time, writing each trial's results and plots into its directory and the
flight's output to ``edith.worker.LOG`` there. A trial that fails leaves the
others untouched; one that outlives the timeout has its process killed, and
the next trial gets a fresh one.
"""

import os
import queue
import subprocess
import sys
import threading
import time as clock
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO

from tqdm import tqdm  # type: ignore[import-untyped]

from edith.worker import READY

# One line per trial as it finishes, in the campaign's output directory
CAMPAIGN_LOG = "edith.log"

# Folder of the campaign's output directory taking each worker process's own
# output, which is empty unless the process itself fails
WORKER_LOGS = "workers"

# Environment of a worker process: plots without a window, and one thread per
# numeric library, since the workers are the parallelism
_WORKER_ENV = {
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class Status(StrEnum):
    """How a trial ended."""

    OK = "OK"
    FAILED = "FAILED"  # The flight raised an error, or its worker process died
    TIMEOUT = "TIMEOUT"  # Stopped at the trial timeout


@dataclass(frozen=True)
class Result:
    """One trial's outcome."""

    directory: Path
    status: Status
    seconds: float  # Wall time of the trial
    reason: str = ""  # Why it failed, for FAILED


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
        workers (int): Worker threads, each with one worker process.
        timeout_s (float | None): Seconds a trial may run before it is
            stopped as TIMEOUT, or None for no limit.
        log (Path): Campaign log, appended with a line per trial as it ends.
            The worker processes' own output goes under WORKER_LOGS beside it.

    Returns:
        list[Result]: One per directory, in the directories' order.
    """
    workers = min(workers, len(directories)) or 1
    assignments = [directories[worker::workers] for worker in range(workers)]
    worker_logs = log.parent / WORKER_LOGS
    worker_logs.mkdir(parents=True, exist_ok=True)
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
                pool.submit(
                    _fly_all,
                    number,
                    assigned,
                    timeout_s,
                    report,
                    worker_logs / f"{number}.log",
                )
                for number, assigned in enumerate(assignments)
            ]
            by_worker = [future.result() for future in futures]
    results: list[Result | None] = [None] * len(directories)
    for worker, worker_results in enumerate(by_worker):
        results[worker::workers] = worker_results
    return [result for result in results if result is not None]


def _fly_all(
    number: int,
    directories: Sequence[Path],
    timeout_s: float | None,
    report: Callable[[Result], None],
    worker_log: Path,
) -> list[Result]:
    """Fly a worker's directories in turn, reporting each, behind its own bar."""
    results = []
    worker = _Worker(worker_log)
    try:
        with tqdm(
            total=len(directories),
            position=number,
            desc=f"worker {number}",
            unit="trial",
            disable=None,
        ) as progress:
            for directory in directories:
                result = worker.fly(directory, timeout_s)
                report(result)
                results.append(result)
                progress.update()
    finally:
        worker.close()
    return results


class _Worker:
    """One ``edith.worker`` process, started when needed and replaced after it dies."""

    def __init__(self, log: Path) -> None:
        """Take the file for the process's own output."""
        self._log = log
        self._process: subprocess.Popen[str] | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()

    def fly(self, directory: Path, timeout_s: float | None) -> Result:
        """Fly a run directory in the process, starting one if there is none.

        Args:
            directory (Path): A prepared run directory.
            timeout_s (float | None): Seconds to wait for the flight, or None
                for no limit. The process is killed when it runs out.

        Returns:
            Result: The trial's status, wall time and reason if it failed.
        """
        if self._process is None:
            self._start()
        assert self._process is not None and self._process.stdin is not None
        start = clock.perf_counter()
        try:
            self._process.stdin.write(f"{directory}\n")
            self._process.stdin.flush()
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self._stop()
            return Result(directory, Status.TIMEOUT, clock.perf_counter() - start)
        except OSError:  # The process died, closing the pipe
            reply = None
        seconds = clock.perf_counter() - start
        if reply is None:
            code = self._stop()
            return Result(
                directory,
                Status.FAILED,
                seconds,
                f"the worker process exited with code {code}; see {self._log.name}",
            )
        status, _, reason = reply.partition(" ")
        if status == "OK":
            return Result(directory, Status.OK, seconds)
        return Result(directory, Status.FAILED, seconds, reason)

    def close(self) -> None:
        """End the process, if there is one, once it has read its input."""
        if self._process is None:
            return
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def _start(self) -> None:
        """Start the process and wait until it has imported flight_sim.

        Raises:
            RuntimeError: If the process ends before it is ready.
        """
        self._replies = queue.Queue()
        with self._log.open("a", encoding="utf-8") as log:
            self._process = subprocess.Popen(  # pylint: disable=consider-using-with
                [sys.executable, "-m", "edith.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env={**os.environ, **_WORKER_ENV},
            )
        threading.Thread(
            target=_read, args=(self._process.stdout, self._replies), daemon=True
        ).start()
        if self._replies.get() != READY:
            code = self._stop()
            raise RuntimeError(
                f"the worker process exited with code {code} before it was ready; "
                f"see {self._log}"
            )

    def _stop(self) -> int | None:
        """Kill the process, if there is one, and return its exit code."""
        process, self._process = self._process, None
        if process is None:
            return None
        process.kill()
        return process.wait()


def _read(stdout: IO[str], replies: "queue.Queue[str | None]") -> None:
    """Queue each line a worker process writes, then None when it ends."""
    for line in stdout:
        replies.put(line.rstrip("\r\n"))
    replies.put(None)


def _log_line(result: Result) -> str:
    """Return a trial's line of the campaign log."""
    line = f"{result.directory.name} {result.status} {result.seconds:.1f} s"
    if result.reason:
        line += f": {result.reason}"
    return line
