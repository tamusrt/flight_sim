"""Persistent worker: fly the run directories read from stdin, importing once.

``edith.parallel`` starts one of these per worker and writes it a run directory
per line. The worker flies each in this process, as ``flight_sim run DIR
--no-show`` would, with that command's output in the directory's LOG, and
answers one line per directory: "OK <seconds>" or "FAILED <reason>". It
answers READY once flight_sim is imported, before the first directory.
"""

import sys
import time as clock
import traceback
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path

from edith import zarr_output
from edith.zarr_output import STORE
from flight_sim import output
from flight_sim.__main__ import fly
from flight_sim.simulation import Simulation

# Output of the flight in each run directory, as flight_sim run would print it
LOG = "flight_sim.log"

# First line the worker answers, once it is ready for a directory
READY = "READY"


def main() -> None:
    """Fly each run directory named on stdin, answering a status line for each."""
    print(READY, flush=True)
    for line in sys.stdin:
        directory = line.strip()
        if directory:
            print(fly_directory(Path(directory)), flush=True)


def fly_directory(directory: Path) -> str:
    """Fly a run directory, writing the flight's output to its LOG.

    Args:
        directory (Path): A prepared run directory.

    Returns:
        str: "OK <seconds>", or "FAILED <reason>" with the error's first line;
            the traceback is in the log.
    """
    start = clock.perf_counter()
    try:
        with (
            (directory / LOG).open("w", encoding="utf-8") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            try:
                simulation = Simulation.from_file(directory / output.RESOLVED_INPUT)
                fly(
                    simulation,
                    Namespace(no_show=True, profile=False),
                    additional_writer=partial(
                        zarr_output.write,
                        path=directory.parents[1] / STORE,
                        trial=directory.name,
                    ),
                )
            except Exception:
                traceback.print_exc()
                raise
        zarr_output.write_job_output(
            directory.parents[1] / STORE,
            directory.name,
            (directory / LOG).read_text(encoding="utf-8"),
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Any failure of one trial, which must not stop the next
        reason = f"{type(error).__name__}: {error}".splitlines()[0]
        return f"FAILED {reason}"
    return f"OK {clock.perf_counter() - start:.1f}"


if __name__ == "__main__":
    main()
