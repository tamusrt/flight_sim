"""Write complete EDITH flight results to a shared Zarr store.

Each trial's group metadata, ``runs/<trial>/zarr.json``, is written whole at
once: Windows refuses to replace a file that another handle, such as a virus
scanner's, still holds just after it was written, which several writes in a
row of one file run into. The writes that remain are retried briefly when it
does.
"""

import time as clock
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

if TYPE_CHECKING:
    from flight_sim.simulation import Flight


STORE = "results.zarr"


def initialize(path: Path) -> None:
    """Create the campaign's shared Zarr store before workers start writing."""
    root = zarr.open_group(path, mode="w")
    root.attrs.update(
        {
            "format": "edith",
            "description": "One runs/<trial> group per EDITH simulation.",
        }
    )
    root.create_group("runs")


# Tries of a write Windows refuses, and the wait after the first, doubling
WRITE_TRIES = 6
FIRST_WAIT_S = 0.05


def write(
    flight: "Flight",
    path: Path,
    trial: str,
) -> None:
    """Write one complete flight result into its own group, telemetry one chunk.

    The group is written with all its attributes at once, then its array.
    """
    frame = flight.to_dataframe()
    columns = frame.drop(columns="event")
    attributes = {
        "columns": list(columns.columns),
        "events": {name: int(row) for name, row in flight.events.items()},
        "event_labels": frame["event"].tolist(),
        "name": flight.simulation.name,
        "units": flight.units,
        "summary": flight.summary(),
        "diagnostics": flight.diagnostics(),
    }
    _retried(
        partial(
            _write_group,
            path,
            trial,
            attributes,
            columns.to_numpy(dtype=np.float64),
        )
    )


def write_job_output(path: Path, trial: str, text: str) -> None:
    """Add the captured console output to an already-written trial group."""
    group = zarr.open_group(path, path=f"runs/{trial}", mode="r+")
    _retried(partial(group.update_attributes, {"job_output": text}))


def _write_group(
    path: Path, trial: str, attributes: dict[str, object], values: np.ndarray
) -> None:
    """Write a trial's group afresh, replacing any it had.

    The store and its runs group are made if ``initialize`` has not made them.
    """
    runs = zarr.open_group(path, mode="a").require_group("runs")
    if trial in runs:
        del runs[trial]
    group = runs.create_group(trial, attributes=attributes)
    group.create_array(
        "telemetry",
        data=values,
        chunks=values.shape,
        dimension_names=("time", "channel"),
    )


def _retried(attempt: Callable[[], object]) -> None:
    """Run a write, trying again after a wait if Windows refuses it.

    Raises:
        PermissionError: If every one of ``WRITE_TRIES`` tries is refused.
    """
    wait = FIRST_WAIT_S
    for _ in range(WRITE_TRIES - 1):
        try:
            attempt()
            return
        except PermissionError:
            clock.sleep(wait)
            wait *= 2.0
    attempt()
