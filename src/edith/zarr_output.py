"""Write complete EDITH flight results to a shared Zarr store."""

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


def write(
    flight: "Flight",
    path: Path,
    trial: str,
) -> None:
    """Write one complete flight result into its own group, telemetry one chunk."""
    root = zarr.open_group(path, mode="a")
    group_path = f"runs/{trial}"
    if group_path in root:
        del root[group_path]
    group = root.create_group(group_path)

    frame = flight.to_dataframe()
    values = frame.drop(columns="event").to_numpy(dtype=np.float64)
    group.create_array(
        "telemetry",
        data=values,
        chunks=values.shape,
        dimension_names=("time", "channel"),
    )
    group.attrs.update(
        {
            "columns": list(frame.drop(columns="event").columns),
            "events": {name: int(row) for name, row in flight.events.items()},
            "event_labels": frame["event"].tolist(),
            "name": flight.simulation.name,
            "units": flight.units,
            "summary": flight.summary(),
            "diagnostics": flight.diagnostics(),
        }
    )


def write_job_output(path: Path, trial: str, text: str) -> None:
    """Add the captured console output to an already-written trial group."""
    root = zarr.open_group(path, mode="a")
    root[f"runs/{trial}"].attrs["job_output"] = text
