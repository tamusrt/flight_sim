"""Read a finished campaign's results: ``trials.csv`` and ``results.zarr``.

The runner and ``edith.zarr_output`` write them; this module only reads, from
their paths, so post-processing starts from one table. ``load_table`` joins
each trial's row of ``trials.csv`` (its batch, drawn inputs, apogee and
status) with the summary and diagnostics its flight stored, by trial number:
trial 7 is the store's group ``runs/0007``. A trial that failed before its
flight was stored, or while storing it, has NaN where values are missing.
Channels are named as the store's ``columns`` attribute names them, such as
"derived.speed [m/s]".
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import zarr
from numpy.typing import NDArray

__all__ = [
    "OK",
    "TIME",
    "event_column",
    "failure_counts",
    "input_keys",
    "load_table",
    "read_channel",
    "read_event",
    "read_runs",
    "read_summary",
]

# Status of a trial that flew and recorded an apogee, as trials.csv writes it
OK = "OK"

# The time channel every trial stores
TIME = "time [s]"

Run = tuple[NDArray[np.float64], NDArray[np.float64]]


def read_summary(store: Path, trial: int) -> dict[str, float]:
    """Return a trial's headline numbers, such as ``apogee_m``, keyed with units."""
    return {
        name: float(value) for name, value in _attrs(_group(store, trial), "summary")
    }


def read_channel(store: Path, trial: int, channel: str) -> NDArray[np.float64]:
    """Return a trial's channel at every stored row.

    Raises:
        KeyError: If the trial has no results, or no channel of that name.
    """
    group = _group(store, trial)
    return np.asarray(_telemetry(group)[:, _column(group, channel)], dtype=float)


def read_event(store: Path, trial: int, event: str, channel: str) -> float:
    """Return a trial's channel at a flight event, such as "rail exit".

    Returns:
        float: The value on the event's row, or NaN if the trial did not hit it.

    Raises:
        KeyError: If the trial has no results, or no channel of that name.
    """
    return _event_value(_group(store, trial), event, channel)


def read_runs(store: Path, channel: str, trials: Sequence[int]) -> list[Run]:
    """Return each trial's times and channel, as ``Envelope.from_runs`` takes them.

    Raises:
        KeyError: If a trial has no results, or no channel of that name.
    """
    groups = _groups(store)
    runs = []
    for trial in trials:
        group = _found(groups, trial, store)
        telemetry = _telemetry(group)
        runs.append(
            (
                np.asarray(telemetry[:, _column(group, TIME)], dtype=float),
                np.asarray(telemetry[:, _column(group, channel)], dtype=float),
            )
        )
    return runs


def event_column(event: str, channel: str) -> str:
    """Return the column ``load_table`` gives a channel at an event."""
    return f"{channel} at {event}"


def load_table(
    trials_path: Path, store: Path, events: Sequence[tuple[str, str]] = ()
) -> pd.DataFrame:
    """Return every trial's inputs, status, summary and diagnostics in one table.

    Args:
        trials_path (Path): The campaign's ``trials.csv``.
        store (Path): The campaign's ``results.zarr``.
        events (Sequence[tuple[str, str]], optional): (event, channel) pairs,
            such as ("rail exit", "derived.speed [m/s]"), each adding a column
            named by ``event_column``. Defaults to none.

    Returns:
        pd.DataFrame: One row per trial, in the order of ``trials.csv``: its
            columns, then each stored summary and diagnostic, the event
            columns, and ``failed``, whether the status is not ``OK``.

    Raises:
        KeyError: If a stored trial has no channel an event names.
    """
    table = pd.read_csv(trials_path)
    groups = _groups(store)
    rows = []
    for trial in table["trial"]:
        group = groups.get(int(trial))
        row: dict[str, Any] = {}
        if group is not None:
            row.update(_attrs(group, "summary"))
            row.update(_attrs(group, "diagnostics"))
        for event, channel in events:
            row[event_column(event, channel)] = (
                math.nan if group is None else _event_value(group, event, channel)
            )
        rows.append(row)
    table = pd.concat([table, pd.DataFrame(rows, index=table.index)], axis=1)
    table["failed"] = table["status"] != OK
    return table


def input_keys(table: pd.DataFrame) -> list[str]:
    """Return the key of each dispersion, as the columns of the drawn inputs.

    The runner writes them between ``batch`` and ``apogee``.
    """
    columns = list(table.columns)
    return columns[columns.index("batch") + 1 : columns.index("apogee")]


def failure_counts(table: pd.DataFrame) -> pd.Series:
    """Return how many trials failed with each status, most common first."""
    return table.loc[table["failed"], "status"].value_counts()


def _groups(store: Path) -> dict[int, zarr.Group]:
    """Return each stored trial's group by trial number."""
    runs = zarr.open_group(store, path="runs", mode="r")
    return {int(name): group for name, group in runs.groups()}


def _group(store: Path, trial: int) -> zarr.Group:
    """Return a trial's group.

    Raises:
        KeyError: If the store has none.
    """
    return _found(_groups(store), trial, store)


def _found(groups: dict[int, zarr.Group], trial: int, store: Path) -> zarr.Group:
    """Return a trial's group from those of a store.

    Raises:
        KeyError: If there is none.
    """
    if trial not in groups:
        raise KeyError(f"trial {trial} has no results in {store}")
    return groups[trial]


def _attrs(group: zarr.Group, name: str) -> list[tuple[str, Any]]:
    """Return the entries of a table a trial's group holds as an attribute.

    A group a failed write left without it has none.
    """
    table = group.attrs.get(name, {})
    assert isinstance(table, dict)
    return list(table.items())


def _telemetry(group: zarr.Group) -> zarr.Array[Any]:
    """Return a trial's telemetry array, one row per time and a column per channel."""
    telemetry = group["telemetry"]
    assert isinstance(telemetry, zarr.Array)
    return telemetry


def _column(group: zarr.Group, channel: str) -> int:
    """Return the column of a channel in a trial's telemetry.

    Raises:
        KeyError: If the trial stores no channel of that name.
    """
    columns = group.attrs["columns"]
    assert isinstance(columns, list)
    if channel not in columns:
        raise KeyError(f"{channel!r} is not a stored channel, such as {TIME!r}")
    return columns.index(channel)


def _event_value(group: zarr.Group, event: str, channel: str) -> float:
    """Return a channel on an event's row of a trial, NaN if it did not hit it."""
    rows = group.attrs.get("events", {})
    assert isinstance(rows, dict)
    column = _column(group, channel)
    if event not in rows:
        return math.nan
    return float(np.asarray(_telemetry(group)[int(rows[event]), column]))
