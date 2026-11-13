"""Tests of reading a finished campaign's results back into one table."""

import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import zarr

from edith.statistics.results import (
    event_column,
    failure_counts,
    input_keys,
    load_table,
    read_channel,
    read_event,
    read_runs,
    read_summary,
)
from edith.zarr_output import initialize

_COLUMNS = ["time [s]", "state.position_x [m]", "derived.speed [m/s]"]
_SPEED = "derived.speed [m/s]"


def _store_trial(
    store: Path, trial: str, apogee: float, events: dict[str, int]
) -> None:
    """Store one trial's flight as ``zarr_output.write`` lays it out."""
    times = np.arange(5.0)
    telemetry = np.column_stack([times, apogee * np.sin(times / 4.0), 10.0 * times])
    group = zarr.open_group(store, mode="a").create_group(f"runs/{trial}")
    group.create_array("telemetry", data=telemetry, chunks=telemetry.shape)
    group.attrs.update(
        {
            "columns": _COLUMNS,
            "events": events,
            "summary": {"apogee_m": apogee, "flight_time_s": 4.0},
            "diagnostics": {"run_time_s": 2.0, "rejected_steps": 3},
        }
    )


@pytest.fixture(name="campaign")
def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Return a campaign of three trials: two flown, and one failed unstored."""
    store = tmp_path / "results.zarr"
    initialize(store)
    _store_trial(store, "0000", 3000.0, {"rail exit": 1, "impact": 4})
    _store_trial(store, "0001", 3100.0, {"impact": 4})  # Missed the rail exit
    trials = tmp_path / "trials.csv"
    pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "batch": [0, 0, 1],
            "mass": [17.0, 17.2, 16.9],
            "wind": [1.0, 2.0, 3.0],
            "apogee": [3000.0, 3100.0, math.nan],
            "status": ["OK", "OK", "TIMEOUT"],
        }
    ).to_csv(trials, index=False)
    return trials, store


def test_the_table_joins_inputs_with_what_each_flight_stored(
    campaign: tuple[Path, Path],
) -> None:
    """Each trial's row has its inputs, summary and diagnostics; failed is set."""
    table = load_table(*campaign)

    assert list(table["trial"]) == [0, 1, 2]
    assert list(table["apogee_m"][:2]) == [3000.0, 3100.0]
    assert math.isnan(table["apogee_m"][2])  # Never stored
    assert list(table["run_time_s"][:2]) == [2.0, 2.0]
    assert list(table["failed"]) == [False, False, True]
    assert input_keys(table) == ["mass", "wind"]


def test_event_values_join_as_columns(campaign: tuple[Path, Path]) -> None:
    """A channel at an event is a column, NaN where a trial did not hit it."""
    table = load_table(*campaign, events=[("rail exit", _SPEED)])
    column = table[event_column("rail exit", _SPEED)]

    assert column[0] == 10.0  # Row 1 of a speed of 10 m/s per second
    assert math.isnan(column[1])
    assert math.isnan(column[2])


def test_one_trials_results_can_be_read(campaign: tuple[Path, Path]) -> None:
    """Its summary, a channel and a value at an event read back."""
    _, store = campaign

    assert read_summary(store, 1)["apogee_m"] == 3100.0
    assert read_channel(store, 0, _SPEED).tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert read_event(store, 0, "impact", _SPEED) == 40.0
    assert math.isnan(read_event(store, 1, "rail exit", _SPEED))
    (times, speeds), _ = read_runs(store, _SPEED, [0, 1])
    assert times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert speeds[-1] == 40.0


def test_missing_results_and_channels_are_named(campaign: tuple[Path, Path]) -> None:
    """A trial with no stored flight, or a channel no flight has, is a KeyError."""
    _, store = campaign

    with pytest.raises(KeyError, match="trial 2 has no results"):
        read_summary(store, 2)
    with pytest.raises(KeyError, match="'thrust' is not a stored channel"):
        read_channel(store, 0, "thrust")


def test_failures_are_counted_by_status(campaign: tuple[Path, Path]) -> None:
    """Only failed trials are counted."""
    counts = failure_counts(load_table(*campaign))

    assert counts.to_dict() == {"TIMEOUT": 1}


def test_a_partly_written_trial_reads_as_missing(campaign: tuple[Path, Path]) -> None:
    """A group a failed write left with only its telemetry has NaN values."""
    trials, store = campaign
    group = zarr.open_group(store, mode="a").create_group("runs/0002")
    group.create_array("telemetry", data=np.zeros((2, 3)), chunks=(2, 3))
    group.attrs["columns"] = _COLUMNS

    table = load_table(trials, store, events=[("impact", _SPEED)])

    assert math.isnan(table["apogee_m"][2])
    assert math.isnan(table[event_column("impact", _SPEED)][2])
