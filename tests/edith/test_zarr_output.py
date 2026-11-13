"""Tests of writing flight results to the campaign's Zarr store."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import zarr
import zarr.storage._local as local_store

from edith import zarr_output
from edith.zarr_output import initialize, write, write_job_output


def _flight() -> Any:
    """Return a stand-in for a Flight, with what ``write`` reads of one."""
    frame = pd.DataFrame(
        {"time [s]": [0.0, 1.0], "thrust [N]": [5.0, 0.0], "event": ["", "impact"]}
    )
    return SimpleNamespace(
        to_dataframe=lambda: frame,
        events={"impact": 1},
        simulation=SimpleNamespace(name="test"),
        units="metric",
        summary=lambda: {"apogee_m": 10.0},
        diagnostics=lambda: {"run_time_s": 0.5},
    )


def test_a_trials_metadata_is_written_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its zarr.json is replaced once, not once per attribute, and holds them all."""
    store = tmp_path / "results.zarr"
    initialize(store)
    replaced: list[str] = []
    put = local_store._put  # pylint: disable=protected-access

    def spy(path: Path, value: Any, exclusive: bool = False) -> int:
        if not exclusive:
            replaced.append(path.relative_to(store).as_posix())
        return put(path, value, exclusive=exclusive)

    monkeypatch.setattr(local_store, "_put", spy)
    write(_flight(), store, "0000")
    write_job_output(store, "0000", "log text")

    assert replaced.count("runs/0000/zarr.json") == 2  # The group, then its log
    attributes = zarr.open_group(store, path="runs/0000", mode="r").attrs
    assert attributes["summary"] == {"apogee_m": 10.0}
    assert attributes["events"] == {"impact": 1}
    assert attributes["job_output"] == "log text"


def test_writing_a_trial_again_replaces_it(tmp_path: Path) -> None:
    """A trial flown again has only its new results."""
    store = tmp_path / "results.zarr"
    initialize(store)
    write(_flight(), store, "0000")
    write_job_output(store, "0000", "first")

    write(_flight(), store, "0000")

    assert "job_output" not in zarr.open_group(store, path="runs/0000").attrs


def test_a_refused_write_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows refusing a replace twice is waited out; refusing every try fails."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    tries: list[int] = []

    def flaky() -> None:
        tries.append(1)
        if len(tries) < 3:
            raise PermissionError("Access is denied")

    zarr_output._retried(flaky)  # pylint: disable=protected-access
    assert len(tries) == 3

    def refused() -> None:
        raise PermissionError("Access is denied")

    with pytest.raises(PermissionError):
        zarr_output._retried(refused)  # pylint: disable=protected-access


def test_a_trial_can_be_written_before_the_store_is_made(tmp_path: Path) -> None:
    """Without initialize, as when fly_campaign is used directly, it is made."""
    store = tmp_path / "results.zarr"

    write(_flight(), store, "0003")

    assert zarr.open_group(store, path="runs/0003").attrs["events"] == {"impact": 1}
