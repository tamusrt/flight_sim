"""Prepare a campaign's run directories and fly them batch by batch.

Each trial's directory is a flight_sim run directory: ``resolved_input.toml``
with the trial's drawn values and the campaign's output settings in place of
the base sim's, and copies of the data files, so ``flight_sim run`` flies it
on its own. The trials come in batches, each an independently seeded
replicate drawn by ``edith.statistics``. A batch is prepared just before it
flies, and the campaign goes on until the stopping rule is met or every trial
has flown, so an early stop leaves only the trials that flew. Each batch's
apogees feed the estimate, and every trial's drawn values, status and apogee
go to ``trials.csv``. ``write_provenance`` records what regenerates the
campaign: the code's commit, the campaign file and the package versions.
"""

import json
import math
import subprocess
import time as clock
import tomllib
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from edith.config import EdithConfig, output_values
from edith.dispersion import StandardDispersion
from edith.parallel import CAMPAIGN_LOG, Result, Status, Workers
from edith.statistics.estimators import BatchStats
from edith.statistics.generators import GENERATORS, stream
from edith.statistics.stopping import (
    ConfidenceIntervalTarget,
    FixedTrials,
    StandardErrorTarget,
    StoppingRule,
)
from edith.targets import node_at
from flight_sim import output
from flight_sim.config.toml_files import display_path

__all__ = [
    "PROVENANCE",
    "RUNS",
    "TRIALS",
    "CampaignResult",
    "Trial",
    "batches",
    "draw_batch",
    "fly_campaign",
    "prepare",
    "prepare_batches",
    "run_directory",
    "write_provenance",
]

# Folder of a campaign's output directory holding a run directory per trial
RUNS = "runs"

# Every trial's drawn values, status and apogee, in the output directory
TRIALS = "trials.csv"

# What regenerates the campaign, in the output directory
PROVENANCE = "provenance.json"

# Packages whose versions the provenance records
_PACKAGES = ("flight_sim", "numpy", "scipy", "pandas", "pint", "zarr")

Values = dict[str, np.ndarray[Any, np.dtype[np.float64]]]


@dataclass
class Trial:
    """One prepared trial: its run directory and the numbers it drew."""

    folder: Path
    drawn: dict[str, float] = field(default_factory=dict)  # By dispersion key


def run_directory(config: EdithConfig, index: int) -> Path:
    """Return the run directory of a trial, such as ``<output_dir>/runs/0007``."""
    width = max(4, len(str(config.num_sims - 1)))
    return config.output_dir / RUNS / f"{index:0{width}d}"


def batches(config: EdithConfig) -> list[slice]:
    """Return the trial indices of each batch, in order.

    There are ``num_batches`` of them, each of ``flight_per_batch`` trials.
    """
    size = config.flight_per_batch
    return [slice(start, start + size) for start in range(0, config.num_sims, size)]


def prepare_batches(
    config: EdithConfig, *, overwrite: bool = False
) -> Generator[list[Trial], None, None]:
    """Create the run directories of one batch at a time, holding dispersed inputs.

    Close the iterator when stopping early, to remove the data files it caches
    under the runs folder.

    Args:
        config (EdithConfig): The campaign.
        overwrite (bool, optional): Write into a runs folder that already
            exists. Defaults to refusing.

    Yields:
        list[Trial]: Each batch's trials, in order, once their directories are
            written.

    Raises:
        FileExistsError: If the runs folder exists and ``overwrite`` is not set.
    """
    runs = config.output_dir / RUNS
    if runs.exists() and not overwrite:
        raise FileExistsError(f"{display_path(runs)} already exists")
    runs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parquet-", dir=runs) as cache:
        source = (
            output.parquet_source(config.source, Path(cache))
            if config.input_mode == "compressed"
            else config.source
        )
        for batch, span in enumerate(batches(config)):
            drawn = draw_batch(config, batch, span.stop - span.start)
            trials = []
            for offset, index in enumerate(range(span.start, span.stop)):
                trial = Trial(
                    run_directory(config, index),
                    {key: float(values[offset]) for key, values in drawn.items()},
                )
                _write(config, source, trial)
                trials.append(trial)
            yield trials


def prepare(config: EdithConfig, *, overwrite: bool = False) -> Iterator[Trial]:
    """Create a run directory per trial, yielding each trial as it is written.

    Args:
        config (EdithConfig): The campaign.
        overwrite (bool, optional): Write into a runs folder that already
            exists. Defaults to refusing.

    Yields:
        Trial: Each trial, in order.

    Raises:
        FileExistsError: If the runs folder exists and ``overwrite`` is not set.
    """
    for trials in prepare_batches(config, overwrite=overwrite):
        yield from trials


def _write(config: EdithConfig, source: Any, trial: Trial) -> None:
    """Write a trial's run directory from the base sim's resolved input."""
    trial.folder.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        d.key: _as_input(d, trial.drawn[d.key], config.source.table)
        for d in _dispersions(config)
    }
    values.update(output_values(config.output))
    values["output.directory"] = trial.folder.as_posix()
    if config.input_mode == "lightweight":
        lightweight_values = dict(values)
        lightweight_values["output.directory"] = "."
        output.prepare_lightweight(config.source, lightweight_values, trial.folder)
    else:
        output.prepare_source(source, values, trial.folder, overwrite=True)


@dataclass
class CampaignResult:
    """What a campaign found: the apogee estimate, in metres, and each flight."""

    stats: BatchStats
    stopped_by: str  # "rule" or "all_batches"
    confidence: float | None
    trials_path: Path
    results: list[Result]  # Each flown trial's outcome, in trial order
    prepare_s: float  # Wall time preparing the batches that flew

    def summary(self) -> str:
        """Return the estimate as a line, saying what it still needs if it has no CI."""
        batch_count = self.stats.batches
        counts = f"{self.stats.trials} trials in {batch_count} batch"
        counts += "es" if batch_count != 1 else ""
        if self.confidence is None:
            return f"apogee {self.stats.mean:.6g} m ({counts})"
        half_width = self.stats.confidence_interval_half_width(self.confidence)
        if half_width is None:
            return (
                f"apogee {self.stats.mean:.6g} m ({counts}). {self.stats.requirement}"
            )
        return (
            f"apogee {self.stats.mean:.6g} +/- {half_width:.3g} m "
            f"({self.confidence:.0%} CI, {counts})"
        )


def fly_campaign(
    config: EdithConfig,
    workers: Workers,
    prepared: Iterable[Sequence[Trial]],
    on_prepared: Callable[[int, Sequence[Trial]], None] | None = None,
    on_batch: Callable[[int, BatchStats], None] | None = None,
) -> CampaignResult:
    """Fly the campaign batch by batch until the stopping rule is met.

    Args:
        config (EdithConfig): The campaign, for its timeout, stopping rule and
            output directory.
        workers (Workers): The worker processes that fly each batch.
        prepared (Iterable[Sequence[Trial]]): The batches, as
            ``prepare_batches`` yields them; the next is prepared only once
            the last has flown.
        on_prepared (Callable[[int, Sequence[Trial]], None] | None,
            optional): Called with each batch's number and trials once they
            are prepared, before they fly. Defaults to nothing.
        on_batch (Callable[[int, BatchStats], None] | None, optional): Called
            with each batch's number and the statistics so far, once it has
            flown. Defaults to nothing.

    Returns:
        CampaignResult: The estimate, why the campaign stopped, and each flown
            trial's outcome.

    Raises:
        RuntimeError: If more than ``max_failed_fraction`` of a batch's trials fail.
    """
    rule = _make_rule(config)
    stats = BatchStats(
        independent_trials=GENERATORS[config.generator]().trials_are_independent
    )
    rows: list[dict[str, Any]] = []
    results: list[Result] = []
    stopped_by = "all_batches"
    prepare_s = 0.0
    trials_path = config.output_dir / TRIALS
    timeout = config.trial_timeout
    try:
        start = clock.perf_counter()
        for batch, trials in enumerate(prepared):
            prepare_s += clock.perf_counter() - start
            if on_prepared is not None:
                on_prepared(batch, trials)
            flown = workers.run(
                [trial.folder for trial in trials],
                None if timeout is None else float(timeout.m_as("s")),
                config.output_dir / CAMPAIGN_LOG,
            )
            results.extend(flown)
            batch_rows = _rows(batch, len(rows), trials, flown)
            rows.extend(batch_rows)
            apogees = [
                row["apogee"] for row in batch_rows if math.isfinite(row["apogee"])
            ]
            failed_limit = config.max_failed_fraction * len(batch_rows)
            if len(batch_rows) - len(apogees) > failed_limit:
                raise RuntimeError(
                    f"More than {config.max_failed_fraction:.0%} of the trials of "
                    f"batch {batch} failed; see {display_path(trials_path)}"
                )
            stats.add_batch(apogees)
            if on_batch is not None:
                on_batch(batch, stats)
            if rule.done(stats):
                stopped_by = "rule"
                break
            start = clock.perf_counter()
    finally:
        pd.DataFrame(rows).to_csv(trials_path, index=False)
    return CampaignResult(
        stats, stopped_by, config.stopping.confidence, trials_path, results, prepare_s
    )


def _dispersions(config: EdithConfig) -> list[StandardDispersion]:
    """Return the campaign's dispersions; only the standard class can run so far."""
    for dispersion in config.dispersions:
        if not isinstance(dispersion, StandardDispersion):
            raise NotImplementedError(f"{type(dispersion).__name__} cannot run yet")
    return [d for d in config.dispersions if isinstance(d, StandardDispersion)]


def _make_rule(config: EdithConfig) -> StoppingRule:
    """Return the stopping rule a campaign's ``[stopping]`` table describes."""
    spec = config.stopping
    if spec.target is None:
        return FixedTrials()
    target = float(spec.target.m_as("m"))
    if spec.rule == StandardErrorTarget.__name__:
        return StandardErrorTarget(target)
    if spec.confidence is None:  # StoppingSettings has checked this
        raise ValueError(f'rule "{spec.rule}" needs a confidence')
    return ConfidenceIntervalTarget(target, spec.confidence)


def draw_batch(config: EdithConfig, batch: int, size: int) -> Values:
    """Return every dispersion's values for one batch of ``size`` trials.

    Each dispersion draws from its own stream, so the others' values do not
    change when one is added or removed.
    """
    generator = GENERATORS[config.generator]()
    return {
        d.key: d.distribution.ppf(
            generator.percentiles(size, stream(config.master_seed, batch, d.key)),
            d.nominal,
        )
        for d in _dispersions(config)
    }


def _as_input(
    dispersion: StandardDispersion, value: float, table: dict[str, Any]
) -> Any:
    """Return a drawn value written as the base input writes its nominal value.

    Text with units, such as "17 kg", stays text in the same units; a plain
    number stays a number.
    """
    if isinstance(node_at(table, dispersion.key), str):
        return f"{value:.12g} {dispersion.units}"
    return float(value)


def _rows(
    batch: int, first: int, trials: Sequence[Trial], flown: Sequence[Result]
) -> list[dict[str, Any]]:
    """Return a batch's rows of the trials table, one per trial flown."""
    rows = []
    for index, (trial, result) in enumerate(zip(trials, flown, strict=True), first):
        apogee, status = _outcome(trial.folder, result)
        rows.append(
            {
                "trial": index,
                "batch": batch,
                **trial.drawn,
                "apogee": apogee,
                "status": status,
            }
        )
    return rows


def _outcome(folder: Path, result: Result) -> tuple[float, str]:
    """Return a flown trial's apogee in metres, NaN if it has none, and its status."""
    if result.status is not Status.OK:
        reason = f": {result.reason}" if result.reason else ""
        return math.nan, f"{result.status}{reason}"
    apogee = _apogee(folder)
    if not math.isfinite(apogee):
        return apogee, f"FAILED: apogee is {apogee}"
    return apogee, "OK"


def _apogee(folder: Path) -> float:
    """Return the apogee a run directory's run.toml records, in metres, or NaN."""
    try:
        with (folder / "run.toml").open("rb") as file:
            return float(tomllib.load(file)["summary"]["apogee_m"])
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError):
        return math.nan


def write_provenance(config: EdithConfig) -> Path:
    """Record what regenerates a campaign in its output directory.

    That is the commit of the code, and whether it had uncommitted changes, a
    copy of the campaign file, and the version of each package the results
    depend on. The master seed is in the campaign file. A value that cannot be
    found, as outside a git checkout, is null.

    Returns:
        Path: The record, ``PROVENANCE`` in the output directory.
    """
    code = Path(__file__).resolve().parent
    changes = _git(code, "status", "--porcelain")
    record = {
        "git_commit": _git(code, "rev-parse", "HEAD"),
        "git_dirty": None if changes is None else bool(changes),
        "campaign_file": config.path.as_posix(),
        "campaign": config.path.read_text(encoding="utf-8"),
        "packages": {name: _version(name) for name in _PACKAGES},
    }
    path = config.output_dir / PROVENANCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _git(folder: Path, *args: str) -> str | None:
    """Return what a git command prints in a folder, or None if it fails."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=folder,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _version(package: str) -> str | None:
    """Return an installed package's version, or None if it is not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None
