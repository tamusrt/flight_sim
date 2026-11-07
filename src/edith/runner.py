"""Prepare a campaign's run directories and fly them batch by batch.

Each trial's directory is a flight_sim run directory: ``resolved_input.toml``
with the trial's drawn values and the campaign's output settings in place of
the base sim's, and copies of the data files, so ``flight_sim run`` flies it
on its own. The trials come in batches, each an independently seeded
replicate drawn by ``edith.statistics``, and fly batch by batch until the
stopping rule is met or every trial has flown. Each batch's apogees feed the
estimate, and every trial's drawn values, status and apogee go to
``trials.csv``.
"""

import math
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
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
    ConfidenceTarget,
    FixedN,
    StandardErrorTarget,
    StoppingRule,
)
from edith.targets import node_at
from flight_sim import output
from flight_sim.config.toml_files import display_path

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "RUNS",
    "TRIALS",
    "CampaignResult",
    "Trial",
    "batches",
    "draw_batch",
    "fly_campaign",
    "prepare",
    "run_directory",
]

DEFAULT_BATCH_SIZE = 50

# Folder of a campaign's output directory holding a run directory per trial
RUNS = "runs"

# Every trial's drawn values, status and apogee, in the output directory
TRIALS = "trials.csv"

# A batch with more than this fraction of failed trials ends the campaign
MAX_FAILED_FRACTION = 0.5

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
    """Return the trial indices of each batch, in order."""
    size = min(config.batch_size or DEFAULT_BATCH_SIZE, config.num_sims)
    return [
        slice(start, min(start + size, config.num_sims))
        for start in range(0, config.num_sims, size)
    ]


def prepare(config: EdithConfig, *, overwrite: bool = False) -> Iterator[Trial]:
    """Create a run directory per trial, holding its dispersed input.

    Args:
        config (EdithConfig): The campaign.
        overwrite (bool, optional): Write into a runs folder that already
            exists. Defaults to refusing.

    Yields:
        Trial: Each trial, in order, as its directory is written.

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
            for offset, index in enumerate(range(span.start, span.stop)):
                folder = run_directory(config, index)
                folder.mkdir(parents=True, exist_ok=True)
                trial = Trial(
                    folder,
                    {key: float(values[offset]) for key, values in drawn.items()},
                )
                values: dict[str, Any] = {
                    d.key: _as_input(d, trial.drawn[d.key], config.source.table)
                    for d in _dispersions(config)
                }
                values.update(output_values(config.output))
                values["output.directory"] = folder.as_posix()
                if config.input_mode == "lightweight":
                    lightweight_values = dict(values)
                    lightweight_values["output.directory"] = "."
                    output.prepare_lightweight(
                        config.source, lightweight_values, folder
                    )
                else:
                    output.prepare_source(source, values, folder, overwrite=True)
                yield trial


@dataclass
class CampaignResult:
    """What a campaign found: the apogee estimate, in metres, and each flight."""

    stats: BatchStats
    stopped_by: str  # "rule" or "num_sims"
    confidence: float
    trials_path: Path
    results: list[Result]  # Each flown trial's outcome, in trial order

    def summary(self) -> str:
        """Return the estimate as a line."""
        return (
            f"apogee {self.stats.mean:.6g} +/- "
            f"{self.stats.ci_half_width(self.confidence):.3g} m "
            f"({self.confidence:.0%} CI, {self.stats.trials} trials in "
            f"{self.stats.batches} batches)"
        )


def fly_campaign(
    config: EdithConfig,
    workers: Workers,
    trials: Sequence[Trial],
    on_batch: Callable[[int, BatchStats], None] | None = None,
) -> CampaignResult:
    """Fly prepared trials batch by batch until the stopping rule is met.

    Args:
        config (EdithConfig): The campaign, for its batches, timeout, stopping
            rule and output directory.
        workers (Workers): The worker processes that fly each batch.
        trials (Sequence[Trial]): The prepared trials, in order.
        on_batch (Callable[[int, BatchStats], None] | None, optional): Called
            with each batch's number and the statistics so far, once it has
            flown. Defaults to nothing.

    Returns:
        CampaignResult: The estimate, why the campaign stopped, and each flown
            trial's outcome.

    Raises:
        RuntimeError: If more than half of a batch's trials fail.
    """
    rule = _make_rule(config)
    stats = BatchStats()
    rows: list[dict[str, Any]] = []
    results: list[Result] = []
    stopped_by = "num_sims"
    trials_path = config.output_dir / TRIALS
    log = config.output_dir / CAMPAIGN_LOG
    timeout = config.trial_timeout
    timeout_s = None if timeout is None else float(timeout.m_as("s"))
    try:
        for batch, span in enumerate(batches(config)):
            flown = workers.run(
                [trial.folder for trial in trials[span]], timeout_s, log
            )
            results.extend(flown)
            batch_rows = _rows(batch, span.start, trials[span], flown)
            rows.extend(batch_rows)
            apogees = [
                row["apogee"] for row in batch_rows if math.isfinite(row["apogee"])
            ]
            if len(batch_rows) - len(apogees) > MAX_FAILED_FRACTION * len(batch_rows):
                raise RuntimeError(
                    f"Most trials of batch {batch} failed; see "
                    f"{display_path(trials_path)}"
                )
            stats.add_batch(apogees)
            if on_batch is not None:
                on_batch(batch, stats)
            if rule.done(stats):
                stopped_by = "rule"
                break
    finally:
        pd.DataFrame(rows).to_csv(trials_path, index=False)
    return CampaignResult(
        stats, stopped_by, config.stopping.confidence, trials_path, results
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
        return FixedN()
    target = float(spec.target.m_as("m"))
    if spec.rule == StandardErrorTarget.NAME:
        return StandardErrorTarget(target)
    return ConfidenceTarget(target, spec.confidence)


def draw_batch(config: EdithConfig, batch: int, size: int) -> Values:
    """Return every dispersion's values for one batch of ``size`` trials.

    Each dispersion draws from its own stream, so the others' values do not
    change when one is added or removed.
    """
    generator = GENERATORS[config.sampling]()
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
