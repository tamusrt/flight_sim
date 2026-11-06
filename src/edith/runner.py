"""Campaign orchestration: the serial loop that flies batches of dispersed trials.

Each batch is one independently seeded replicate. The loop stops when the
stopping rule is met or ``num_sims`` trials have run.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from edith.config import EdithConfig
from edith.dispersion import StandardDispersion
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
from flight_sim.config import ConfigError, build
from flight_sim.simulation import Simulation

__all__ = ["CampaignResult", "run_campaign"]

DEFAULT_BATCH_SIZE = 50

# A batch with more than this fraction of failed trials ends the campaign
MAX_FAILED_FRACTION = 0.5

# A failed trial is recorded and left out of the statistics
_TRIAL_ERRORS = (ConfigError, ValueError, RuntimeError, ArithmeticError)

Values = dict[str, np.ndarray[Any, np.dtype[np.float64]]]


@dataclass
class CampaignResult:
    """What a campaign found: the apogee estimate, in metres."""

    stats: BatchStats
    stopped_by: str  # "rule" or "num_sims"
    confidence: float
    trials_path: Path

    def summary(self) -> str:
        """Return the estimate as a line."""
        return (
            f"apogee {self.stats.mean:.6g} +/- "
            f"{self.stats.ci_half_width(self.confidence):.3g} m "
            f"({self.confidence:.0%} CI, {self.stats.trials} trials in "
            f"{self.stats.batches} batches)"
        )


def run_campaign(config: EdithConfig) -> CampaignResult:
    """Run a campaign's trials in batches until its stopping rule is met.

    Each trial's telemetry goes to its own folder under ``runs``, and every
    trial's values and apogee to ``trials.csv``, in the output folder.

    Raises:
        RuntimeError: If more than half of a batch's trials fail.
    """
    rule = _make_rule(config)
    batch_size = min(config.batch_size or DEFAULT_BATCH_SIZE, config.num_sims)
    stats = BatchStats()
    rows: list[dict[str, Any]] = []
    stopped_by = "num_sims"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = config.output_dir / "trials.csv"

    try:
        batch = 0
        while len(rows) < config.num_sims:
            size = min(batch_size, config.num_sims - len(rows))
            drawn = _draw_batch(config, batch, size)
            apogees = []
            for index in range(size):
                row: dict[str, Any] = {"trial": len(rows), "batch": batch}
                row.update({key: float(values[index]) for key, values in drawn.items()})
                try:
                    row["apogee"] = _fly(config, row["trial"], drawn, index)
                    row["status"] = "OK"
                    apogees.append(row["apogee"])
                except _TRIAL_ERRORS as error:
                    row["apogee"] = math.nan
                    row["status"] = f"FAILED: {error}"
                rows.append(row)
            if size - len(apogees) > MAX_FAILED_FRACTION * size:
                raise RuntimeError(
                    f"Most trials of batch {batch} failed; see {trials_path}"
                )
            stats.add_batch(apogees)
            batch += 1
            if rule.done(stats):
                stopped_by = "rule"
                break
    finally:
        pd.DataFrame(rows).to_csv(trials_path, index=False)

    return CampaignResult(stats, stopped_by, config.stopping.confidence, trials_path)


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


def _draw_batch(config: EdithConfig, batch: int, size: int) -> Values:
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


def _fly(config: EdithConfig, trial: int, drawn: Values, index: int) -> float:
    """Fly one trial, write its folder, and return its apogee in metres."""
    values: dict[str, Any] = {
        d.key: _as_input(d, float(drawn[d.key][index]), config.source.table)
        for d in _dispersions(config)
    }
    folder = config.output_dir / "runs" / f"trial_{trial:05d}"
    values["output.directory"] = folder.as_posix()
    flight = build(config.source.table_with(values), Simulation).run()
    output.write(flight, figures={})
    apogee = flight.summary()["apogee_m"]
    if not math.isfinite(apogee):
        raise ArithmeticError(f"apogee is {apogee}")
    return float(apogee)
