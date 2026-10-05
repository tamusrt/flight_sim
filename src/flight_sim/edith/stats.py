"""Confidence intervals for the numbers EDITH reports (90% unless asked otherwise).

* A probability from yes/no outcomes uses the Wilson interval, which behaves
  near 0 and 1 where the usual normal formula does not. When the samples come
  from several independent scrambled Sobol sets, the spread between the sets
  gives a second interval, and the wider of the two is reported. When nothing
  was ever seen (zero events), the bound is the exact Clopper-Pearson one.
* The intervals are two-sided: a 90% interval leaves 5% of the probability below
  its low end and 5% above its high end. So its high end alone is a one-sided
  bound with more confidence: "95% sure it is below X" (``one_sided``). Say it
  that way; calling the high end "90% sure it is below X" would be wrong.
* A mean, a percentile or a footprint size uses the bootstrap: resample the
  runs with replacement many times and take the middle 90% of the answers.

All of these treat the runs as independent draws, which is safe: a Sobol set
spreads its points more evenly than random ones, so the true uncertainty is
no larger than the interval says.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.90
BOOTSTRAPS = 1000


def one_sided(confidence: float) -> float:
    """The confidence of the high end of a two-sided interval, taken alone.

    A two-sided 90% interval puts 5% of the probability above its high end, so
    that end is a one-sided bound at 95%.
    """
    return 1.0 - 0.5 * (1.0 - confidence)


@dataclass(frozen=True)
class Interval:
    """An estimate and the range it is likely to be in (two-sided, ``confidence``)."""

    estimate: float
    low: float
    high: float
    confidence: float = CONFIDENCE

    @property
    def half_width(self) -> float:
        """Half the width of the interval."""
        return 0.5 * (self.high - self.low)

    def as_dict(self, digits: int = 4) -> dict[str, float]:
        """The interval as plain numbers, for the report file."""
        return {
            "estimate": round(self.estimate, digits),
            "low": round(self.low, digits),
            "high": round(self.high, digits),
            "confidence": self.confidence,
        }


def wilson(events: int, runs: int, confidence: float = CONFIDENCE) -> Interval:
    """Wilson score interval for ``events`` out of ``runs``."""
    if runs <= 0:
        return Interval(float("nan"), 0.0, 1.0, confidence)
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    p = events / runs
    denom = 1.0 + z * z / runs
    centre = (p + z * z / (2.0 * runs)) / denom
    spread = z * np.sqrt(p * (1.0 - p) / runs + z * z / (4.0 * runs * runs)) / denom
    return Interval(
        p,
        float(max(centre - spread, 0.0)),
        float(min(centre + spread, 1.0)),
        confidence,
    )


def clopper_pearson(events: int, runs: int, confidence: float = CONFIDENCE) -> Interval:
    """The exact two-sided interval; its upper end bounds a probability never seen
    (with the confidence ``one_sided(confidence)``, 95% for a 90% interval)."""
    if runs <= 0:
        return Interval(float("nan"), 0.0, 1.0, confidence)
    alpha = 1.0 - confidence
    low = (
        0.0
        if events == 0
        else float(stats.beta.ppf(alpha / 2, events, runs - events + 1))
    )
    high = (
        1.0
        if events == runs
        else float(stats.beta.ppf(1 - alpha / 2, events + 1, runs - events))
    )
    return Interval(events / runs, low, high, confidence)


def proportion(
    flags: np.ndarray,
    sets: np.ndarray | None = None,
    confidence: float = CONFIDENCE,
) -> Interval:
    """A probability with its interval.

    Args:
        flags (np.ndarray): True where the event happened, one per run.
        sets (np.ndarray | None): Which independent Sobol set each run is from.
        confidence (float): Confidence level.
    """
    flags = np.asarray(flags, dtype=bool)
    runs, events = int(flags.size), int(flags.sum())
    result = wilson(events, runs, confidence)
    if events in (0, runs):
        exact = clopper_pearson(events, runs, confidence)
        result = Interval(
            result.estimate,
            min(result.low, exact.low),
            max(result.high, exact.high),
            confidence,
        )
    if sets is not None:
        labels = np.unique(sets)
        if labels.size >= 4:
            shares = np.array([flags[sets == s].mean() for s in labels])
            t = float(stats.t.ppf(0.5 + 0.5 * confidence, labels.size - 1))
            error = t * float(shares.std(ddof=1)) / np.sqrt(labels.size)
            result = Interval(
                result.estimate,
                min(result.low, result.estimate - error),
                max(result.high, result.estimate + error),
                confidence,
            )
    return Interval(
        result.estimate, max(result.low, 0.0), min(result.high, 1.0), confidence
    )


def widen(interval: Interval, extra: float) -> Interval:
    """The interval made ``extra`` wider on both sides, kept inside 0 and 1."""
    return Interval(
        interval.estimate,
        max(interval.low - extra, 0.0),
        min(interval.high + extra, 1.0),
        interval.confidence,
    )


def bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    confidence: float = CONFIDENCE,
    resamples: int = BOOTSTRAPS,
) -> Interval:
    """Bootstrap interval of a statistic of ``values`` (one row per run)."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence)
    estimate = float(statistic(values))
    if values.shape[0] < 3:
        return Interval(estimate, estimate, estimate, confidence)
    draws = np.empty(resamples)
    n = values.shape[0]
    for i in range(resamples):
        draws[i] = statistic(values[rng.integers(0, n, n)])
    alpha = 1.0 - confidence
    low, high = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(estimate, float(low), float(high), confidence)


def mean_interval(
    values: np.ndarray, rng: np.random.Generator, confidence: float = CONFIDENCE
) -> Interval:
    """Mean with its bootstrap interval."""
    return bootstrap(values, lambda v: float(np.mean(v)), rng, confidence)


def percentile_interval(
    values: np.ndarray,
    percent: float,
    rng: np.random.Generator,
    confidence: float = CONFIDENCE,
) -> Interval:
    """A percentile of the runs with its bootstrap interval."""
    return bootstrap(
        values, lambda v: float(np.percentile(v, percent)), rng, confidence
    )
