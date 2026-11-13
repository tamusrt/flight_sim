"""Running mean of a metric and its uncertainty.

How the standard error is found depends on the point generator. With independent
trials (random sampling) every trial is an independent draw, so it comes from the
spread of the trials, and one batch is enough. Otherwise (Latin hypercube
sampling) the trials of a batch are stratified, not independent, so it comes from
the spread of the batch means, each batch being an independent replicate, and it
takes two batches. With fewer than ``RELIABLE_TRIALS`` trials the estimate may
not be reliable. The mean is always that of all the trials. When failed trials
leave batches of unequal sizes, the batch-based standard error is approximate.

The same spread says how many trials a target needs, so the first batches act
as a pilot: ``pilot_message`` reports that, and the time the rest should take.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats  # type: ignore[import-untyped]

__all__ = ["RELIABLE_TRIALS", "BatchStats", "pilot_message"]

# Fewest trials for a reliable interval of a mean, a common rule of thumb
RELIABLE_TRIALS = 30


@dataclass(kw_only=True)
class BatchStats:
    """Mean of a metric over the trials run so far, and how well it is known."""

    # Whether every trial is an independent draw, as the point generator says
    independent_trials: bool
    batch_means: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)  # Every trial's value

    def add_batch(self, values: ArrayLike) -> None:
        """Add one batch's values; an empty batch is skipped."""
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return
        self.batch_means.append(float(array.mean()))
        self.values.extend(array.tolist())

    @property
    def trials(self) -> int:
        """Trials counted, over all batches."""
        return len(self.values)

    @property
    def batches(self) -> int:
        """Number of batches added."""
        return len(self.batch_means)

    @property
    def mean(self) -> float:
        """Mean of all the trials, NaN before the first batch."""
        return float(np.mean(self.values)) if self.values else math.nan

    @property
    def standard_error(self) -> float | None:
        """Standard error of the mean, or None until ``requirement`` is met."""
        samples = self._samples
        if len(samples) < 2:
            return None
        return float(np.std(samples, ddof=1) / math.sqrt(len(samples)))

    def confidence_interval_half_width(self, confidence: float) -> float | None:
        """Return the two-sided confidence interval's half-width (Student's t).

        Returns None until ``requirement`` is met.
        """
        standard_error = self.standard_error
        if standard_error is None:
            return None
        t_value = float(stats.t.ppf((1.0 + confidence) / 2.0, len(self._samples) - 1))
        return t_value * standard_error

    def trials_needed(self, target: float, confidence: float | None) -> int | None:
        """Return the trials in all a target needs, from the spread so far.

        It is ``(k s / target)**2`` samples, ``s`` being the spread the
        standard error comes from and ``k`` Student's t at ``confidence``, with
        the degrees of freedom so far, or 1 for a standard error target. With
        stratified trials the samples are batch means, so it is that many
        batches of the mean batch size so far, which is when the stopping rule
        would fire.

        Args:
            target (float): The standard error, or the confidence interval's
                half-width, to reach.
            confidence (float | None): The interval's confidence, or None for
                a standard error target.

        Returns:
            int | None: The trials, at least those of two samples, or None
                until ``requirement`` is met.
        """
        samples = self._samples
        if len(samples) < 2:
            return None
        k = 1.0
        if confidence is not None:
            k = float(stats.t.ppf((1.0 + confidence) / 2.0, len(samples) - 1))
        count = max(math.ceil((k * float(np.std(samples, ddof=1)) / target) ** 2), 2)
        if self.independent_trials:
            return count
        return count * math.ceil(self.trials / self.batches)

    @property
    def requirement(self) -> str | None:
        """What a standard error still needs, or None once there is one."""
        if self.standard_error is not None:
            return None
        if self.independent_trials:
            return "Requires at least two trials for standard error"
        return "Requires additional batches for standard error"

    @property
    def warning(self) -> str | None:
        """A warning that the estimate may be unreliable, or None if it is not."""
        if self.trials >= RELIABLE_TRIALS:
            return None
        return (
            f"only {self.trials} trials flown, fewer than {RELIABLE_TRIALS}; "
            "the output might not be reliable"
        )

    @property
    def _samples(self) -> list[float]:
        """The values the standard error is the spread of."""
        return self.values if self.independent_trials else self.batch_means


def pilot_message(
    estimate: BatchStats,
    *,
    target: float | None,
    confidence: float | None,
    flight_per_batch: int,
    num_batches: int,
    batch_seconds: float,
    units: str = "m",
) -> str | None:
    """Return what the batches so far say about the rest of the campaign.

    Args:
        estimate (BatchStats): The statistics so far.
        target (float | None): The stopping rule's target, or None for none.
        confidence (float | None): Its confidence, or None for a standard
            error target.
        flight_per_batch (int): Trials in each batch.
        num_batches (int): The batches the campaign may fly at most.
        batch_seconds (float): The mean wall time of a batch so far.
        units (str, optional): The target's units. Defaults to "m".

    Returns:
        str | None: Such as "about 400 trials (8 batches) needed for +/- 5 m
            at 95%; about 3 min left", or None with nothing left to say.
    """
    remaining = num_batches - estimate.batches
    parts = []
    needed = None if target is None else estimate.trials_needed(target, confidence)
    if target is not None and needed is not None:
        batches = math.ceil(needed / flight_per_batch)
        goal = (
            f"a standard error of {target:g} {units}"
            if confidence is None
            else f"+/- {target:g} {units} at {confidence:.0%}"
        )
        plural = "es" if batches != 1 else ""
        parts.append(
            f"about {needed} trials ({batches} batch{plural}) needed for {goal}"
        )
        if batches > num_batches:
            parts[-1] += f", more than the {num_batches} set"
        remaining = min(remaining, max(batches - estimate.batches, 0))
    if remaining > 0:
        parts.append(f"about {_duration(remaining * batch_seconds)} left")
    return "; ".join(parts) or None


def _duration(seconds: float) -> str:
    """Return a wall time in words, such as "40 s", "12 min" or "2.5 h"."""
    if seconds < 120.0:
        return f"{seconds:.0f} s"
    if seconds < 7200.0:
        return f"{seconds / 60.0:.0f} min"
    return f"{seconds / 3600.0:.1f} h"
