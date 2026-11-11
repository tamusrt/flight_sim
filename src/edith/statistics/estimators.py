"""Running mean of a metric and its uncertainty.

How the standard error is found depends on the point generator. With independent
trials (random sampling) every trial is an independent draw, so it comes from the
spread of the trials, and one batch is enough. Otherwise (Latin hypercube
sampling) the trials of a batch are stratified, not independent, so it comes from
the spread of the batch means, each batch being an independent replicate, and it
takes two batches. With fewer than ``RELIABLE_TRIALS`` trials the estimate may
not be reliable. The mean is always that of all the trials. When failed trials
leave batches of unequal sizes, the batch-based standard error is approximate.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats  # type: ignore[import-untyped]

__all__ = ["RELIABLE_TRIALS", "BatchStats"]

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
