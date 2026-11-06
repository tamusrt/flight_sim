"""Running mean of a metric and its uncertainty.

Each batch is an independent replicate, so the standard error comes from the
spread of the batch means. That stays valid under Latin hypercube sampling.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats  # type: ignore[import-untyped]

__all__ = ["BatchStats"]


@dataclass
class BatchStats:
    """Means of a metric over the batches run so far."""

    batch_means: list[float] = field(default_factory=list)
    trials: int = 0  # Trials counted, over all batches

    def add_batch(self, values: ArrayLike) -> None:
        """Add one batch's values; an empty batch is skipped."""
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return
        self.batch_means.append(float(array.mean()))
        self.trials += int(array.size)

    @property
    def batches(self) -> int:
        """Number of batches added."""
        return len(self.batch_means)

    @property
    def mean(self) -> float:
        """Mean of the batch means, NaN before the first batch."""
        return float(np.mean(self.batch_means)) if self.batch_means else math.nan

    @property
    def standard_error(self) -> float:
        """Standard error of the mean; infinite until there are two batches."""
        if self.batches < 2:
            return math.inf
        return float(np.std(self.batch_means, ddof=1) / math.sqrt(self.batches))

    def ci_half_width(self, confidence: float = 0.95) -> float:
        """Return the confidence interval's half-width (Student's t)."""
        if self.batches < 2:
            return math.inf
        t_value = float(stats.t.ppf((1.0 + confidence) / 2.0, self.batches - 1))
        return t_value * self.standard_error
