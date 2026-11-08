"""Rules that say when a campaign has run enough trials.

The last batch caps the run whatever the rule says. A campaign names a rule by the
name of a ``StoppingRule`` subclass in this module, which ``STOPPING_RULES``
lists, so a new rule is one new subclass here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from edith.statistics.estimators import BatchStats

__all__ = [
    "STOPPING_RULES",
    "ConfidenceIntervalTarget",
    "FixedTrials",
    "StandardErrorTarget",
    "StoppingRule",
]


class StoppingRule(ABC):
    """Base of the stopping rules a campaign may name with its ``[stopping]`` rule."""

    @abstractmethod
    def done(self, stats: BatchStats) -> bool:
        """Return whether the trials so far are enough."""


@dataclass(frozen=True)
class FixedTrials(StoppingRule):
    """Never done early."""

    def done(self, stats: BatchStats) -> bool:
        """Return False."""
        del stats
        return False


@dataclass(frozen=True)
class StandardErrorTarget(StoppingRule):
    """Done once the standard error is at or below ``target``."""

    target: float

    def done(self, stats: BatchStats) -> bool:
        """Return whether the standard error is known and has reached the target."""
        standard_error = stats.standard_error
        return standard_error is not None and standard_error <= self.target


@dataclass(frozen=True)
class ConfidenceIntervalTarget(StoppingRule):
    """Done once the confidence interval's half-width is at or below ``target``."""

    target: float
    confidence: float

    def done(self, stats: BatchStats) -> bool:
        """Return whether the interval is known and has narrowed to the target."""
        half_width = stats.confidence_interval_half_width(self.confidence)
        return half_width is not None and half_width <= self.target


# Each stopping rule class, by the name a campaign writes; adding a subclass of
# StoppingRule adds it here. Subclasses are concrete, whatever mypy infers.
STOPPING_RULES: dict[str, type[StoppingRule]] = {
    cls.__name__: cls  # type: ignore[type-abstract]
    for cls in StoppingRule.__subclasses__()
}
