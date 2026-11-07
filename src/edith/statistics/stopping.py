"""Rules that say when a campaign has run enough trials.

``num_sims`` caps the run whatever the rule says. A new rule is one new class
here, added to ``STOPPING_RULES``.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from edith.statistics.estimators import BatchStats

__all__ = [
    "STOPPING_RULES",
    "ConfidenceTarget",
    "FixedN",
    "StandardErrorTarget",
    "StoppingRule",
]


class StoppingRule(Protocol):
    """A stopping rule, named by a campaign's ``[stopping]`` rule."""

    NAME: ClassVar[str]

    def done(self, stats: BatchStats) -> bool:
        """Return whether the trials so far are enough."""


@dataclass(frozen=True)
class FixedN:
    """Never done early."""

    NAME: ClassVar[str] = "fixed"

    def done(self, stats: BatchStats) -> bool:
        """Return False."""
        del stats
        return False


@dataclass(frozen=True)
class StandardErrorTarget:
    """Done once the standard error is at or below ``target``."""

    NAME: ClassVar[str] = "standard_error"

    target: float

    def done(self, stats: BatchStats) -> bool:
        """Return whether the standard error has reached the target."""
        return stats.standard_error <= self.target


@dataclass(frozen=True)
class ConfidenceTarget:
    """Done once the confidence interval's half-width is at or below ``target``."""

    NAME: ClassVar[str] = "ci_half_width"

    target: float
    confidence: float = 0.95

    def done(self, stats: BatchStats) -> bool:
        """Return whether the interval has narrowed to the target."""
        return stats.ci_half_width(self.confidence) <= self.target


STOPPING_RULES: dict[
    str, type[FixedN] | type[StandardErrorTarget] | type[ConfidenceTarget]
] = {cls.NAME: cls for cls in (FixedN, StandardErrorTarget, ConfidenceTarget)}
