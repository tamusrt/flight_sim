"""Defaults of the statistics settings, and the constants the statistics rely on.

The defaults are the values used when the campaign file leaves a setting out:
``generator``, ``min_trials``, ``max_failed_fraction`` and ``[stopping]
confidence``. The campaign file cannot change the numerical constants.
"""

from edith.statistics.generators import LatinHypercubeGenerator

__all__ = [
    "BOUND_TOLERANCE",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_GENERATOR",
    "DEFAULT_MAX_FAILED_FRACTION",
    "DEFAULT_MIN_TRIALS",
    "PERCENTILE_CUTOFF",
]

# The point generator, by the name a campaign writes
DEFAULT_GENERATOR = LatinHypercubeGenerator.__name__

# Confidence level of the apogee's interval and of the confidence-interval rule,
# as a fraction (0.95 is 95 %)
DEFAULT_CONFIDENCE = 0.95

# A batch with more than this fraction of its trials failed ends the campaign
DEFAULT_MAX_FAILED_FRACTION = 0.5

# Fewest trials before the standard error, the interval and the stopping rules are
# given; a common rule of thumb for the interval of a mean
DEFAULT_MIN_TRIALS = 30

# Percentiles stay this far from 0 and 1, so no value is infinite
PERCENTILE_CUTOFF = 1e-6

# How far a bound may lie beyond the nominal value, in the nominal value's units,
# before it counts as not bracketing it, so a bound equal to it passes
BOUND_TOLERANCE = 1e-12
