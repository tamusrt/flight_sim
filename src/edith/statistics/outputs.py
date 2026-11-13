"""Distributions of a campaign's outputs, from the trials flown.

Pure statistics on arrays, as ``estimators`` is: ``edith.statistics.results``
reads the arrays. ``OutputDistribution`` describes one scalar output, such as
the apogee: empirically by default, exact for the trials flown, or from the
best of several fitted families, which is smooth and reaches tails beyond the
sample. ``Envelope`` gives an output over time, such as the altitude, as a
curve of percentiles across trials at each time, whose points come from
different trials. ``pass_probability`` is the chance a requirement is met.
NaN values, as failed trials give, are left out.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats  # type: ignore[import-untyped]

from edith.statistics.estimators import RELIABLE_TRIALS

__all__ = [
    "BOOTSTRAP_SAMPLES",
    "FAMILIES",
    "GOOD_FIT_P_VALUE",
    "Envelope",
    "Fit",
    "OutputDistribution",
    "PassProbability",
    "Requirement",
    "joint_pass_probability",
    "pass_probability",
    "trials_for_reliability",
]

# The scipy family of each distribution an output may be fitted with, named as
# the input distributions are. VonMises suits angles only, in radians within
# half a turn of their mean, so a caller asks for it by name.
FAMILIES: dict[str, Any] = {
    "Normal": stats.norm,
    "StudentT": stats.t,
    "Uniform": stats.uniform,
    "Triangular": stats.triang,
    "Weibull": stats.weibull_min,
    "LogNormal": stats.lognorm,
    "Beta": stats.beta,
    "VonMises": stats.vonmises,
}

# Families fitted unless others are asked for
DEFAULT_FAMILIES = tuple(name for name in FAMILIES if name != "VonMises")

# A fit whose Kolmogorov-Smirnov p-value is below this describes the sample poorly
GOOD_FIT_P_VALUE = 0.05

# Samples drawn from the best fit, and refitted, to find its p-value, and the
# seed they are drawn with, so the same values always give the same p-value
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 0

Method = Literal["empirical", "fit"]


class Fit(NamedTuple):
    """One family fitted to an output by maximum likelihood."""

    name: str  # Key of FAMILIES
    parameters: tuple[float, ...]  # As the scipy family takes them
    aic: float  # Akaike information criterion; lower is better
    ks_statistic: float  # Kolmogorov-Smirnov distance from the sample

    @property
    def frozen(self) -> Any:
        """The fitted scipy distribution."""
        return FAMILIES[self.name](*self.parameters)


class OutputDistribution:
    """Distribution of one scalar output over the trials flown."""

    def __init__(
        self, values: ArrayLike, families: Sequence[str] = DEFAULT_FAMILIES
    ) -> None:
        """Keep the finite values, and the families ``fits`` tries.

        Args:
            values (ArrayLike): The output of each trial; NaN for failed ones.
            families (Sequence[str], optional): Keys of ``FAMILIES``. Defaults
                to all but VonMises.

        Raises:
            KeyError: If a family is not one of ``FAMILIES``.
        """
        array = np.asarray(values, dtype=float).ravel()
        self.values: NDArray[np.float64] = array[np.isfinite(array)]
        unknown = [name for name in families if name not in FAMILIES]
        if unknown:
            raise KeyError(f"{unknown} are not among {list(FAMILIES)}")
        self.families = tuple(families)

    @property
    def mean(self) -> float:
        """Mean of the values."""
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        """Sample standard deviation of the values."""
        return float(np.std(self.values, ddof=1))

    @property
    def warning(self) -> str | None:
        """A warning that there are too few values to fit, or None."""
        if len(self.values) >= RELIABLE_TRIALS:
            return None
        return (
            f"only {len(self.values)} values, fewer than {RELIABLE_TRIALS}; "
            "no family is fitted, and percentiles are empirical only"
        )

    def percentile(
        self, q: ArrayLike, method: Method = "empirical"
    ) -> NDArray[np.float64]:
        """Return the value below which ``q`` percent of the output lies.

        Args:
            q (ArrayLike): Percentiles, from 0 to 100.
            method (Method, optional): "empirical", from the values, or "fit",
                from ``best_fit``. Defaults to "empirical".

        Raises:
            ValueError: If the method is "fit" and there is no good fit.
        """
        percentiles = np.asarray(q, dtype=np.float64)
        if method == "empirical":
            return np.asarray(np.percentile(self.values, percentiles), dtype=float)
        return np.asarray(self._good_fit().frozen.ppf(percentiles / 100.0), dtype=float)

    def fits(self) -> list[Fit]:
        """Return each family that could be fitted, lowest AIC (best) first.

        None is fitted with fewer than ``RELIABLE_TRIALS`` values. Only the
        best is tested for how well it fits, by ``best_fit_p_value``.
        """
        return list(self._fits)

    @property
    def best_fit(self) -> Fit | None:
        """The fit of lowest AIC, whether good or not, or None if there is none."""
        return self._fits[0] if self._fits else None

    @cached_property
    def best_fit_p_value(self) -> float | None:
        """The Kolmogorov-Smirnov p-value of the best fit, or None with no fit.

        The usual p-value assumes the family's parameters were known
        beforehand. Fitted to the same values, the family sits closer to them
        than the truth would, so that p-value is too high. This one is found
        by drawing ``BOOTSTRAP_SAMPLES`` samples from the fit, refitting each
        and comparing their distances with the sample's (a parametric
        bootstrap). It takes up to several seconds, once. NaN if refitting
        fails. Like any Kolmogorov-Smirnov p-value it assumes independent
        trials; Latin hypercube trials follow their distribution more evenly
        than those, so with them it runs high and rejects few fits.
        """
        best = self.best_fit
        if best is None:
            return None
        family = FAMILIES[best.name]
        names = [*(family.shapes.split(", ") if family.shapes else []), "loc", "scale"]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                result = stats.goodness_of_fit(
                    family,
                    self.values,
                    guessed_params=dict(zip(names, best.parameters, strict=True)),
                    statistic="ks",
                    n_mc_samples=BOOTSTRAP_SAMPLES,
                    rng=np.random.default_rng(BOOTSTRAP_SEED),
                )
            except (RuntimeError, ValueError, FloatingPointError):
                return math.nan
        return float(result.pvalue)

    @property
    def good_fit(self) -> bool:
        """Whether there is a best fit and the sample is consistent with it."""
        p_value = self.best_fit_p_value
        return p_value is not None and p_value >= GOOD_FIT_P_VALUE

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return the best fit's probability density at ``x``.

        Raises:
            ValueError: If there is no good fit.
        """
        return np.asarray(self._good_fit().frozen.pdf(x), dtype=float)

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return the best fit's cumulative probability at ``x``.

        Raises:
            ValueError: If there is no good fit.
        """
        return np.asarray(self._good_fit().frozen.cdf(x), dtype=float)

    def percentile_interval(self, q: float, confidence: float) -> tuple[float, float]:
        """Return a confidence interval of the true ``q``th percentile, 0 to 100.

        It comes from order statistics: the count of values below the true
        percentile is binomial, so two sorted values bracket it with at least
        ``confidence``, whatever the distribution. It assumes independent
        trials, so with Latin hypercube sampling it is cautious.

        Returns:
            tuple[float, float]: The bounds, -inf or inf where too few values
                lie beyond the percentile to bound it.
        """
        ordered = np.sort(self.values)
        n, fraction = len(ordered), q / 100.0
        tail = (1.0 - confidence) / 2.0
        # Ranks, from 1, of the values bracketing it: no more than ``tail`` of
        # the chance lies on either side
        lower = int(stats.binom.ppf(tail, n, fraction))
        upper = int(stats.binom.ppf(1.0 - tail, n, fraction)) + 1
        return (
            float(ordered[lower - 1]) if lower >= 1 else -math.inf,
            float(ordered[upper - 1]) if upper <= n else math.inf,
        )

    @cached_property
    def _fits(self) -> tuple[Fit, ...]:
        """Each family fitted, best first."""
        if len(self.values) < RELIABLE_TRIALS or np.ptp(self.values) == 0.0:
            return ()
        fits = (_fit(name, self.values) for name in self.families)
        return tuple(
            sorted((fit for fit in fits if fit is not None), key=lambda f: f.aic)
        )

    def _good_fit(self) -> Fit:
        """Return the best fit, checking it describes the sample.

        Raises:
            ValueError: If there is none, or it is not good.
        """
        best = self.best_fit
        if best is None:
            raise ValueError(f"no family is fitted: {self.warning or 'no spread'}")
        if not self.good_fit:
            raise ValueError(
                f"no good fit: the best, {best.name}, has a Kolmogorov-Smirnov "
                f"p-value of {self.best_fit_p_value:.3g}; use the empirical method"
            )
        return best


def _fit(name: str, values: NDArray[np.float64]) -> Fit | None:
    """Return a family fitted to values by maximum likelihood, or None if it fails."""
    family = FAMILIES[name]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            parameters = tuple(float(p) for p in family.fit(values))
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        log_likelihood = float(np.sum(family.logpdf(values, *parameters)))
        if not math.isfinite(log_likelihood):
            return None  # A value lies outside the fitted support
        ks = stats.kstest(values, family.cdf, args=parameters)
    aic = 2.0 * len(parameters) - 2.0 * log_likelihood
    return Fit(name, parameters, aic, float(ks.statistic))


@dataclass(frozen=True)
class Envelope:
    """An output over time across trials: a curve per trial on a shared time grid.

    Each statistic is taken across trials at each time, so the 90th percentile
    curve's points come from different trials. A trial's curve is NaN after it
    ends, and is left out at those times.
    """

    times: NDArray[np.float64]  # The grid, in seconds
    curves: NDArray[np.float64]  # One row per trial, one column per time

    @classmethod
    def from_runs(
        cls,
        runs: Sequence[tuple[ArrayLike, ArrayLike]],
        interval: float,
    ) -> Self:
        """Interpolate each trial's output onto a grid from launch to the last end.

        Args:
            runs (Sequence[tuple[ArrayLike, ArrayLike]]): Each trial's times,
                in seconds, and output, as ``results.read_runs`` returns them.
            interval (float): Seconds between grid times, such as the
                campaign's ``sample_interval``.

        Returns:
            Envelope: The curves, NaN beyond each trial's end.
        """
        series = [
            (np.asarray(t, dtype=float), np.asarray(v, dtype=float)) for t, v in runs
        ]
        start = min(float(t[0]) for t, _ in series)
        end = max(float(t[-1]) for t, _ in series)
        times = np.arange(start, end + interval / 2.0, interval)
        curves = np.array(
            [np.interp(times, t, v, left=np.nan, right=np.nan) for t, v in series]
        )
        return cls(times, curves)

    def percentile(self, q: float) -> NDArray[np.float64]:
        """Return the curve of the ``q``th percentile across trials, 0 to 100."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN times
            return np.asarray(np.nanpercentile(self.curves, q, axis=0), dtype=float)

    def std(self) -> NDArray[np.float64]:
        """Return the curve of the sample standard deviation across trials."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Times of one trial
            return np.asarray(np.nanstd(self.curves, axis=0, ddof=1), dtype=float)


class PassProbability(NamedTuple):
    """How likely a requirement is met, from the trials flown."""

    probability: float  # Fraction of trials that met it
    low: float  # Bounds of its confidence interval (Wilson)
    high: float
    passed: int
    trials: int  # Trials counted, those with a value

    def __str__(self) -> str:
        """Return it as a line, such as "98.0% (93.0% to 99.4%, 98 of 100 trials)"."""
        return (
            f"{self.probability:.1%} ({self.low:.1%} to {self.high:.1%}, "
            f"{self.passed} of {self.trials} trials)"
        )


class Requirement(NamedTuple):
    """An output and the limits it must lie within, None for no limit."""

    values: ArrayLike
    low: float | None = None
    high: float | None = None


def pass_probability(
    values: ArrayLike,
    low: float | None = None,
    high: float | None = None,
    confidence: float = 0.95,
) -> PassProbability:
    """Return how likely an output lies from ``low`` to ``high``, inclusive.

    Trials with NaN are left out. The interval assumes independent trials, so
    with Latin hypercube sampling it is approximate, generally cautious.

    Raises:
        ValueError: If neither limit is given.
    """
    return joint_pass_probability([Requirement(values, low, high)], confidence)


def joint_pass_probability(
    requirements: Sequence[Requirement], confidence: float = 0.95
) -> PassProbability:
    """Return how likely every requirement is met in the same trial.

    Each requirement's values are those of the same trials, in order. A trial
    with NaN in any of them is left out, as ``pass_probability`` says.

    Raises:
        ValueError: If a requirement has neither limit, or there are none.
    """
    if not requirements:
        raise ValueError("no requirements")
    arrays = [np.asarray(r.values, dtype=float) for r in requirements]
    counted = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    passed = counted.copy()
    for requirement, array in zip(requirements, arrays, strict=True):
        if requirement.low is None and requirement.high is None:
            raise ValueError("a requirement needs a low or high limit")
        with np.errstate(invalid="ignore"):
            if requirement.low is not None:
                passed &= array >= requirement.low
            if requirement.high is not None:
                passed &= array <= requirement.high
    trials, count = int(counted.sum()), int(passed.sum())
    if trials == 0:
        return PassProbability(math.nan, math.nan, math.nan, 0, 0)
    interval = stats.binomtest(count, trials).proportion_ci(
        confidence_level=confidence, method="wilson"
    )
    return PassProbability(
        count / trials, float(interval.low), float(interval.high), count, trials
    )


def trials_for_reliability(reliability: float, confidence: float) -> int:
    """Return the trials, all passing, that show a reliability at a confidence.

    If every one of n trials passes, the chance of that with a reliability
    below ``reliability`` is under ``1 - confidence`` once n is at least
    ln(1 - confidence) / ln(reliability): 59 for 95 % at 95 %. It needs no
    data, so it can size a campaign before it flies.

    Raises:
        ValueError: If either is not between 0 and 1.
    """
    for name, value in (("reliability", reliability), ("confidence", confidence)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return math.ceil(math.log(1.0 - confidence) / math.log(reliability) - 1e-9)
