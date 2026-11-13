"""Tests of the distributions of a campaign's outputs."""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

from edith.statistics.outputs import (
    Envelope,
    OutputDistribution,
    Requirement,
    joint_pass_probability,
    pass_probability,
    trials_for_reliability,
)


def test_empirical_percentiles_and_moments_ignore_failed_trials() -> None:
    """NaN values are dropped; the rest give the sample's own percentiles."""
    output = OutputDistribution([*range(101), math.nan])

    assert output.percentile(50) == 50.0
    assert output.percentile([10, 90]).tolist() == [10.0, 90.0]
    assert output.mean == 50.0
    assert output.std == pytest.approx(np.std(np.arange(101), ddof=1))


def test_the_best_fit_recovers_the_family_drawn_from() -> None:
    """A log-normal sample is fitted best by a log-normal, ranked by AIC."""
    rng = np.random.default_rng(0)
    output = OutputDistribution(rng.lognormal(1.0, 0.5, 250))

    fits = output.fits()
    best = output.best_fit

    assert best is not None
    assert best.name == "LogNormal"
    assert output.good_fit
    assert [fit.aic for fit in fits] == sorted(fit.aic for fit in fits)
    assert "VonMises" not in {fit.name for fit in fits}  # Asked for by name only
    true_p99 = math.exp(1.0 + 0.5 * 2.3263479)
    assert float(output.percentile(99, method="fit")) == pytest.approx(
        true_p99, rel=0.1
    )
    assert output.cdf(output.percentile(50, method="fit")) == pytest.approx(0.5)
    assert float(output.pdf(math.e)) > 0.0


def test_too_few_values_are_not_fitted() -> None:
    """Under RELIABLE_TRIALS values there is a warning and no fit."""
    output = OutputDistribution(np.linspace(0.0, 1.0, 10))

    assert output.warning is not None
    assert output.fits() == []
    assert output.best_fit is None
    with pytest.raises(ValueError, match="no family is fitted"):
        output.percentile(50, method="fit")


def test_a_poor_best_fit_is_refused() -> None:
    """Two separate clusters fit no family; the empirical method still works."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0, 0.1, 100), rng.normal(10, 0.1, 100)])
    output = OutputDistribution(values, families=["Normal"])

    assert output.best_fit is not None
    assert not output.good_fit
    with pytest.raises(ValueError, match="no good fit"):
        output.pdf(0.0)
    assert np.isfinite(output.percentile(50))


def test_an_unknown_family_is_refused() -> None:
    """Families are those of FAMILIES."""
    with pytest.raises(KeyError, match="Gamma"):
        OutputDistribution([1.0], families=["Gamma"])


def test_an_envelope_takes_percentiles_across_trials_at_each_time() -> None:
    """A trial that ended early is left out after it ends."""
    short = ([0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
    long = ([0.0, 2.0, 4.0], [0.0, 40.0, 80.0])

    envelope = Envelope.from_runs([short, long], interval=1.0)

    assert envelope.times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert envelope.curves.shape == (2, 5)
    assert envelope.percentile(50).tolist() == [0.0, 15.0, 30.0, 60.0, 80.0]
    assert envelope.percentile(100)[1] == 20.0
    spread = envelope.std()
    assert spread[1] == pytest.approx(np.std([10.0, 20.0], ddof=1))
    assert math.isnan(spread[3])  # Only one trial is still flying


def test_a_pass_probability_has_a_binomial_interval() -> None:
    """98 of 100 trials pass; the Wilson interval brackets 98 %."""
    values = [*([3100.0] * 98), 2900.0, 3400.0, math.nan]

    result = pass_probability(values, low=3000.0, high=3300.0)

    assert (result.passed, result.trials) == (98, 100)
    assert result.probability == 0.98
    assert result.low < 0.98 < result.high
    assert result.low == pytest.approx(0.9300, abs=1e-3)
    assert str(result) == "98.0% (93.0% to 99.4%, 98 of 100 trials)"


def test_a_joint_pass_needs_every_requirement_in_the_same_trial() -> None:
    """A trial passes only if both outputs are within their limits."""
    apogee = [3100.0, 3100.0, 2900.0, 3100.0]
    rail_speed = [31.0, 25.0, 31.0, math.nan]

    result = joint_pass_probability(
        [Requirement(apogee, low=3000.0), Requirement(rail_speed, low=30.0)]
    )

    assert (result.passed, result.trials) == (1, 3)  # The NaN trial is left out


def test_a_requirement_needs_a_limit() -> None:
    """With neither low nor high, nothing would be required."""
    with pytest.raises(ValueError, match="needs a low or high limit"):
        pass_probability([1.0, 2.0])
    with pytest.raises(ValueError, match="no requirements"):
        joint_pass_probability([])
    assert math.isnan(pass_probability([math.nan], low=0.0).probability)


def test_a_percentile_interval_covers_the_true_percentile() -> None:
    """Over many samples, the 95 % interval of the 90th percentile covers it."""
    true_p90 = 1.2815516
    covered = 0
    for seed in range(400):
        sample = np.random.default_rng(seed).normal(size=100)
        low, high = OutputDistribution(sample).percentile_interval(90, 0.95)
        covered += low <= true_p90 <= high

    assert covered / 400 >= 0.93  # At least the confidence, less sampling noise


def test_a_percentile_beyond_the_sample_has_an_open_interval() -> None:
    """Ten values cannot bound the 99th percentile from above."""
    low, high = OutputDistribution(np.arange(10.0)).percentile_interval(99, 0.95)

    assert low == 8.0  # P(9 or more of 10 below it) > 97.5 %
    assert high == math.inf


@pytest.mark.parametrize(
    ("reliability", "confidence", "expected"),
    [(0.95, 0.95, 59), (0.99, 0.95, 299), (0.9, 0.9, 22)],
)
def test_trials_for_reliability(
    reliability: float, confidence: float, expected: int
) -> None:
    """The success-run count: ln(1 - C) / ln(R), rounded up."""
    assert trials_for_reliability(reliability, confidence) == expected


def test_trials_for_reliability_checks_its_fractions() -> None:
    """A reliability of 1 can never be shown."""
    with pytest.raises(ValueError, match="reliability must be between 0 and 1"):
        trials_for_reliability(1.0, 0.95)


def test_the_fits_p_value_allows_for_fitting_to_the_same_values() -> None:
    """A normal fitted to skewed values passes the usual test but not this one.

    The usual p-value takes the fitted parameters as known beforehand, so it
    is too high; the bootstrap one refits each simulated sample.
    """
    values = np.random.default_rng(0).lognormal(1.0, 0.2, 250)
    output = OutputDistribution(values, families=["Normal"])
    best = output.best_fit
    assert best is not None

    usual = scipy_stats.kstest(values, best.frozen.cdf).pvalue
    p_value = output.best_fit_p_value

    assert usual > 0.05  # Wrongly passed
    assert p_value is not None
    assert p_value < 0.05
    assert not output.good_fit
    assert output.best_fit_p_value == p_value  # Worked out once


def test_no_fit_has_no_p_value() -> None:
    """With too few values there is nothing to test."""
    output = OutputDistribution(np.arange(5.0))

    assert output.best_fit_p_value is None
    assert not output.good_fit
