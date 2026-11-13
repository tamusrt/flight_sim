"""Tests of the distributions of a campaign's outputs."""

import math

import numpy as np
import pytest

from edith.statistics.outputs import (
    Envelope,
    OutputDistribution,
    Requirement,
    joint_pass_probability,
    pass_probability,
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
    assert best.good
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
    assert not output.best_fit.good
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
