"""Tests of which inputs drive a campaign's outputs and failures."""

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from edith.statistics.sensitivity import failure_drivers, spearman


def _inputs(n: int = 200) -> pd.DataFrame:
    """Return two drawn inputs, a driver and noise, and one that never varies."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"mass": rng.normal(17.0, 0.25, n), "noise": rng.random(n), "fixed": 1.0}
    )


def test_a_monotone_output_correlates_with_its_driver_alone() -> None:
    """Apogee falls with mass; the noise column correlates with nothing."""
    inputs = _inputs()
    outputs = pd.DataFrame({"apogee": 3500.0 - 40.0 * (inputs["mass"] - 17.0) ** 3})

    result = spearman(inputs, outputs)

    assert result.correlation.loc["mass", "apogee"] == pytest.approx(-1.0)
    assert result.p_value.loc["mass", "apogee"] < 1e-6
    assert abs(result.correlation.loc["noise", "apogee"]) < 0.2
    assert math.isnan(result.correlation.loc["fixed", "apogee"])


def test_failed_trials_are_left_out_of_each_output() -> None:
    """NaN in one output does not drop that trial from another."""
    inputs = _inputs(6)
    outputs = pd.DataFrame(
        {
            "apogee": [1.0, 2.0, 3.0, math.nan, 5.0, 6.0],
            "max_mach": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    ranks = inputs.assign(mass=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = spearman(ranks, outputs)

    assert result.correlation.loc["mass", "apogee"] == pytest.approx(1.0)
    assert result.correlation.loc["mass", "max_mach"] == pytest.approx(-1.0)


def test_failure_drivers_rank_the_input_that_makes_trials_fail_first() -> None:
    """Heavy trials fail; mass comes first, and the constant input last."""
    table = _inputs()
    table["failed"] = table["mass"] > 17.3

    ranked = failure_drivers(table, ["noise", "fixed", "mass"])

    assert list(ranked.index) == ["mass", "noise", "fixed"]
    assert ranked.loc["mass", "correlation"] > 0.5
    assert math.isnan(ranked.loc["fixed", "correlation"])
