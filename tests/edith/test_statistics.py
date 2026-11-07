"""EDITH statistics tests: distributions, generators, estimators and the loop."""

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from edith.config import EdithConfig
from edith.parallel import Workers
from edith.runner import CampaignResult, fly_campaign, prepare
from edith.statistics.distributions import Normal, Uniform, clip_percentiles
from edith.statistics.estimators import BatchStats
from edith.statistics.generators import GENERATORS, stream
from edith.statistics.stopping import (
    STOPPING_RULES,
    ConfidenceTarget,
    FixedN,
    StandardErrorTarget,
)

_EXAMPLE = Path("examples/sol_invictus")


def test_normal_median_is_the_nominal_value() -> None:
    """The 50th percentile of a normal distribution is its nominal value."""
    assert Normal(2.0).ppf(np.array([0.5]), 10.0)[0] == pytest.approx(10.0)


def test_truncated_normal_stays_inside_its_bounds() -> None:
    """Percentiles from 0 to 1 give values between low and high."""
    values = Normal(1.0, low=9.0, high=10.5).ppf(np.linspace(0, 1, 101), 10.0)

    assert values.min() >= 9.0 - 1e-9
    assert values.max() <= 10.5 + 1e-9


def test_zero_sigma_keeps_the_nominal_value() -> None:
    """A normal distribution with no spread always gives the nominal value."""
    assert np.all(Normal(0.0).ppf(np.array([0.0, 0.3, 1.0]), 4.0) == 4.0)


def test_extreme_percentiles_give_finite_values() -> None:
    """Percentiles of exactly 0 and 1 are clipped, so no value is infinite."""
    extremes = np.array([0.0, 1.0])

    assert np.all(np.isfinite(Normal(1.0).ppf(extremes, 0.0)))
    assert np.all(np.isfinite(Normal(1.0, high=2.0).ppf(extremes, 0.0)))
    assert clip_percentiles(extremes).tolist() == [1e-6, 1 - 1e-6]


def test_uniform_ppf_spans_low_to_high() -> None:
    """A uniform distribution maps the percentiles linearly onto its range."""
    values = Uniform(2.0, 4.0).ppf(np.array([0.0, 0.5, 1.0]), 3.0)

    assert values == pytest.approx([2.0, 3.0, 4.0], abs=1e-5)


@pytest.mark.parametrize("name", ["lhs", "random"])
def test_generators_are_reproducible(name: str) -> None:
    """The same seed, batch and key give the same percentiles."""
    first = GENERATORS[name]().percentiles(10, stream(1, 0, "a"))
    second = GENERATORS[name]().percentiles(10, stream(1, 0, "a"))
    other_batch = GENERATORS[name]().percentiles(10, stream(1, 1, "a"))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_batch)


def test_lhs_puts_one_point_in_each_stratum() -> None:
    """Latin hypercube percentiles fill every tenth of [0, 1] once."""
    percentiles = GENERATORS["lhs"]().percentiles(10, stream(3, 0, "a"))

    assert sorted(np.floor(percentiles * 10).astype(int)) == list(range(10))


def test_streams_are_independent_of_other_keys() -> None:
    """One dispersion's percentiles do not depend on which others exist."""
    generator = GENERATORS["lhs"]()
    alone = generator.percentiles(8, stream(5, 2, "mass"))
    with_another = [
        generator.percentiles(8, stream(5, 2, key)) for key in ("wind", "mass")
    ]

    assert np.array_equal(alone, with_another[1])


def test_batch_statistics() -> None:
    """The standard error and interval come from the spread of batch means."""
    stats = BatchStats()
    assert math.isinf(stats.standard_error)
    stats.add_batch([1.0, 3.0])  # mean 2
    stats.add_batch([3.0, 5.0])  # mean 4
    stats.add_batch([5.0, 7.0])  # mean 6

    assert (stats.batches, stats.trials, stats.mean) == (3, 6, 4.0)
    assert stats.standard_error == pytest.approx(2.0 / math.sqrt(3))
    assert stats.ci_half_width(0.95) == pytest.approx(4.302653 * 2.0 / math.sqrt(3))


def test_stopping_rules() -> None:
    """Fixed never stops early; the others stop once their target is met."""
    stats = BatchStats()
    for mean in (10.0, 11.0, 9.0, 10.0):
        stats.add_batch([mean])

    assert set(STOPPING_RULES) == {"fixed", "standard_error", "ci_half_width"}
    assert not FixedN().done(stats)
    assert StandardErrorTarget(1.0).done(stats)
    assert not StandardErrorTarget(0.1).done(stats)
    assert ConfidenceTarget(5.0).done(stats)
    assert not ConfidenceTarget(0.5).done(stats)


def _campaign(tmp_path: Path, num_sims: int = 4, stopping: str = "") -> Path:
    """Write a small campaign beside a copy of the example sim."""
    shutil.copytree(_EXAMPLE, tmp_path / "base", ignore=shutil.ignore_patterns("out"))
    path = tmp_path / "edith_config.toml"
    path.write_text(
        f"""
num_sims = {num_sims}
master_seed = 7
base_sim = "base/sim.toml"
output_dir = "out"
batch_size = 2
{stopping}
[output]
plots = []

[dispersions."config.truth.launch_rail.elevation"]
dist_type = "normal"
sigma = "0.5 deg"
high = "90 deg"

[dispersions."properties.dry_mass_properties.mass"]
dist_type = "uniform"
half_width = "0.25 kg"
""",
        encoding="utf-8",
    )
    return path


def _fly(config: EdithConfig) -> tuple[pd.DataFrame, CampaignResult]:
    """Prepare and fly a campaign on two workers, returning its trials and result."""
    trials = list(prepare(config))
    with Workers(2, config.output_dir / "workers") as workers:
        result = fly_campaign(config, workers, trials)
    return pd.read_csv(result.trials_path), result


def test_a_small_campaign_flies_and_records_every_trial(tmp_path: Path) -> None:
    """Four trials in two batches give four finite apogees and their folders."""
    config = EdithConfig.from_file(_campaign(tmp_path))

    frame, result = _fly(config)

    assert len(frame) == 4
    assert frame["status"].eq("OK").all()
    assert np.isfinite(frame["apogee"]).all()
    assert (frame["apogee"] > 100).all()
    assert list(frame["batch"]) == [0, 0, 1, 1]
    assert "properties.dry_mass_properties.mass" in frame.columns
    assert (result.stats.batches, result.stats.trials) == (2, 4)
    assert result.stopped_by == "num_sims"
    assert (config.output_dir / "runs" / "0003" / "run.toml").is_file()
    assert "apogee" in result.summary()


def test_a_campaign_stops_early_when_its_target_is_met(tmp_path: Path) -> None:
    """A loose standard-error target ends the campaign after two batches."""
    stopping = '[stopping]\nrule = "standard_error"\ntarget = "100000 m"\n'
    config = EdithConfig.from_file(_campaign(tmp_path, num_sims=20, stopping=stopping))

    frame, result = _fly(config)

    assert result.stopped_by == "rule"
    assert result.stats.batches == 2
    assert len(frame) == 4
    assert len(result.results) == 4
