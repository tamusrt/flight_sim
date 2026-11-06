"""EDITH dispersion tests: drawing values and recording them in a trial."""

from pathlib import Path

import numpy as np
import pytest

from edith.config import EdithConfig
from edith.dispersion import Normal, StandardDispersion, Trial, Uniform


def test_a_normal_draws_about_its_mean() -> None:
    """The median is the mean, one sigma up is mean + sigma, and sigma 0 is the mean."""
    normal = Normal(17.0, 0.25)

    assert normal.ppf(0.5) == pytest.approx(17.0)
    assert normal.ppf(0.8413447460685429) == pytest.approx(17.25)
    assert Normal(17.0, 0.0).ppf(0.01) == 17.0


def test_a_truncated_normal_stays_within_its_bounds() -> None:
    """Quantiles 0 and 1 land on low and high, and the median on the mean."""
    normal = Normal(85.0, 2.0, low=84.0, high=86.0)

    assert normal.ppf(0.0) == pytest.approx(84.0)
    assert normal.ppf(0.5) == pytest.approx(85.0)
    assert normal.ppf(1.0) == pytest.approx(86.0)


def test_a_uniform_draws_from_low_to_high() -> None:
    """Quantiles 0, 0.5 and 1 land on low, the middle and high."""
    uniform = Uniform(2.0, 8.0)

    assert [uniform.ppf(u) for u in (0.0, 0.5, 1.0)] == [2.0, 5.0, 8.0]


def test_a_standard_dispersion_draws_one_number(campaign: Path) -> None:
    """Every dispersion of the campaign draws from one uniform number."""
    config = EdithConfig.from_file(campaign)

    assert all(isinstance(d, StandardDispersion) for d in config.dispersions)
    assert [dispersion.DRAWS for dispersion in config.dispersions] == [1, 1, 1]


def test_a_standard_dispersion_records_the_value_in_the_inputs_form(
    campaign: Path, tmp_path: Path
) -> None:
    """A drawn value is written as the input writes it: quoted, bare or plain."""
    config = EdithConfig.from_file(campaign)
    trial = Trial(tmp_path / "run")

    for dispersion in config.dispersions:
        dispersion.apply(np.array([0.5]), trial)
    table = config.source.table_with(trial.values)

    assert set(trial.values) == {dispersion.key for dispersion in config.dispersions}
    mass = table["properties"]["dry_mass_properties"]["mass"]
    assert mass.endswith(" kg") and float(mass.split()[0]) == pytest.approx(17.0)
    cg_x = table["properties"]["dry_mass_properties"]["cg_location"]["value"][0]
    assert cg_x == pytest.approx(-1.0)
    assert table["config"]["truth"]["launch_rail"]["friction_coefficient"] == 0.1
