"""EDITH sampler tests: the uniform numbers trials draw from."""

from pathlib import Path

import numpy as np
import pytest

from edith.config import EdithConfig
from edith.sampler import draw_slices, sample

_MASS_TABLE = '[dispersions."properties.dry_mass_properties.mass"]'
_CG_TABLE = '[dispersions."properties.dry_mass_properties.cg_location.value[0]"]'


def _edited(campaign: Path, old: str, new: str) -> EdithConfig:
    """Return the campaign with one edit to its file."""
    text = campaign.read_text(encoding="utf-8")
    assert old in text
    campaign.write_text(text.replace(old, new), encoding="utf-8")
    return EdithConfig.from_file(campaign)


def _without_mass(text: str) -> str:
    """Return a campaign file's text without its first dispersion table."""
    return text[: text.index(_MASS_TABLE)] + text[text.index(_CG_TABLE) :]


def test_samples_have_a_column_per_draw_in_the_open_unit_interval(
    campaign: Path,
) -> None:
    """Three dispersions of one draw each give three columns, all in (0, 1)."""
    config = EdithConfig.from_file(campaign)

    samples = sample(config)

    assert samples.shape == (3, 3)
    assert np.all((samples > 0.0) & (samples < 1.0))
    assert draw_slices(config.dispersions) == [slice(0, 1), slice(1, 2), slice(2, 3)]


def test_lhs_puts_one_sample_in_each_stratum(campaign: Path) -> None:
    """With 3 trials, each column has one sample in each third of (0, 1)."""
    samples = sample(EdithConfig.from_file(campaign))

    for column in samples.T:
        assert sorted(np.floor(column * 3).astype(int)) == [0, 1, 2]


def test_samples_repeat_for_a_seed_and_differ_between_seeds(campaign: Path) -> None:
    """The same campaign samples the same numbers; another master seed does not."""
    config = EdithConfig.from_file(campaign)
    first = sample(config)

    assert np.array_equal(sample(config), first)
    other = sample(_edited(campaign, "master_seed = 7", "master_seed = 8"))
    assert not np.array_equal(other, first)


@pytest.mark.parametrize("sampling", ["lhs", "random"])
def test_a_dispersions_samples_do_not_depend_on_the_others(
    campaign: Path, sampling: str
) -> None:
    """Removing one dispersion leaves the other dispersions' columns unchanged."""
    config = _edited(
        campaign, "master_seed = 7", f'master_seed = 7\nsampling = "{sampling}"'
    )
    full = sample(config)
    text = campaign.read_text(encoding="utf-8")
    campaign.write_text(_without_mass(text), encoding="utf-8")

    fewer = sample(EdithConfig.from_file(campaign))

    assert fewer.shape == (3, 2)
    assert np.array_equal(fewer, full[:, 1:])
