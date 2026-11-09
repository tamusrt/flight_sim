"""Draw the uniform numbers a campaign's trials disperse from.

The sample is one row per trial and one column per draw, each dispersion
taking ``DRAWS`` columns in campaign order. Each dispersion's columns come
from its own stream, seeded from ``master_seed`` and the dispersion's key, so
adding, removing or reordering dispersions leaves the other columns unchanged.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from edith.config import EdithConfig
from edith.dispersion import BaseDispersion

# Least distance of a sample from 0 and 1, where a normal's quantile is infinite
_MARGIN = 1e-12


def sample(config: EdithConfig) -> np.ndarray:
    """Return the uniform numbers every trial draws from, one row per trial.

    Args:
        config (EdithConfig): The campaign, for its trials, seed, sampling and
            dispersions.

    Returns:
        np.ndarray: Numbers in (0, 1) of shape (num_sims, total draws). With
            "lhs" sampling each column has one sample in each of num_sims
            equal strata, in a random order, so changing num_sims changes
            every sample; with "random" the samples are independent.
    """
    columns = []
    for dispersion in config.dispersions:
        generator = np.random.default_rng(_seed(config.master_seed, dispersion.key))
        for _ in range(dispersion.DRAWS):
            u = generator.random(config.num_sims)
            if config.sampling == "lhs":
                u = (generator.permutation(config.num_sims) + u) / config.num_sims
            columns.append(np.clip(u, _MARGIN, 1.0 - _MARGIN))
    return np.column_stack(columns)


def draw_slices(dispersions: Sequence[BaseDispersion]) -> list[slice]:
    """Return the columns of the sample each dispersion draws from."""
    slices = []
    start = 0
    for dispersion in dispersions:
        slices.append(slice(start, start + dispersion.DRAWS))
        start += dispersion.DRAWS
    return slices


def _seed(master_seed: int, key: str) -> np.random.SeedSequence:
    """Return the seed of a dispersion's stream, from the master seed and its key."""
    digest = hashlib.sha256(key.encode()).digest()
    words = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)]
    return np.random.SeedSequence([master_seed, *words])
