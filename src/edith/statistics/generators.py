"""Point generators: one column of percentiles per dispersion.

Each column has its own seeded stream, so adding or removing a dispersion leaves
the others' samples unchanged. A new generator is one new class here, added to
``GENERATORS``.
"""

import zlib
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc  # type: ignore[import-untyped]

__all__ = [
    "GENERATORS",
    "LatinHypercubeGenerator",
    "PointGenerator",
    "RandomGenerator",
    "stream",
]


class PointGenerator(Protocol):
    """A point generator, named by a campaign's ``sampling``."""

    NAME: ClassVar[str]

    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` percentiles in (0, 1)."""


class RandomGenerator:
    """Independent uniform percentiles."""

    NAME: ClassVar[str] = "random"

    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` independent uniform percentiles."""
        return rng.random(n)


class LatinHypercubeGenerator:
    """Latin hypercube percentiles: one in each of ``n`` equal strata."""

    NAME: ClassVar[str] = "lhs"

    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` percentiles, one in each of ``n`` equal strata."""
        return np.asarray(qmc.LatinHypercube(d=1, rng=rng).random(n)[:, 0], dtype=float)


# Each generator class, by the ``sampling`` a campaign names it with
GENERATORS: dict[str, type[PointGenerator]] = {
    cls.NAME: cls for cls in (RandomGenerator, LatinHypercubeGenerator)
}


def stream(master_seed: int, batch: int, key: str) -> np.random.Generator:
    """Return the random stream of one dispersion in one batch."""
    return np.random.default_rng(
        np.random.SeedSequence([master_seed, batch, zlib.crc32(key.encode())])
    )
