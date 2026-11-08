"""Point generators: one column of percentiles per dispersion.

Each column has its own seeded stream, so adding or removing a dispersion leaves
the others' samples unchanged. A campaign names a generator by the name of a
``PointGenerator`` subclass in this module, which ``GENERATORS`` lists, so a new
generator is one new subclass here.
"""

import zlib
from abc import ABC, abstractmethod
from typing import Any

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


class PointGenerator(ABC):
    """Base of the point generators a campaign may name with its ``generator``."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass that leaves an abstract method or property undefined.

        Python checks these only when a class is instantiated, which a generator
        is only when a campaign first uses it.

        Raises:
            TypeError: Naming what the subclass must define.
        """
        super().__init_subclass__(**kwargs)
        abstract = sorted(
            name
            for name in dir(cls)
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False)
        )
        if abstract:
            raise TypeError(f"{cls.__name__} must define {', '.join(abstract)}")

    @property
    @abstractmethod
    def trials_are_independent(self) -> bool:
        """Whether the trials of a batch are independent draws.

        If not, as with stratified points, the spread of a batch's trials does not
        measure the uncertainty of their mean, so the standard error comes from
        several batches instead.
        """

    @abstractmethod
    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` percentiles in (0, 1)."""


class RandomGenerator(PointGenerator):
    """Independent uniform percentiles."""

    @property
    def trials_are_independent(self) -> bool:
        """Return True: every percentile is drawn independently."""
        return True

    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` independent uniform percentiles."""
        return rng.random(n)


class LatinHypercubeGenerator(PointGenerator):
    """Latin hypercube percentiles: one in each of ``n`` equal strata."""

    @property
    def trials_are_independent(self) -> bool:
        """Return False: the strata tie each percentile to the others."""
        return False

    def percentiles(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Return ``n`` percentiles, one in each of ``n`` equal strata."""
        return np.asarray(qmc.LatinHypercube(d=1, rng=rng).random(n)[:, 0], dtype=float)


# Each generator class, by the name a campaign writes; adding a subclass of
# PointGenerator adds it here. Subclasses are concrete, whatever mypy infers.
GENERATORS: dict[str, type[PointGenerator]] = {
    cls.__name__: cls  # type: ignore[type-abstract]
    for cls in PointGenerator.__subclasses__()
}


def stream(master_seed: int, batch: int, key: str) -> np.random.Generator:
    """Return the random stream of one dispersion in one batch."""
    return np.random.default_rng(
        np.random.SeedSequence([master_seed, batch, zlib.crc32(key.encode())])
    )
