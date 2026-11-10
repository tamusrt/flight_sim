"""Distributions a dispersion draws from, each with an inverse CDF (``ppf``).

A campaign names a distribution by the name of a ``Distribution`` subclass,
which ``DISTRIBUTIONS`` lists. Defining a subclass registers it, in any module
that is imported, so a new distribution is one new subclass. Numbers are in
the units of the nominal value being dispersed.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from edith.statistics.abstract import check_abstract_members
from edith.targets import Nominal

__all__ = [
    "DISTRIBUTIONS",
    "Distribution",
    "Normal",
    "Parameter",
    "ParameterError",
    "Uniform",
    "clip_percentiles",
    "quantity_text",
]

# Percentiles stay this far from 0 and 1, so no value is infinite
PERCENTILE_CUTOFF = 1e-6

# How far a bound may lie beyond the nominal value, in the nominal value's units,
# before it counts as not bracketing it, so a bound equal to it passes
BOUND_TOLERANCE = 1e-12


def clip_percentiles(percentiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return percentiles clipped to ``[PERCENTILE_CUTOFF, 1 - PERCENTILE_CUTOFF]``.

    Only a distribution with an unbounded tail needs this, as the percentiles 0 and
    1 give it an infinite value; for now that is only ``Normal``. A bounded
    distribution, such as ``Uniform``, takes its percentiles as they are.
    """
    clipped: NDArray[np.float64] = np.clip(
        np.asarray(percentiles, dtype=float), PERCENTILE_CUTOFF, 1.0 - PERCENTILE_CUTOFF
    )
    return clipped


class Parameter(NamedTuple):
    """One of a distribution's parameters, in its nominal value's units."""

    value: float
    text: str  # As the campaign file writes it, such as "2 cm"


class ParameterError(ValueError):
    """Error in a distribution's parameters, naming the one at fault if one is."""

    def __init__(self, message: str, key: str | None = None) -> None:
        """Keep the message, and the key of the parameter at fault, if any."""
        super().__init__(message)
        self.key = key


# Each distribution class, by the name a campaign writes; defining a subclass
# of Distribution adds it here
DISTRIBUTIONS: dict[str, type["Distribution"]] = {}


class Distribution(ABC):
    """Base of the distributions a dispersion may name with its ``distribution``.

    A subclass lists the parameters it takes, builds itself from them and
    draws values at percentiles. It is a frozen dataclass of its own numbers.
    Defining one checks its class variables and adds it to ``DISTRIBUTIONS``,
    so a subclass that is itself a base, without them, is not supported.
    """

    # The campaign keys it requires and may take besides ``distribution``, and
    # how an error says what it takes
    REQUIRED: ClassVar[tuple[str, ...]]
    OPTIONAL: ClassVar[tuple[str, ...]]
    USAGE: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's class variables and register it by name.

        Raises:
            TypeError: If ``REQUIRED``, ``OPTIONAL`` or ``USAGE`` is missing, if
                the first two are not tuples of key names or ``USAGE`` is not
                text, if a key is in both ``REQUIRED`` and ``OPTIONAL``, if an
                abstract method is left undefined, or if another distribution
                has the same name.
        """
        super().__init_subclass__(**kwargs)
        for name in ("REQUIRED", "OPTIONAL", "USAGE"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must define {name}")
        for name in ("REQUIRED", "OPTIONAL"):
            keys = getattr(cls, name)
            if not isinstance(keys, tuple) or not all(isinstance(k, str) for k in keys):
                raise TypeError(f"{cls.__name__}.{name} must be a tuple of key names")
        if not isinstance(cls.USAGE, str) or not cls.USAGE:
            raise TypeError(f"{cls.__name__}.USAGE must be text")
        both = set(cls.REQUIRED) & set(cls.OPTIONAL)
        if both:
            raise TypeError(
                f"{cls.__name__} lists {sorted(both)} as both required and optional"
            )
        check_abstract_members(cls)
        if cls.__name__ in DISTRIBUTIONS:
            other = DISTRIBUTIONS[cls.__name__]
            raise TypeError(
                f"two distributions are named {cls.__name__}: "
                f"{other.__module__} and {cls.__module__}"
            )
        DISTRIBUTIONS[cls.__name__] = cls

    @classmethod
    @abstractmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Args:
            parameters (Mapping[str, Parameter]): The keys of ``REQUIRED`` and
                ``OPTIONAL`` the campaign gave, in the nominal value's units.
            nominal (Nominal): The value it disperses.

        Returns:
            Self: The distribution.

        Raises:
            ParameterError: If the parameters do not describe one, naming the
                one at fault if there is one.
        """

    @abstractmethod
    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile (the inverse CDF)."""

    @abstractmethod
    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "normal, sigma 0.25 kg"."""


@dataclass(frozen=True)
class Normal(Distribution):
    """Normal distribution about the dispersion's nominal value.

    ``low`` and ``high``, where given, truncate it on either side of the
    nominal value.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("sigma",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ("low", "high")
    USAGE: ClassVar[str] = (
        "a normal dispersion takes sigma, and optionally low and high"
    )

    sigma: float
    low: float | None = None
    high: float | None = None

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Args:
            parameters (Mapping[str, Parameter]): sigma, and any of low and
                high, in the nominal value's units.
            nominal (Nominal): The value it disperses.

        Returns:
            Normal: The distribution.

        Raises:
            ParameterError: If sigma is negative, or low and high do not lie on
                either side of the nominal value.
        """
        sigma = parameters["sigma"]
        low, high = parameters.get("low"), parameters.get("high")
        if sigma.value < 0.0:
            raise ParameterError(f"expected 0 or more, got {sigma.text}", "sigma")
        side = "low and high truncate a normal distribution on either side of it"
        if low is not None and _above(low.value, nominal.value):
            raise ParameterError(
                f"{low.text} is above the nominal value, {nominal.text}; {side}", "low"
            )
        if high is not None and _above(nominal.value, high.value):
            raise ParameterError(
                f"{high.text} is below the nominal value, {nominal.text}; {side}",
                "high",
            )
        _check_order(low, high)
        return cls(
            sigma.value,
            None if low is None else low.value,
            None if high is None else high.value,
        )

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile, truncated by low and high."""
        # A normal distribution is unbounded, and a truncated one with only one
        # bound still is on the other side, so percentiles 0 and 1 would give an
        # infinite value
        clipped = clip_percentiles(percentiles)
        if self.sigma == 0.0:
            return np.full(clipped.shape, nominal)
        if self.low is None and self.high is None:
            return np.asarray(
                stats.norm.ppf(clipped, loc=nominal, scale=self.sigma), dtype=float
            )
        lower = -np.inf if self.low is None else (self.low - nominal) / self.sigma
        upper = np.inf if self.high is None else (self.high - nominal) / self.sigma
        return np.asarray(
            stats.truncnorm.ppf(clipped, lower, upper, loc=nominal, scale=self.sigma),
            dtype=float,
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "normal, sigma 0.25 kg"."""
        parts = [f"normal, sigma {quantity_text(self.sigma, units)}"]
        if self.low is not None:
            parts.append(f"low {quantity_text(self.low, units)}")
        if self.high is not None:
            parts.append(f"high {quantity_text(self.high, units)}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform distribution from ``low`` to ``high``."""

    REQUIRED: ClassVar[tuple[str, ...]] = ()
    OPTIONAL: ClassVar[tuple[str, ...]] = ("half_width", "low", "high")
    USAGE: ClassVar[str] = "a uniform dispersion takes half_width, or low and high"

    low: float
    high: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution of half_width about the nominal, or low to high.

        Args:
            parameters (Mapping[str, Parameter]): half_width, or low and high,
                in the nominal value's units.
            nominal (Nominal): The value it disperses.

        Returns:
            Uniform: The distribution.

        Raises:
            ParameterError: If it has both forms or neither, half_width is
                negative, or low is not below high.
        """
        half_width = parameters.get("half_width")
        low, high = parameters.get("low"), parameters.get("high")
        if half_width is not None:
            if low is not None or high is not None:
                raise ParameterError(
                    f"give half_width, or low and high, not both; {cls.USAGE}"
                )
            if half_width.value < 0.0:
                raise ParameterError(
                    f"expected 0 or more, got {half_width.text}", "half_width"
                )
            return cls(
                nominal.value - half_width.value, nominal.value + half_width.value
            )
        if low is None and high is None:
            raise ParameterError(f"missing half_width, or low and high; {cls.USAGE}")
        if low is None or high is None:
            raise ParameterError(
                f"missing {'low' if low is None else 'high'}; {cls.USAGE}"
            )
        _check_order(low, high)
        side = "low and high must lie on either side of it"
        if _above(low.value, nominal.value):
            raise ParameterError(
                f"{low.text} is above the nominal value, {nominal.text}; {side}", "low"
            )
        if _above(nominal.value, high.value):
            raise ParameterError(
                f"{high.text} is below the nominal value, {nominal.text}; {side}",
                "high",
            )
        return cls(low.value, high.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile; the bounds are absolute."""
        del nominal
        # Bounded, so percentiles 0 and 1 give low and high and need no clip
        return self.low + np.asarray(percentiles, dtype=float) * (self.high - self.low)

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "uniform from 0 to 0.2"."""
        return f"uniform from {self.low:g} to {quantity_text(self.high, units)}"


def quantity_text(value: float, units: str) -> str:
    """Return a number with its units, such as "0.25 kg", or bare for none."""
    return f"{value:g} {units}".rstrip()


def _above(first: float, second: float) -> bool:
    """Return whether one number is above another by more than rounding."""
    return first > second and not math.isclose(first, second, abs_tol=BOUND_TOLERANCE)


def _check_order(low: Parameter | None, high: Parameter | None) -> None:
    """Check low is below high where both are given.

    Raises:
        ParameterError: If it is not.
    """
    if low is not None and high is not None and not low.value < high.value:
        raise ParameterError(f"low ({low.text}) must be below high ({high.text})")
