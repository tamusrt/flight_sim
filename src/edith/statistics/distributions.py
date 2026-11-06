"""Distributions a dispersion draws from, each with an inverse CDF (``ppf``).

A new distribution is one new class here, added to ``Distribution``. Numbers are
in the units of the nominal value being dispersed.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Self, TypeAlias, get_args

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

from edith.targets import Nominal

__all__ = [
    "DISTRIBUTIONS",
    "PERCENTILE_EPS",
    "Distribution",
    "Normal",
    "Parameter",
    "ParameterError",
    "Uniform",
    "clip_percentiles",
]

# Percentiles stay this far from 0 and 1, so no value is infinite
PERCENTILE_EPS = 1e-6


def clip_percentiles(percentiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return percentiles clipped to ``[PERCENTILE_EPS, 1 - PERCENTILE_EPS]``."""
    clipped: NDArray[np.float64] = np.clip(
        np.asarray(percentiles, dtype=float), PERCENTILE_EPS, 1.0 - PERCENTILE_EPS
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


@dataclass(frozen=True)
class Normal:
    """Normal distribution about the dispersion's nominal value.

    ``low`` and ``high``, where given, truncate it on either side of the
    nominal value.
    """

    DIST_TYPE: ClassVar[str] = "normal"
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


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution from ``low`` to ``high``."""

    DIST_TYPE: ClassVar[str] = "uniform"
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
        return self.low + clip_percentiles(percentiles) * (self.high - self.low)


Distribution: TypeAlias = Normal | Uniform

# Each distribution class, by the dist_type a campaign names it with; adding a
# class to Distribution adds it here
DISTRIBUTIONS: dict[str, type[Distribution]] = {
    cls.DIST_TYPE: cls for cls in get_args(Distribution)
}


def _above(first: float, second: float) -> bool:
    """Return whether one number is above another by more than rounding."""
    return first > second and not math.isclose(first, second, abs_tol=1e-12)


def _check_order(low: Parameter | None, high: Parameter | None) -> None:
    """Check low is below high where both are given.

    Raises:
        ParameterError: If it is not.
    """
    if low is not None and high is not None and not low.value < high.value:
        raise ParameterError(f"low ({low.text}) must be below high ({high.text})")
