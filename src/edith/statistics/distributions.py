"""Distributions a dispersion draws from, each with an inverse CDF (``ppf``).

A campaign names a distribution by the name of a ``Distribution`` subclass,
which ``DISTRIBUTIONS`` lists. Defining a subclass registers it, in any module
that is imported, so a new distribution is one new subclass. Numbers are in
the units of the nominal value being dispersed, except the keys a class lists
in ``PLAIN``, such as a shape parameter, which are plain numbers.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray
from pint import DimensionalityError
from scipy import stats  # type: ignore[import-untyped]

from edith.statistics.abstract import check_abstract_members
from edith.targets import Nominal
from flight_sim.units import convert

__all__ = [
    "DISTRIBUTIONS",
    "Beta",
    "Distribution",
    "LogNormal",
    "Normal",
    "Parameter",
    "ParameterError",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
    "Weibull",
    "clip_percentiles",
    "quantity_text",
]

# Percentiles stay this far from 0 and 1, so no value is infinite
PERCENTILE_CUTOFF = 1e-6

# How far a bound may lie beyond the nominal value, in the nominal value's units,
# before it counts as not bracketing it, so a bound equal to it passes
BOUND_TOLERANCE = 1e-12

# Points of the grid a von Mises distribution tabulates its CDF on, and the
# sigmas, 1 / sqrt(kappa) radians, it spans either side of the nominal value
VON_MISES_POINTS = 4097
VON_MISES_SPAN = 10.0


def clip_percentiles(percentiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return percentiles clipped to ``[PERCENTILE_CUTOFF, 1 - PERCENTILE_CUTOFF]``.

    Only a distribution with an unbounded tail needs this, as the percentiles 0 and
    1 give it an infinite value, such as ``Normal``. A bounded distribution, such
    as ``Uniform``, takes its percentiles as they are.
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

    # Those of its keys that take a plain number whatever the nominal value's
    # units, such as a shape parameter
    PLAIN: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the subclass's class variables and register it by name.

        Raises:
            TypeError: If ``REQUIRED``, ``OPTIONAL`` or ``USAGE`` is missing, if
                they and ``PLAIN`` are not tuples of key names or ``USAGE`` is
                not text, if a key is in both ``REQUIRED`` and ``OPTIONAL``, if
                ``PLAIN`` lists a key that is in neither, if an abstract method
                is left undefined, or if another distribution has the same name.
        """
        super().__init_subclass__(**kwargs)
        for name in ("REQUIRED", "OPTIONAL", "USAGE"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must define {name}")
        for name in ("REQUIRED", "OPTIONAL", "PLAIN"):
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
        unknown = set(cls.PLAIN) - {*cls.REQUIRED, *cls.OPTIONAL}
        if unknown:
            raise TypeError(
                f"{cls.__name__} lists {sorted(unknown)} as plain but does not take it"
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


@dataclass(frozen=True)
class StudentT(Distribution):
    """Student's t distribution about the nominal value, heavier-tailed than normal.

    ``scale`` is in the nominal value's units; ``df``, the degrees of freedom,
    is a plain number. As ``df`` grows it tends to ``Normal`` of sigma ``scale``.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("scale", "df")
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a Student's t dispersion takes scale and df"
    PLAIN: ClassVar[tuple[str, ...]] = ("df",)

    scale: float
    df: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If scale is negative or df is not above 0.
        """
        scale, df = parameters["scale"], parameters["df"]
        if scale.value < 0.0:
            raise ParameterError(f"expected 0 or more, got {scale.text}", "scale")
        _check_positive(df, "df")
        return cls(scale.value, df.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile, centred on the nominal value."""
        clipped = clip_percentiles(percentiles)  # Unbounded on both sides
        if self.scale == 0.0:
            return np.full(clipped.shape, nominal)
        return np.asarray(
            stats.t.ppf(clipped, self.df, loc=nominal, scale=self.scale), dtype=float
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "Student's t, scale 1 m, df 4"."""
        return f"Student's t, scale {quantity_text(self.scale, units)}, df {self.df:g}"


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular distribution from ``low`` to ``high``, peaking at the nominal."""

    REQUIRED: ClassVar[tuple[str, ...]] = ("low", "high")
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a triangular dispersion takes low and high"

    low: float
    high: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If low is not below high, or they do not lie on
                either side of the nominal value.
        """
        low, high = parameters["low"], parameters["high"]
        _check_order(low, high)
        _check_bracket(low, high, nominal)
        return cls(low.value, high.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile; the bounds are absolute."""
        width = self.high - self.low
        # The nominal value is the mode, kept in range against rounding
        mode = min(max((nominal - self.low) / width, 0.0), 1.0)
        return np.asarray(
            stats.triang.ppf(percentiles, mode, loc=self.low, scale=width), dtype=float
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "triangular from 0 to 2 m"."""
        return f"triangular from {self.low:g} to {quantity_text(self.high, units)}"


@dataclass(frozen=True)
class Weibull(Distribution):
    """Weibull distribution whose scale is the nominal value.

    The nominal value, which must be above 0, is the 63.2th percentile.
    ``shape`` is a plain number, such as 2 for a Rayleigh-like wind speed.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("shape",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a Weibull dispersion takes shape"
    PLAIN: ClassVar[tuple[str, ...]] = ("shape",)

    shape: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If shape or the nominal value is not above 0.
        """
        _check_positive_nominal(nominal, "Weibull")
        shape = parameters["shape"]
        _check_positive(shape, "shape")
        return cls(shape.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile, the nominal value as the scale."""
        clipped = clip_percentiles(percentiles)  # Unbounded above
        return np.asarray(
            stats.weibull_min.ppf(clipped, self.shape, scale=nominal), dtype=float
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "Weibull, shape 2"."""
        del units
        return f"Weibull, shape {self.shape:g}"


@dataclass(frozen=True)
class VonMises(Distribution):
    """Von Mises distribution of an angle about the nominal value.

    Values lie within half a turn of the nominal value. ``kappa``, a plain
    number, is the concentration, roughly 1 / sigma**2 with sigma in radians.
    The nominal value is an angle, or a plain number taken as radians.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("kappa",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a von Mises dispersion takes kappa"
    PLAIN: ClassVar[tuple[str, ...]] = ("kappa",)

    kappa: float
    radians_per_unit: float = 1.0  # Radians in one of the nominal value's units

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If kappa is not above 0, or the nominal value is not
                an angle.
        """
        kappa = parameters["kappa"]
        _check_positive(kappa, "kappa")
        try:
            radians = convert(1.0, nominal.units, "rad") if nominal.units else 1.0
        except DimensionalityError as error:
            raise ParameterError(
                f"the nominal value, {nominal.text}, is not an angle; {cls.USAGE}"
            ) from error
        return cls(kappa.value, radians)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile, in the nominal value's units.

        scipy inverts the CDF numerically, a few milliseconds a value, so the
        CDF is tabulated on a grid instead and inverted by interpolation,
        within 1e-5 of a sigma of scipy's values. The grid spans
        ``VON_MISES_SPAN`` sigmas either side, or half a turn if that is less,
        beyond which lies under 1e-20 of the chance.
        """
        span = min(math.pi, VON_MISES_SPAN / math.sqrt(self.kappa))
        grid = np.linspace(-span, span, VON_MISES_POINTS)
        cdf = stats.vonmises.cdf(grid, self.kappa)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])  # 0 and 1 at the ends exactly
        radians = np.interp(np.asarray(percentiles, dtype=float), cdf, grid)
        return np.asarray(nominal + radians / self.radians_per_unit, dtype=float)

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "von Mises, kappa 400"."""
        del units
        return f"von Mises, kappa {self.kappa:g}"


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal distribution whose median is the nominal value.

    The nominal value must be above 0. ``sigma``, a plain number, is the
    standard deviation of the value's natural logarithm.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("sigma",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a log-normal dispersion takes sigma, of the logarithm"
    PLAIN: ClassVar[tuple[str, ...]] = ("sigma",)

    sigma: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If sigma is negative or the nominal value is not
                above 0.
        """
        _check_positive_nominal(nominal, "log-normal")
        sigma = parameters["sigma"]
        if sigma.value < 0.0:
            raise ParameterError(f"expected 0 or more, got {sigma.text}", "sigma")
        return cls(sigma.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile, the nominal value as the median."""
        clipped = clip_percentiles(percentiles)  # Unbounded above
        if self.sigma == 0.0:
            return np.full(clipped.shape, nominal)
        return np.asarray(
            stats.lognorm.ppf(clipped, self.sigma, scale=nominal), dtype=float
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "log-normal, sigma 0.1"."""
        del units
        return f"log-normal, sigma {self.sigma:g}"


@dataclass(frozen=True)
class Beta(Distribution):
    """Beta distribution from ``low`` to ``high``, of shape ``alpha`` and ``beta``.

    The bounds are absolute and must lie on either side of the nominal value;
    ``alpha`` and ``beta`` are plain numbers.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = ("alpha", "beta", "low", "high")
    OPTIONAL: ClassVar[tuple[str, ...]] = ()
    USAGE: ClassVar[str] = "a beta dispersion takes alpha, beta, low and high"
    PLAIN: ClassVar[tuple[str, ...]] = ("alpha", "beta")

    alpha: float
    beta: float
    low: float
    high: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Raises:
            ParameterError: If alpha or beta is not above 0, low is not below
                high, or they do not lie on either side of the nominal value.
        """
        alpha, beta = parameters["alpha"], parameters["beta"]
        low, high = parameters["low"], parameters["high"]
        _check_positive(alpha, "alpha")
        _check_positive(beta, "beta")
        _check_order(low, high)
        _check_bracket(low, high, nominal)
        return cls(alpha.value, beta.value, low.value, high.value)

    def ppf(
        self, percentiles: NDArray[np.float64], nominal: float
    ) -> NDArray[np.float64]:
        """Return the value at each percentile; the bounds are absolute."""
        del nominal
        width = self.high - self.low
        return np.asarray(
            stats.beta.ppf(
                percentiles, self.alpha, self.beta, loc=self.low, scale=width
            ),
            dtype=float,
        )

    def describe(self, units: str) -> str:
        """Return the distribution in words, such as "beta 2, 5 from 0 to 1 m"."""
        return (
            f"beta {self.alpha:g}, {self.beta:g} from {self.low:g} to "
            f"{quantity_text(self.high, units)}"
        )


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


def _check_positive(parameter: Parameter, key: str) -> None:
    """Check a parameter is above 0.

    Raises:
        ParameterError: If it is not, naming its key.
    """
    if not parameter.value > 0.0:
        raise ParameterError(f"expected above 0, got {parameter.text}", key)


def _check_positive_nominal(nominal: Nominal, name: str) -> None:
    """Check a nominal value is above 0, as a distribution of positive values needs.

    Raises:
        ParameterError: If it is not.
    """
    if not nominal.value > 0.0:
        raise ParameterError(
            f"a {name} dispersion needs a nominal value above 0, got {nominal.text}"
        )


def _check_bracket(low: Parameter, high: Parameter, nominal: Nominal) -> None:
    """Check low and high lie on either side of the nominal value.

    Raises:
        ParameterError: If either does not, naming it.
    """
    side = "low and high must lie on either side of it"
    if _above(low.value, nominal.value):
        raise ParameterError(
            f"{low.text} is above the nominal value, {nominal.text}; {side}", "low"
        )
    if _above(nominal.value, high.value):
        raise ParameterError(
            f"{high.text} is below the nominal value, {nominal.text}; {side}", "high"
        )
