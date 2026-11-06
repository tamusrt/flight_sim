"""Dispersions of a campaign: what each changes in the base sim, and how.

A campaign names a dispersion class by the name of a ``BaseDispersion``
subclass in this module, which ``DISPERSION_CLASSES`` lists. Each reads its
own table of the campaign file, draws from the uniform numbers the sampler
gives it, and records what it changes in a ``Trial``. StandardDispersion, the
default, draws one number of the base sim from a distribution: a class in
``Distribution``, found by its ``DIST_TYPE`` in ``DISTRIBUTIONS``, which lists
and checks its own parameters, so a new distribution is one new class here,
added to ``Distribution``. A standard dispersion's numbers are all in the
units its nominal value is written in.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, ClassVar, NamedTuple, Self, TypeAlias, get_args

import numpy as np
from pint import DimensionalityError, OffsetUnitCalculusError

from edith.errors import EdithConfigError, check_keys, quoted_key_path
from edith.targets import Nominal, find_target
from flight_sim.config import InputSource
from flight_sim.units import UNREADABLE_UNITS, convert, parse_quantity

__all__ = [
    "DISPERSION_CLASSES",
    "DISTRIBUTIONS",
    "STANDARD_DISPERSION",
    "BaseDispersion",
    "Distribution",
    "Normal",
    "Parameter",
    "ParameterError",
    "StandardDispersion",
    "Trial",
    "Uniform",
]


@dataclass
class Trial:
    """One trial's changes to the base sim, which the runner writes out."""

    folder: Path  # The trial's run directory, for files a dispersion writes

    # Key path of each value to change, and the value, as
    # InputSource.table_with takes them
    values: dict[str, Any] = field(default_factory=dict)


class BaseDispersion(ABC):
    """Base of the dispersion classes a campaign may name.

    A campaign's ``dispersion = "<name>"`` names a subclass of this in this
    module, such as StandardDispersion, which reads the rest of its table.
    Each trial then gives it ``DRAWS`` uniform numbers in (0, 1) from the
    sampler, and it records what it changes.
    """

    DRAWS: ClassVar[int] = 1  # Uniform numbers it draws from, per trial
    key: str  # Key path of what it changes in the base sim, which seeds its draws

    @classmethod
    @abstractmethod
    def from_table(
        cls, key: str, table: Mapping[str, object], source: InputSource, where: str
    ) -> Self:
        """Read the dispersion a campaign table describes.

        Args:
            key (str): The table's key, naming what the dispersion changes in
                the base sim, such as "properties.dry_mass_properties.mass".
            table (Mapping[str, object]): The table, ``dispersion`` included.
            source (InputSource): The base sim's resolved input.
            where (str): Key path of the table, for messages.

        Returns:
            Self: The dispersion.

        Raises:
            EdithConfigError: If the table does not describe one, naming the
                key at fault.
        """

    @abstractmethod
    def apply(self, u: np.ndarray, trial: Trial) -> None:
        """Draw from ``u`` and record the change in ``trial``.

        Args:
            u (np.ndarray): ``DRAWS`` uniform numbers in (0, 1).
            trial (Trial): The trial, which takes the values to change and
                the folder for any files written.
        """


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
    """Normal distribution about ``mean``, the dispersion's nominal value.

    ``low`` and ``high``, where given, truncate it on either side of the mean.
    """

    # How a campaign names it, the parameters it needs and takes, and how
    # errors say so
    DIST_TYPE: ClassVar[str] = "normal"
    REQUIRED: ClassVar[tuple[str, ...]] = ("sigma",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ("low", "high")
    USAGE: ClassVar[str] = (
        "a normal dispersion takes sigma, and optionally low and high"
    )

    mean: float
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
            nominal (Nominal): The value it disperses, its mean.

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
            nominal.value,
            sigma.value,
            None if low is None else low.value,
            None if high is None else high.value,
        )

    def ppf(self, u: float) -> float:
        """Return the value at quantile ``u`` in (0, 1), within low and high."""
        if self.sigma == 0.0:
            return self.mean
        normal = NormalDist(self.mean, self.sigma)
        lowest = 0.0 if self.low is None else normal.cdf(self.low)
        highest = 1.0 if self.high is None else normal.cdf(self.high)
        return normal.inv_cdf(lowest + u * (highest - lowest))


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution from ``low`` to ``high``."""

    # How a campaign names it, the parameters it takes, and how errors say so
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
        return cls(low.value, high.value)

    def ppf(self, u: float) -> float:
        """Return the value at quantile ``u`` in (0, 1)."""
        return self.low + u * (self.high - self.low)


Distribution: TypeAlias = Normal | Uniform

# Each distribution class, by the dist_type a campaign names it with; adding a
# class to Distribution adds it here
DISTRIBUTIONS: dict[str, type[Distribution]] = {
    cls.DIST_TYPE: cls for cls in get_args(Distribution)
}

# The dist_type values a dispersion may take, as messages list them
_DIST_TYPES = " or ".join(f'"{name}"' for name in DISTRIBUTIONS)


@dataclass(frozen=True)
class StandardDispersion(BaseDispersion):
    """Dispersion of one number of the base sim, drawn from a distribution.

    The default dispersion class. The nominal value and the distribution's
    numbers are all in ``units``.
    """

    # Key path of the number in the base sim's resolved input, as
    # InputSource.table_with takes it, such as
    # "properties.dry_mass_properties.mass"
    key: str
    nominal: float  # Value in the base sim
    distribution: Distribution

    # Units of the nominal value as the input writes them, such as "kg"; ""
    # for a plain number
    units: str

    # Whether the input writes the number as a string with its units, such
    # as "17 kg", rather than bare, as in an array's value list
    quoted: bool

    @classmethod
    def from_table(
        cls, key: str, table: Mapping[str, object], source: InputSource, where: str
    ) -> Self:
        """Read the dispersion of one number of the base sim from a distribution.

        Raises:
            EdithConfigError: If the key names no number of the base sim, or
                the table does not describe a distribution of it.
        """
        key_path, nominal = find_target(source, key, where)
        distribution_class = DISTRIBUTIONS[_dist_type(table, where)]
        check_keys(
            table,
            where,
            ("dist_type", *distribution_class.REQUIRED),
            ("dispersion", *distribution_class.OPTIONAL),
            distribution_class.USAGE,
        )
        parameters = {
            name: _number(table[name], quoted_key_path(where, name), nominal)
            for name in (*distribution_class.REQUIRED, *distribution_class.OPTIONAL)
            if name in table
        }
        try:
            distribution = distribution_class.from_parameters(parameters, nominal)
        except ParameterError as error:
            place = where if error.key is None else quoted_key_path(where, error.key)
            raise EdithConfigError(f"{place}: {error}") from error
        return cls(key_path, nominal.value, distribution, nominal.units, nominal.quoted)

    def apply(self, u: np.ndarray, trial: Trial) -> None:
        """Draw the number from ``u[0]`` and record it in the input's form."""
        value = self.distribution.ppf(float(u[0]))
        trial.values[self.key] = f"{value!r} {self.units}" if self.quoted else value


# The default dispersion class's name, as a campaign file writes it
STANDARD_DISPERSION = StandardDispersion.__name__

# Each dispersion class, by the name a campaign writes; adding a subclass of
# BaseDispersion adds it here. Subclasses are concrete, whatever mypy infers.
DISPERSION_CLASSES: dict[str, type[BaseDispersion]] = {
    cls.__name__: cls  # type: ignore[type-abstract]
    for cls in BaseDispersion.__subclasses__()
}


def _dist_type(table: Mapping[str, object], where: str) -> str:
    """Return a dispersion's distribution type, a key of ``DISTRIBUTIONS``.

    Raises:
        EdithConfigError: If it is missing, or names no distribution EDITH has.
    """
    if "dist_type" not in table:
        raise EdithConfigError(f"{where}: missing dist_type, {_DIST_TYPES}")
    dist_type = table["dist_type"]
    if not isinstance(dist_type, str) or dist_type not in DISTRIBUTIONS:
        raise EdithConfigError(
            f"{quoted_key_path(where, 'dist_type')}: expected {_DIST_TYPES}, "
            f"got {dist_type!r}"
        )
    return dist_type


def _number(value: object, where: str, nominal: Nominal) -> Parameter:
    """Return one of a dispersion's numbers, converted to its nominal value's units.

    A nominal value with units, such as "17 kg", takes a number with units of
    the same kind; a plain number takes a plain number. A nominal value in
    units with an offset, such as degC, takes numbers in those units only.

    Raises:
        EdithConfigError: If the value is not a finite number in that form, or
            its units do not convert to the nominal value's.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = _finite(float(value), where, value)
        if nominal.units:
            raise EdithConfigError(
                f"{where}: needs units, since the nominal value is "
                f'{nominal.text}; write "{value:g} {nominal.units}"'
            )
        return Parameter(number, f"{value:g}")
    text = value.strip() if isinstance(value, str) else ""
    quantity = parse_quantity(text)
    if quantity is None:
        example = f'"1 {nominal.units}"' if nominal.units else "0.05"
        raise EdithConfigError(
            f"{where}: expected a number such as {example}, got {value!r}"
        )
    number = _finite(quantity[0], where, value)
    if not nominal.units:
        # No number is suggested: "20 percent" could mean 0.2, or 20 % of
        # the nominal value
        raise EdithConfigError(
            f"{where}: takes a plain number, since the nominal value is "
            f"{nominal.text}; got {value!r}"
        )
    try:
        converted = convert(number, quantity[1], nominal.units)
    except DimensionalityError as error:
        raise EdithConfigError(
            f'{where}: "{text}" does not convert to {nominal.units}, the units '
            f"of the nominal value, {nominal.text}"
        ) from error
    except OffsetUnitCalculusError as error:
        raise EdithConfigError(
            f"{where}: write {text!r} in {nominal.units}, the units of the nominal "
            f"value, {nominal.text}, since a spread and a value convert differently"
        ) from error
    except UNREADABLE_UNITS as error:
        raise EdithConfigError(f"{where}: cannot read the units of {text!r}") from error
    return Parameter(_finite(converted, where, value), text)


def _finite(number: float, where: str, value: object) -> float:
    """Return a number, checking it is finite.

    Raises:
        EdithConfigError: If it is infinite or NaN.
    """
    if not math.isfinite(number):
        raise EdithConfigError(f"{where}: expected a finite number, got {value!r}")
    return number


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
