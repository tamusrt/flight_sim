"""Dispersions of a campaign: what each changes in the base sim, and how.

A campaign names a dispersion class by the name of a ``BaseDispersion``
subclass in this module, which ``DISPERSION_CLASSES`` lists, and each reads
its own table of the campaign file. StandardDispersion, the default, disperses
one number of the base sim with a distribution: a class in ``Distribution``,
found by its ``DIST_TYPE`` in ``DISTRIBUTIONS``. Those live in
``edith.statistics.distributions``, which this module re-exports. A standard
dispersion's numbers are all in the units its nominal value is written in.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from pint import DimensionalityError, OffsetUnitCalculusError

from edith.errors import EdithConfigError, check_keys, quoted_key_path
from edith.statistics.distributions import (
    DISTRIBUTIONS,
    Distribution,
    Normal,
    Parameter,
    ParameterError,
    Uniform,
)
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
    "Uniform",
]


class BaseDispersion(ABC):
    """Base of the dispersion classes a campaign may name.

    A campaign's ``dispersion = "<name>"`` names a subclass of this in this
    module, such as StandardDispersion, which reads the rest of its table.
    """

    key: str  # Key path of what it changes in the base sim

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


# The dist_type values a dispersion may take, as messages list them
_DIST_TYPES = " or ".join(f'"{name}"' for name in DISTRIBUTIONS)


@dataclass(frozen=True)
class StandardDispersion(BaseDispersion):
    """Dispersion of one number of the base sim with a distribution.

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

    @classmethod
    def from_table(
        cls, key: str, table: Mapping[str, object], source: InputSource, where: str
    ) -> Self:
        """Read the dispersion of one number of the base sim with a distribution.

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
        return cls(key_path, nominal.value, distribution, nominal.units)


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
