"""Canonical Pint unit registry, quantity types, and unit-checked dataclass base.

Pint quantities may only be combined when they originate from the same
``UnitRegistry``, so every module imports ``ureg`` from here rather than
building a registry of its own.

The ``scalar``/``vector``/``matrix`` constructors exist because Pint's own
API is largely untyped. They are the single place where Pint's ``Any`` values
are pinned to a concrete type, which is what lets the rest of the codebase
type-check under strict mypy.
"""

from collections.abc import Sequence
from dataclasses import fields
from functools import cache
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from pint import DimensionalityError, Quantity, UnitRegistry
from pint.facets.plain import PlainUnit
from pint.util import UnitsContainer

ureg: UnitRegistry[Any] = UnitRegistry()

# A single measurement, such as a mass in kilograms.
Scalar: TypeAlias = Quantity[float]

# A three-component measurement, such as a position in metres.
Vector: TypeAlias = Quantity[np.ndarray]

# A 3x3 measurement, such as an inertia tensor in kg*m**2.
Matrix: TypeAlias = Quantity[np.ndarray]


@cache
def parse_units(units: str) -> PlainUnit:
    """Parse a unit expression, caching the result.

    Args:
        units (str): Pint unit expression, such as "kg" or "m/s**2".

    Returns:
        PlainUnit: The parsed unit, shared by every caller passing the same string.
    """
    result: PlainUnit = ureg.Unit(units)
    return result


# Units output is shown in: "metric" keeps the SI units quantities are declared
# in, and "imperial" converts those it has a unit for
UnitSystem: TypeAlias = Literal["metric", "imperial"]

# Imperial units, by the dimensionality they show; angles, times and the like
# keep their own
_IMPERIAL = {
    parse_units(units).dimensionality: units
    for units in (
        "ft",
        "ft/s",
        "ft/s**2",
        "lb",
        "lb*ft**2",
        "lbf",
        "ft*lbf",
        "psi",
        "lb/ft**3",
        "degF",
    )
}


def system_units(units: str, system: UnitSystem) -> str:
    """Return the units a unit system shows a quantity declared in ``units`` in."""
    if system == "metric":
        return units
    return _IMPERIAL.get(parse_units(units).dimensionality, units)


def scalar(magnitude: float, units: str) -> Scalar:
    """Build a scalar quantity from a magnitude and a unit string.

    Args:
        magnitude (float): Numeric value of the measurement.
        units (str): Pint unit expression, such as "kg" or "m/s**2".

    Returns:
        Scalar: The magnitude tagged with the given units.
    """
    result: Scalar = ureg.Quantity(magnitude, parse_units(units))
    return result


def vector(components: tuple[float, float, float] | np.ndarray, units: str) -> Vector:
    """Build a three-dimensional vector quantity.

    Args:
        components (tuple[float, float, float] | np.ndarray): The x, y and z
            components, which are copied.
        units (str): Pint unit expression applied to every component.

    Returns:
        Vector: The components tagged with the given units.
    """
    result: Vector = ureg.Quantity(
        np.array(components, dtype=float), parse_units(units)
    )
    return result


def matrix(components: Sequence[Sequence[float]] | np.ndarray, units: str) -> Matrix:
    """Build a 3x3 matrix quantity.

    Args:
        components (Sequence[Sequence[float]] | np.ndarray): The three rows,
            which are copied.
        units (str): Pint unit expression applied to every component.

    Returns:
        Matrix: The components tagged with the given units.
    """
    result: Matrix = ureg.Quantity(
        np.array(components, dtype=float), parse_units(units)
    )
    return result


class _QuantityField(NamedTuple):
    """A quantity-valued dataclass field and the units its annotation declares."""

    name: str
    units: PlainUnit
    dimensionality: UnitsContainer
    in_union: bool  # The field also accepts values that are not quantities


# Cached per class
_EXPECTED: dict[type[Any], tuple[_QuantityField, ...]] = {}


def _expected_fields(cls: type[Any]) -> tuple[_QuantityField, ...]:
    """Read the units each field declares through an ``Annotated`` unit string.

    Args:
        cls (type[Any]): Dataclass to inspect.

    Returns:
        tuple[_QuantityField, ...]: One entry per field annotated with a unit
            string.
    """
    cached = _EXPECTED.get(cls)
    if cached is not None:
        return cached

    hints = get_type_hints(cls, include_extras=True)
    expected = []
    for quantity_field in fields(cls):
        hint = hints[quantity_field.name]
        unit_string = annotated_units(hint)
        if unit_string is not None:
            units = parse_units(unit_string)
            in_union = get_origin(get_args(hint)[0]) in (Union, UnionType)
            expected.append(
                _QuantityField(
                    quantity_field.name, units, units.dimensionality, in_union
                )
            )

    _EXPECTED[cls] = tuple(expected)
    return _EXPECTED[cls]


def annotated_units(hint: Any) -> str | None:
    """Return the unit string of an ``Annotated`` hint, or None without one."""
    if get_origin(hint) is not Annotated:
        return None
    return next((meta for meta in get_args(hint)[1:] if isinstance(meta, str)), None)


class UnitChecked:
    """Base class for dataclasses whose fields hold physical quantities.

    A field declares the units it accepts with a unit string in its
    annotation::

        position: Annotated[Vector, "m"]

    That field then takes a position in any unit of length and rejects anything
    else. Any Pint unit expression works, such as "kg*m**2" or "rad/s**2".
    Subclasses need no ``__post_init__`` of their own.

    A field may also take other types, such as
    ``Annotated[Scalar | EventTime, "s"]``; it is checked only when it holds a
    quantity.

    Fields without a unit annotation, such as orientations and nested
    dataclasses, are left alone, as are annotated fields holding None. A
    subclass that defines ``__post_init__`` must call
    ``super().__post_init__()`` to keep the checking.
    """

    def __post_init__(self) -> None:
        """Check every quantity field against the units its annotation declares.

        Raises:
            DimensionalityError: If a field holds a quantity whose dimensionality
                differs from that of the field's declared units.
        """
        cls: type[Any] = type(self)
        for expected in _expected_fields(cls):
            actual: Quantity[Any] | None = getattr(self, expected.name)
            if actual is None or (
                expected.in_union and not isinstance(actual, Quantity)
            ):
                continue
            if actual.dimensionality != expected.dimensionality:
                raise DimensionalityError(
                    actual.units,
                    expected.units,
                    str(actual.dimensionality),
                    str(expected.dimensionality),
                    extra_msg=f" (assigned to {cls.__name__}.{expected.name})",
                )
