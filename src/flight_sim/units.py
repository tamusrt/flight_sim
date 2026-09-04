"""Canonical Pint unit registry, quantity types, and unit-checked dataclass base.

Pint quantities may only be combined when they originate from the same
``UnitRegistry``, so every module imports ``ureg`` from here rather than
building a registry of its own.

The ``scalar``/``vector``/``zero_vector`` constructors exist because Pint's own
API is largely untyped. They are the single place where Pint's ``Any`` values
are pinned to a concrete type, which is what lets the rest of the codebase
type-check under strict mypy.
"""

from dataclasses import MISSING, fields
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from pint import DimensionalityError, Quantity, UnitRegistry
from pint.facets.plain import PlainUnit
from pint.util import UnitsContainer

ureg: UnitRegistry[Any] = UnitRegistry()

# A single measurement, such as a mass in kilograms.
Scalar: TypeAlias = Quantity[float]

# A three-component measurement, such as a position in metres.
Vector: TypeAlias = Quantity[np.ndarray]


def scalar(magnitude: float, units: str) -> Scalar:
    """Build a scalar quantity from a magnitude and a unit string.

    Args:
        magnitude (float): Numeric value of the measurement.
        units (str): Pint unit expression, such as "kg" or "m/s**2".

    Returns:
        Scalar: The magnitude tagged with the given units.
    """
    result: Scalar = ureg.Quantity(magnitude, units)
    return result


def vector(components: tuple[float, float, float], units: str) -> Vector:
    """Build a three-dimensional vector quantity.

    Args:
        components (tuple[float, float, float]): The x, y and z components.
        units (str): Pint unit expression applied to every component.

    Returns:
        Vector: The components tagged with the given units.
    """
    result: Vector = ureg.Quantity(np.array(components, dtype=float), units)
    return result


def zero_vector(units: str) -> Vector:
    """Build a zero-valued three-dimensional vector quantity.

    Args:
        units (str): Pint unit expression applied to every component.

    Returns:
        Vector: A vector of three zeros in the given units.
    """
    return vector((0.0, 0.0, 0.0), units)


class _QuantityField(NamedTuple):
    """A quantity-valued dataclass field and the units its default declares."""

    name: str
    units: PlainUnit
    dimensionality: UnitsContainer


# Cached per class: defaults cannot change at runtime, and the lookup runs on
# every state construction inside the integration loop.
_EXPECTED: dict[type[Any], tuple[_QuantityField, ...]] = {}


def _expected_fields(cls: type[Any]) -> tuple[_QuantityField, ...]:
    """Read the units each quantity field declares through its default factory.

    Results are cached per class, since the lookup runs on every construction
    inside the integration loop.

    Args:
        cls (type[Any]): Dataclass to inspect.

    Returns:
        tuple[_QuantityField, ...]: One entry per field whose default factory
            produces a quantity.
    """
    cached = _EXPECTED.get(cls)
    if cached is not None:
        return cached

    expected = []
    for quantity_field in fields(cls):
        if quantity_field.default_factory is MISSING:
            continue
        default = quantity_field.default_factory()
        if isinstance(default, Quantity):
            expected.append(
                _QuantityField(
                    quantity_field.name, default.units, default.dimensionality
                )
            )

    _EXPECTED[cls] = tuple(expected)
    return _EXPECTED[cls]


class UnitChecked:
    """Base class for dataclasses whose fields hold physical quantities.

    A field's default declares the dimensionality that field accepts::

        position: Vector = field(default_factory=lambda: zero_vector("m"))

    That field then takes a position in any unit of length and rejects anything
    else. Subclasses need no ``__post_init__`` of their own.

    A field is checked only when its ``default_factory`` produces a quantity,
    so orientations, nested dataclasses, and fields without defaults are left
    alone. A subclass that defines ``__post_init__`` must call
    ``super().__post_init__()`` to keep the checking.
    """

    def __post_init__(self) -> None:
        """Check every quantity field against the units its default declares.

        Raises:
            DimensionalityError: If a field holds a quantity whose dimensionality
                differs from that of the field's default.
        """
        cls: type[Any] = type(self)
        for expected in _expected_fields(cls):
            actual: Quantity[Any] = getattr(self, expected.name)
            if actual.dimensionality != expected.dimensionality:
                raise DimensionalityError(
                    actual.units,
                    expected.units,
                    str(actual.dimensionality),
                    str(expected.dimensionality),
                    extra_msg=f" (assigned to {cls.__name__}.{expected.name})",
                )
