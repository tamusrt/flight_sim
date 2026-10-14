"""Record the arguments a build left to their defaults, in input file form."""

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pint import PintError, Quantity

from flight_sim.config.toml_files import key_path


def record_defaults(
    target: Any,
    parameters: Mapping[str, inspect.Parameter],
    *,
    owner: type | None,
    table: Mapping[str, Any],
    result: Any,
    where: str,
    defaults: dict[str, Any],
) -> None:
    """Put the input form of each argument a call left to its default in ``defaults``.

    Defaults are read off the result where it has them, since dataclass
    factory defaults are not in the signature. A passed-through field of
    ``owner`` counts when the result still holds that field's default.
    """
    for name, parameter in parameters.items():
        if name in table or parameter.default is inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        value = parameter.default
        if isinstance(target, type) and hasattr(result, name):
            value = getattr(result, name)
        defaults[key_path(where, name)] = _input_form(value)
    if owner is None or not dataclasses.is_dataclass(owner):
        return
    if not isinstance(result, owner):
        return
    for owner_field in dataclasses.fields(owner):
        name = owner_field.name
        if not owner_field.init or name in table or name in parameters:
            continue
        if owner_field.default is not dataclasses.MISSING:
            default = owner_field.default
        elif owner_field.default_factory is not dataclasses.MISSING:
            default = owner_field.default_factory()
        else:
            continue
        if _same(getattr(result, name), default):
            defaults[key_path(where, name)] = _input_form(default)


def _same(first: Any, second: Any) -> bool:
    """Return whether two values are equal, counting incomparable ones unequal."""
    try:
        return bool(np.all(first == second))
    except (PintError, TypeError, ValueError):
        return False


def _input_form(value: Any) -> Any:
    """Return a value as an input file would give it.

    Objects with no input form, and None, are returned as they are.
    """
    if isinstance(value, Quantity):
        return _quantity_form(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_form(value)
    if isinstance(value, type) or inspect.isfunction(value):
        return {"ref": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [_input_form(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _quantity_form(value: Quantity[Any]) -> Any:
    """Return a scalar quantity as a string such as "3.0 m", or an array as a table."""
    units = f"{value.units:~}"
    if np.ndim(value.magnitude) == 0:
        return f"{float(value.magnitude)!r} {units}".strip()
    return {"value": np.asarray(value.magnitude).tolist(), "units": units}


def _dataclass_form(value: Any) -> dict[str, Any]:
    """Return a dataclass instance as a table naming its type and init fields."""
    cls = type(value)
    table: dict[str, Any] = {"type": f"{cls.__module__}.{cls.__qualname__}"}
    for item in dataclasses.fields(value):
        item_value = getattr(value, item.name)
        if item.init and item_value is not None:
            table[item.name] = _input_form(item_value)
    return table
