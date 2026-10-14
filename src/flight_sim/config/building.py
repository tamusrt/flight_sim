"""Build objects from tables read from TOML input files.

Each table builds one object, and its keys are the keyword arguments of the
callable its ``type`` names, or of the class its field's type hint names.
"""

import dataclasses
import importlib
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from pint import PintError, Quantity

from flight_sim.config.defaults import key_path, record_defaults
from flight_sim.config.errors import ConfigError
from flight_sim.config.toml_files import origin, plain, source_path, source_table
from flight_sim.units import parse_units, scalar, ureg

# Built values of these types are never filled into another table's arguments
_PLAIN_TYPES = (str, int, float, bool)


def build(
    value: Any,
    hint: Any,
    where: str = "",
    defaults: dict[str, Any] | None = None,
) -> Any:
    """Build an object from a table read by ``load_toml``.

    Args:
        value (Any): Table, list or value to build.
        hint (Any): Type the result must fit, which picks the class of a
            table without a ``type``.
        where (str, optional): Key path of the value, for error messages.
            Defaults to the top level.
        defaults (dict[str, Any] | None, optional): Gets the input form of
            each argument left to its default, by key path. Defaults to none.

    Returns:
        Any: The built object.

    Raises:
        ConfigError: If the value does not describe an object of the hinted
            type, or building it fails.
    """
    return _build(value, hint, where, {}, {} if defaults is None else defaults)


def _members(hint: Any) -> list[Any]:
    """Return the types a hint allows, without annotations or None."""
    if get_origin(hint) is Annotated:
        return _members(get_args(hint)[0])
    if get_origin(hint) in (Union, UnionType):
        return [member for arg in get_args(hint) for member in _members(arg)]
    if hint is NoneType:
        return []
    return [hint]


def _is_quantity(hint: Any) -> bool:
    """Return whether a type is a Pint quantity."""
    return hint is Quantity or get_origin(hint) is Quantity


def _is_buildable(hint: Any) -> bool:
    """Return whether a type is a class a table can build without a ``type``."""
    return (
        isinstance(hint, type)
        and hint.__module__ != "builtins"
        and hint not in (Path, np.ndarray)
        and not _is_quantity(hint)
        and not inspect.isabstract(hint)
    )


def _build(
    value: Any,
    hint: Any,
    where: str,
    context: Mapping[str, Any],
    defaults: dict[str, Any],
) -> Any:
    """Build a value to fit a type hint.

    Args:
        value (Any): Table, list or value read from TOML.
        hint (Any): Type the result must fit.
        where (str): Key path of the value, for error messages.
        context (Mapping[str, Any]): Objects already built in this table and
            the tables enclosing it, by name.
        defaults (dict[str, Any]): Gets the input form of each argument left
            to its default, by key path.

    Returns:
        Any: The built value.

    Raises:
        ConfigError: If the value does not fit the hint.
    """
    members = _members(hint)
    if isinstance(value, dict):
        return _build_table(value, members, where, context, defaults)
    if isinstance(value, list):
        return _build_list(value, members, where, context, defaults)
    return _build_value(value, members, where)


def _build_value(value: Any, members: list[Any], where: str) -> Any:
    """Build a string or number as a path, quantity or plain value.

    Raises:
        ConfigError: If a number is given for a quantity.
    """
    quantity = any(_is_quantity(member) for member in members)
    if isinstance(value, str):
        if Path in members:
            return source_path(value, where)
        if quantity:
            return _quantity(value, where)
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if number and quantity and float not in members and int not in members:
        raise ConfigError(f'{where}: {value} needs units, such as "{value} m"')
    expected = _plain_mismatch(plain(value), members)
    if expected is not None:
        raise ConfigError(f"{where}: expected {expected}, got {plain(value)!r}")
    if isinstance(value, int) and not isinstance(value, bool) and float in members:
        return float(value)
    return plain(value)


# How each plain type is named in error messages
_PLAIN_NAMES = {
    bool: "true or false",
    int: "an integer",
    float: "a number",
    str: "a string",
}


def _plain_mismatch(value: Any, members: list[Any]) -> str | None:
    """Return what a hint of only plain types and literals expected, if not value.

    Returns None when the value fits, or when the hint allows anything else.
    """
    if not members or not all(
        member in _PLAIN_NAMES or get_origin(member) is Literal for member in members
    ):
        return None
    for member in members:
        if get_origin(member) is Literal:
            if any(
                value == option and type(value) is type(option)
                for option in get_args(member)
            ):
                return None
        elif member is bool:
            if isinstance(value, bool):
                return None
        elif isinstance(value, bool):
            continue
        elif (member is float and isinstance(value, int | float)) or isinstance(
            value, member
        ):
            return None
    names = []
    for member in members:
        if get_origin(member) is Literal:
            options = ", ".join(repr(option) for option in get_args(member))
            names.append(f"one of {options}")
        else:
            names.append(_PLAIN_NAMES[member])
    return " or ".join(names)


def _build_table(
    table: dict[str, Any],
    members: list[Any],
    where: str,
    context: Mapping[str, Any],
    defaults: dict[str, Any],
) -> Any:
    """Build the object a table describes.

    Raises:
        ConfigError: If the table names nothing to build, or building fails.
    """
    if "ref" in table:
        if len(table) > 1:
            raise ConfigError(f"{where}: a table with ref takes no other keys")
        return _resolve(table["ref"], where)
    if "type" in table:
        target = _resolve(table["type"], where)
        if not callable(target):
            raise ConfigError(f"{where}: {table['type']} cannot be called")
        arguments = source_table(
            {key: item for key, item in table.items() if key != "type"},
            getattr(table, "files", ()),
        )
        return _call(target, arguments, where, context, defaults)
    if any(_is_quantity(member) for member in members) and set(table) <= {
        "value",
        "diag",
        "units",
    }:
        return _array(table, where)
    classes = [member for member in members if _is_buildable(member)]
    if len(classes) == 1:
        return _call(classes[0], table, where, context, defaults)
    abstract = [m for m in members if inspect.isabstract(m)]
    if not abstract:
        raise ConfigError(
            f'{origin(where, table)}: the table needs type = "dotted.path"'
        )
    options = [
        f'"{option.__module__}.{option.__qualname__}"'
        for member in abstract
        for option in _concrete_subclasses(member)
    ]
    names = " or ".join(member.__name__ for member in abstract)
    message = f"{origin(where, table)}: {names} is abstract, so the table needs a type"
    if options:
        message += f", such as type = {' or '.join(options)}"
    raise ConfigError(message)


def _concrete_subclasses(cls: type) -> list[type]:
    """Return the imported subclasses of a class that are not abstract."""
    found = []
    for subclass in cls.__subclasses__():
        if not inspect.isabstract(subclass):
            found.append(subclass)
        found += _concrete_subclasses(subclass)
    return list(dict.fromkeys(found))


def _build_list(
    items: list[Any],
    members: list[Any],
    where: str,
    context: Mapping[str, Any],
    defaults: dict[str, Any],
) -> Any:
    """Build a list as an array or as a sequence of built items."""
    if np.ndarray in members:
        return np.asarray(plain(items), dtype=float)
    for member in members:
        container = get_origin(member)
        if container in (list, tuple, Sequence):
            args = get_args(member)
            item_hint = args[0] if args else Any
            built = [
                _build(item, item_hint, f"{where}[{index}]", context, defaults)
                for index, item in enumerate(items)
            ]
            return built if container is list else tuple(built)
    return [
        _build(item, Any, f"{where}[{index}]", context, defaults)
        for index, item in enumerate(items)
    ]


def _quantity(text: str, where: str) -> Any:
    """Read a scalar quantity such as "17 ft".

    Raises:
        ConfigError: If the text is not a quantity with units.
    """
    try:
        quantity: Any = ureg.Quantity(str(text))
    except (PintError, AttributeError, SyntaxError, TypeError, ValueError) as error:
        raise ConfigError(f"{where}: cannot read {text!r} as a quantity") from error
    if not isinstance(quantity, Quantity) or quantity.unitless:
        raise ConfigError(f'{where}: {text!r} needs units, such as "{text} m"')
    return scalar(float(quantity.magnitude), str(quantity.units))


def _array(table: dict[str, Any], where: str) -> Any:
    """Read an array quantity from a ``value`` or ``diag`` table with ``units``.

    Raises:
        ConfigError: If the table lacks units or holds both or neither array.
    """
    units = table.get("units")
    if not isinstance(units, str):
        raise ConfigError(f"{where}: an array quantity needs units")
    if ("value" in table) == ("diag" in table):
        raise ConfigError(f"{where}: an array quantity needs one of value or diag")
    components = np.asarray(plain(table.get("value", table.get("diag"))), float)
    if "diag" in table:
        components = np.diag(components)
    try:
        return ureg.Quantity(components, parse_units(str(units)))
    except (PintError, AttributeError, SyntaxError, TypeError, ValueError) as error:
        raise ConfigError(f"{where}: cannot read units {units!r}") from error


def _resolve(dotted: Any, where: str) -> Any:
    """Import the object a dotted path names, such as a class or classmethod.

    Raises:
        ConfigError: If no module or attribute has that name.
    """
    if not isinstance(dotted, str):
        raise ConfigError(f"{where}: expected a dotted path, got {dotted!r}")
    parts = str(dotted).split(".")
    for count in range(len(parts), 0, -1):
        module_name = ".".join(parts[:count])
        try:
            target: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if module_name != missing and not module_name.startswith(missing + "."):
                raise
            continue
        for attribute in parts[count:]:
            try:
                target = getattr(target, attribute)
            except AttributeError as error:
                raise ConfigError(f"{where}: {dotted} does not exist") from error
        return target
    raise ConfigError(f"{where}: no module in {dotted}")


def _hints(target: Any) -> dict[str, Any]:
    """Return the type hints of a callable's parameters, or of a class's fields."""
    if isinstance(target, type) and not dataclasses.is_dataclass(target):
        target = cast(Any, target).__init__
    try:
        return get_type_hints(target, include_extras=True)
    except (NameError, TypeError):
        return {}


def _call(
    target: Callable[..., Any],
    table: dict[str, Any],
    where: str,
    context: Mapping[str, Any],
    defaults: dict[str, Any],
) -> Any:
    """Call a callable with a table's keys as its keyword arguments.

    Arguments are built in the callable's parameter order, each able to use
    the ones before it. Keys the callable does not name pass through its
    ``**kwargs``, built to fit the fields of the class it is a classmethod of
    or returns. Arguments left to their defaults go into ``defaults``.

    Raises:
        ConfigError: If a key matches no parameter, a required argument is
            missing, or the call fails.
    """
    parameters = inspect.signature(target).parameters
    hints = _hints(target)
    named = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    owner = _kwargs_owner(target, parameters, hints)
    usage = _usage(target, parameters, named, owner)
    where_text = origin(where, table)
    extra = [key for key in table if key not in named]
    if extra and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        raise ConfigError(f"{where_text}: unknown keys {extra}; {usage}")
    extra_hints = _hints(owner) if owner is not None else {}

    local = dict(context)
    arguments: dict[str, Any] = {}
    for name in (*named, *extra):
        if name in table:
            hint = hints.get(name, Any) if name in named else extra_hints.get(name, Any)
            child = key_path(where, name)
            arguments[name] = _build(table[name], hint, child, local, defaults)
            if not isinstance(arguments[name], _PLAIN_TYPES):
                local[name] = arguments[name]
        elif parameters[name].default is inspect.Parameter.empty and name in context:
            arguments[name] = context[name]
    missing = [
        name
        for name in named
        if name not in arguments and parameters[name].default is inspect.Parameter.empty
    ]
    if missing:
        raise ConfigError(f"{where_text}: missing {', '.join(missing)}; {usage}")
    result = _invoke(target, arguments, where_text, usage)
    record_defaults(
        target,
        parameters,
        owner=owner,
        table=table,
        result=result,
        where=where,
        defaults=defaults,
    )
    return result


def _kwargs_owner(
    target: Any, parameters: Mapping[str, inspect.Parameter], hints: Mapping[str, Any]
) -> type | None:
    """Return the class whose fields a callable's ``**kwargs`` take, if any.

    That is the class a classmethod belongs to, or else the class it returns.
    """
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    owner = getattr(target, "__self__", None)
    if not isinstance(owner, type):
        owner = hints.get("return")
    return owner if isinstance(owner, type) else None


def _invoke(
    target: Callable[..., Any], arguments: dict[str, Any], where: str, usage: str
) -> Any:
    """Call a callable, turning its errors into ``ConfigError``.

    Raises:
        ConfigError: If the call fails, naming any arguments it was missing.
    """
    try:
        return target(**arguments)
    except TypeError as error:
        # Raised by the class a function builds, for a field only the table gives
        match = _MISSING_ARGUMENTS.search(str(error))
        if match is None:
            raise ConfigError(f"{where}: {error}") from error
        names = ", ".join(re.findall(r"'(\w+)'", match.group(1)))
        raise ConfigError(f"{where}: missing {names}; {usage}") from error
    except (ValueError, OSError) as error:
        raise ConfigError(f"{where}: {error}") from error


_MISSING_ARGUMENTS = re.compile(
    r"missing \d+ required (?:positional|keyword-only) arguments?: (.*)"
)


def _usage(
    target: Any,
    parameters: Mapping[str, inspect.Parameter],
    named: Sequence[str],
    owner: type | None,
) -> str:
    """Describe the keys a callable needs and takes, for error messages."""
    required = [n for n in named if parameters[n].default is inspect.Parameter.empty]
    optional = [n for n in named if n not in required]
    if owner is not None:
        optional.append(f"other {owner.__name__} fields")
    text = f"{getattr(target, '__qualname__', target)} needs "
    text += ", ".join(required) if required else "nothing"
    return text + (f" and takes {', '.join(optional)}" if optional else "")
