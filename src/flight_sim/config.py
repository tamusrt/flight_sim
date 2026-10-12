"""Build objects from TOML input files, in the format configs/README.md describes.

Each table builds one object, and its keys are the keyword arguments of the
callable its ``type`` names, or of the class its field's type hint names.
"""

import dataclasses
import importlib
import inspect
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from pint import PintError, Quantity

from flight_sim.units import parse_units, scalar, ureg

T = TypeVar("T")

# Built values of these types are never filled into another table's arguments
_PLAIN_TYPES = (str, int, float, bool)


class ConfigError(ValueError):
    """Input file that cannot be read or built."""


class _SourceStr(str):
    """String read from a TOML file, which remembers the file's directory."""

    base: Path

    def __new__(cls, text: str, base: Path) -> "_SourceStr":
        """Tag a string with the directory of the file it came from."""
        result = super().__new__(cls, text)
        result.base = base  # pylint: disable=attribute-defined-outside-init
        return result


def load(path: str | Path, cls: type[T]) -> T:
    """Build an object from a TOML file.

    Args:
        path (str | Path): TOML file whose top-level table builds the object.
        cls (type[T]): Class of the object, used when the file names no
            ``type``.

    Returns:
        T: The built object.

    Raises:
        ConfigError: If the file cannot be read or does not build a ``cls``.
    """
    result = build(load_toml(path), cls)
    if not isinstance(result, cls):
        raise ConfigError(f"{path} builds a {type(result).__name__}, not a {cls}")
    return result


def load_toml(path: str | Path, chain: Sequence[Path] = ()) -> dict[str, Any]:
    """Read a TOML file, with its includes inserted and its base file merged in.

    Args:
        path (str | Path): TOML file to read.
        chain (Sequence[Path], optional): Files already being read, to catch
            files that include themselves. Defaults to none.

    Returns:
        dict[str, Any]: The file's table, its strings remembering the
            directory of the file each came from.

    Raises:
        ConfigError: If a file is not valid TOML or includes itself.
    """
    path = Path(path).resolve()
    if path in chain:
        raise ConfigError(f"{path} includes itself")
    chain = (*chain, path)
    with path.open("rb") as file:
        try:
            raw = tomllib.load(file)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: {error}") from error
    table: dict[str, Any] = _expand(_located(raw, path.parent), chain)
    extends = table.pop("extends", None)
    if extends is not None:
        table = _merge(load_toml(_path(extends, "extends"), chain), table)
    return table


def build(value: Any, hint: Any, where: str = "") -> Any:
    """Build an object from a table read by ``load_toml``.

    Args:
        value (Any): Table, list or value to build.
        hint (Any): Type the result must fit, which picks the class of a
            table without a ``type``.
        where (str, optional): Key path of the value, for error messages.
            Defaults to the top level.

    Returns:
        Any: The built object.

    Raises:
        ConfigError: If the value does not describe an object of the hinted
            type, or building it fails.
    """
    return _build(value, hint, where, {})


def _located(node: Any, base: Path) -> Any:
    """Tag every string in a TOML table with the directory of its file."""
    if isinstance(node, dict):
        return {key: _located(item, base) for key, item in node.items()}
    if isinstance(node, list):
        return [_located(item, base) for item in node]
    if isinstance(node, str):
        return _SourceStr(node, base)
    return node


def _expand(node: Any, chain: Sequence[Path]) -> Any:
    """Insert the file each ``include`` names under the table holding it."""
    if isinstance(node, list):
        return [_expand(item, chain) for item in node]
    if not isinstance(node, dict):
        return node
    table = {key: _expand(item, chain) for key, item in node.items()}
    include = table.pop("include", None)
    if include is None:
        return table
    return _merge(load_toml(_path(include, "include"), chain), table)


def _merge(base: Any, override: Any) -> Any:
    """Return ``base`` with ``override`` on top, merging tables key by key.

    A table naming a different ``type`` or ``ref`` than its base replaces it.
    """
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    if any(
        key in override and override[key] != base.get(key) for key in ("type", "ref")
    ):
        return override
    merged = dict(base)
    for key, item in override.items():
        merged[key] = _merge(base.get(key), item)
    return merged


def _path(value: Any, where: str) -> Path:
    """Return a path from a file, relative to that file's directory."""
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected a path, got {value!r}")
    if isinstance(value, _SourceStr):
        return value.base / value
    return Path(value)


def _plain(value: Any) -> Any:
    """Return a value with its strings untagged."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return str(value)
    return value


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


def _build(value: Any, hint: Any, where: str, context: Mapping[str, Any]) -> Any:
    """Build a value to fit a type hint.

    Args:
        value (Any): Table, list or value read from TOML.
        hint (Any): Type the result must fit.
        where (str): Key path of the value, for error messages.
        context (Mapping[str, Any]): Objects already built in this table and
            the tables enclosing it, by name.

    Returns:
        Any: The built value.

    Raises:
        ConfigError: If the value does not fit the hint.
    """
    members = _members(hint)
    if isinstance(value, dict):
        return _build_table(value, members, where, context)
    if isinstance(value, list):
        return _build_list(value, members, where, context)
    return _build_value(value, members, where)


def _build_value(value: Any, members: list[Any], where: str) -> Any:
    """Build a string or number as a path, quantity or plain value.

    Raises:
        ConfigError: If a number is given for a quantity.
    """
    quantity = any(_is_quantity(member) for member in members)
    if isinstance(value, str):
        if Path in members:
            return _path(value, where)
        return _quantity(value, where) if quantity else str(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        if quantity and float not in members and int not in members:
            raise ConfigError(f'{where}: {value} needs units, such as "{value} m"')
        if float in members:
            return float(value)
    return value


def _build_table(
    table: dict[str, Any],
    members: list[Any],
    where: str,
    context: Mapping[str, Any],
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
        arguments = {key: item for key, item in table.items() if key != "type"}
        return _call(target, arguments, where, context)
    if any(_is_quantity(member) for member in members) and set(table) <= {
        "value",
        "diag",
        "units",
    }:
        return _array(table, where)
    classes = [member for member in members if _is_buildable(member)]
    if len(classes) == 1:
        return _call(classes[0], table, where, context)
    abstract = [m.__name__ for m in members if inspect.isabstract(m)]
    reason = f"{', '.join(abstract)} is abstract, so " if abstract else ""
    raise ConfigError(f'{where}: {reason}the table needs type = "dotted.path"')


def _build_list(
    items: list[Any], members: list[Any], where: str, context: Mapping[str, Any]
) -> Any:
    """Build a list as an array or as a sequence of built items."""
    if np.ndarray in members:
        return np.asarray(_plain(items), dtype=float)
    for member in members:
        origin = get_origin(member)
        if origin in (list, tuple, Sequence):
            args = get_args(member)
            item_hint = args[0] if args else Any
            built = [
                _build(item, item_hint, f"{where}[{index}]", context)
                for index, item in enumerate(items)
            ]
            return built if origin is list else tuple(built)
    return [
        _build(item, Any, f"{where}[{index}]", context)
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
    components = np.asarray(_plain(table.get("value", table.get("diag"))), float)
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
) -> Any:
    """Call a callable with a table's keys as its keyword arguments.

    Arguments are built in the callable's parameter order, each able to use
    the ones before it. Keys the callable does not name pass through its
    ``**kwargs``, built to fit the fields of the class it is a classmethod of
    or returns.

    Raises:
        ConfigError: If a key matches no parameter, or the call fails.
    """
    parameters = inspect.signature(target).parameters
    hints = _hints(target)
    named = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    extra = [key for key in table if key not in named]
    extra_hints: dict[str, Any] = {}
    if extra:
        if not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            raise ConfigError(
                f"{where}: unknown keys {extra}; "
                f"{getattr(target, '__qualname__', target)} takes {named}"
            )
        owner = getattr(target, "__self__", None)
        if not isinstance(owner, type):
            owner = hints.get("return")
        extra_hints = _hints(owner) if isinstance(owner, type) else {}

    local = dict(context)
    arguments: dict[str, Any] = {}
    for name in (*named, *extra):
        if name in table:
            hint = hints.get(name, Any) if name in named else extra_hints.get(name, Any)
            key_path = f"{where}.{name}" if where else name
            arguments[name] = _build(table[name], hint, key_path, local)
            if not isinstance(arguments[name], _PLAIN_TYPES):
                local[name] = arguments[name]
        elif parameters[name].default is inspect.Parameter.empty and name in context:
            arguments[name] = context[name]
    try:
        return target(**arguments)
    except (TypeError, ValueError, OSError) as error:
        raise ConfigError(f"{where or 'top level'}: {error}") from error
