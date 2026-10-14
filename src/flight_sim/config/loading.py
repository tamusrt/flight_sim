"""Load objects from TOML input files."""

from pathlib import Path
from typing import Any, TypeVar

from flight_sim.config.building import build
from flight_sim.config.defaults import with_defaults
from flight_sim.config.errors import ConfigError
from flight_sim.config.toml_files import absolute, load_toml

T = TypeVar("T")


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
    return load_with_input(path, cls)[0]


def load_with_input(
    path: str | Path, cls: type[T]
) -> tuple[T, dict[str, Any], frozenset[str]]:
    """Build an object from a TOML file, and return the input it was built from.

    Args:
        path (str | Path): TOML file whose top-level table builds the object.
        cls (type[T]): Class of the object, used when the file names no
            ``type``.

    Returns:
        tuple[T, dict[str, Any], frozenset[str]]: The built object; the
            file's table with its includes and base file merged in and every
            argument left to its default filled in; and the key paths of
            those defaults, such as "config.truth.wind". Strings naming
            existing files are made absolute, so the table rebuilds the object
            from anywhere. Defaults that have no input form, such as None,
            are left as they are.

    Raises:
        ConfigError: If the file cannot be read or does not build a ``cls``.
    """
    table = load_toml(path)
    defaults: dict[str, Any] = {}
    result = build(table, cls, defaults=defaults)
    if not isinstance(result, cls):
        raise ConfigError(f"{path} builds a {type(result).__name__}, not a {cls}")
    by_table: dict[str, dict[str, Any]] = {}
    for key_path, value in defaults.items():
        parent, _, name = key_path.rpartition(".")
        by_table.setdefault(parent, {})[name] = value
    return result, with_defaults(absolute(table), "", by_table), frozenset(defaults)
