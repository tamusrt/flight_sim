"""Load objects from TOML input files."""

import time as clock
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from flight_sim.config.building import BuildRecord, build
from flight_sim.config.errors import ConfigError
from flight_sim.config.toml_files import item_path, key_path, load_toml, untagged

T = TypeVar("T")


@dataclass(frozen=True)
class InputSource:
    """Input file an object was built from, and the table that rebuilds it."""

    path: Path  # Absolute

    # The file's table with its includes and base file merged in, every path
    # made absolute and every argument left to its default filled in. Defaults
    # that have no input form, such as None, are kept as they are.
    table: dict[str, Any]

    # Key paths in ``table`` of the arguments left to their defaults, such as
    # "config.truth.wind"
    defaults: frozenset[str]

    # Wall time reading and building took, in seconds
    load_time_s: float


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
    return load_with_source(path, cls)[0]


def load_with_source(path: str | Path, cls: type[T]) -> tuple[T, InputSource]:
    """Build an object from a TOML file, and describe the input it came from.

    Args:
        path (str | Path): TOML file whose top-level table builds the object.
        cls (type[T]): Class of the object, used when the file names no
            ``type``.

    Returns:
        tuple[T, InputSource]: The built object, and the file and resolved
            table it was built from.

    Raises:
        ConfigError: If the file cannot be read or does not build a ``cls``.
    """
    start = clock.perf_counter()
    table = load_toml(path)
    record = BuildRecord()
    result = build(table, cls, record=record)
    if not isinstance(result, cls):
        raise ConfigError(
            f"{path} builds a {type(result).__name__}, not a {cls.__name__}"
        )
    defaults_by_table: dict[str, dict[str, Any]] = {}
    for default_path, value in record.defaults.items():
        parent, _, name = default_path.rpartition(".")
        defaults_by_table.setdefault(parent, {})[name] = value
    source = InputSource(
        path=Path(path).resolve(),
        table=_resolved(table, "", record.paths, defaults_by_table),
        defaults=frozenset(record.defaults),
        load_time_s=clock.perf_counter() - start,
    )
    return result, source


def _resolved(
    node: Any,
    where: str,
    paths: Mapping[str, Path],
    defaults_by_table: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Return a table with its built paths made absolute and its defaults added."""
    if where in paths:
        return paths[where].as_posix()
    if isinstance(node, list):
        return [
            _resolved(item, item_path(where, index), paths, defaults_by_table)
            for index, item in enumerate(node)
        ]
    if not isinstance(node, dict):
        return untagged(node)
    table = {
        key: _resolved(item, key_path(where, key), paths, defaults_by_table)
        for key, item in node.items()
    }
    for name, value in defaults_by_table.get(where, {}).items():
        table.setdefault(name, value)
    return table
