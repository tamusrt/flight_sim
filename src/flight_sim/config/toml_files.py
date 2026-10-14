"""Read TOML input files, inserting includes and merging base files."""

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flight_sim.config.errors import ConfigError


class SourceStr(str):
    """String read from a TOML file, which remembers the file's directory."""

    base: Path

    def __new__(cls, text: str, base: Path) -> "SourceStr":
        """Tag a string with the directory of the file it came from."""
        result = super().__new__(cls, text)
        result.base = base  # pylint: disable=attribute-defined-outside-init
        return result


class SourceTable(dict[str, Any]):
    """Table read from TOML, which remembers the files its keys came from."""

    files: tuple[Path, ...] = ()


def source_table(items: dict[str, Any], files: Sequence[Path]) -> SourceTable:
    """Return a table of items that came from these files."""
    table = SourceTable(items)
    table.files = tuple(dict.fromkeys(files))
    return table


def origin(where: str, table: Any) -> str:
    """Return a key path with the files its table came from, for error messages."""
    files = getattr(table, "files", ())
    if not files:
        return where or "top level"
    names = []
    for file in files:
        try:
            names.append(file.relative_to(Path.cwd()).as_posix())
        except ValueError:
            names.append(file.as_posix())
    return f"{where or 'top level'} in {', '.join(names)}"


def absolute(node: Any) -> Any:
    """Return a table with its strings untagged, those naming files made absolute."""
    if isinstance(node, dict):
        return {key: absolute(item) for key, item in node.items()}
    if isinstance(node, list):
        return [absolute(item) for item in node]
    if isinstance(node, SourceStr) and node and (node.base / node).is_file():
        return (node.base / node).resolve().as_posix()
    if isinstance(node, str):
        return str(node)
    return node


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
    table: dict[str, Any] = _expand(_located(raw, path), chain)
    extends = table.pop("extends", None)
    if extends is not None:
        table = _merge(load_toml(source_path(extends, "extends"), chain), table)
    return table


def _located(node: Any, file: Path) -> Any:
    """Tag every table and string in a TOML table with the file it came from."""
    if isinstance(node, dict):
        return source_table(
            {key: _located(item, file) for key, item in node.items()}, (file,)
        )
    if isinstance(node, list):
        return [_located(item, file) for item in node]
    if isinstance(node, str):
        return SourceStr(node, file.parent)
    return node


def _expand(node: Any, chain: Sequence[Path]) -> Any:
    """Insert the file each ``include`` names under the table holding it."""
    if isinstance(node, list):
        return [_expand(item, chain) for item in node]
    if not isinstance(node, dict):
        return node
    table = source_table(
        {key: _expand(item, chain) for key, item in node.items()},
        getattr(node, "files", ()),
    )
    include = table.pop("include", None)
    if include is None:
        return table
    return _merge(load_toml(source_path(include, "include"), chain), table)


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
    merged = source_table(
        dict(base), (*getattr(base, "files", ()), *getattr(override, "files", ()))
    )
    for key, item in override.items():
        merged[key] = _merge(base.get(key), item)
    return merged


def source_path(value: Any, where: str) -> Path:
    """Return a path from a file, relative to that file's directory."""
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected a path, got {value!r}")
    if isinstance(value, SourceStr):
        return value.base / value
    return Path(value)


def plain(value: Any) -> Any:
    """Return a value with its strings untagged."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, str):
        return str(value)
    return value
