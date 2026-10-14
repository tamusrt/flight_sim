"""Write tables as TOML files, with comments on chosen keys."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import tomli_w

from flight_sim.config.toml_files import item_path, key_path


def write_toml(
    path: Path, table: Mapping[str, Any], comments: Mapping[str, str] | None = None
) -> None:
    """Write a table as a TOML file.

    Args:
        path (Path): File to write.
        table (Mapping[str, Any]): Table to write. Values TOML cannot hold,
            such as None, are written as commented-out keys.
        comments (Mapping[str, str] | None, optional): Comment to put after
            the key or table header at each key path, such as
            "config.truth.wind" or "events[0].delay". Defaults to none.
    """
    lines = _toml_table(table, (), "", comments or {})
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# Longest inline table written on one line before it becomes a table section
_INLINE_LENGTH = 80


def _toml_table(
    table: Mapping[str, Any],
    header: tuple[str, ...],
    where: str,
    comments: Mapping[str, str],
) -> list[str]:
    """Return the lines of a table's keys, then its sub-tables."""
    lines: list[str] = []
    sections: list[tuple[str, Any, str]] = []
    for key, value in table.items():
        key_where = key_path(where, key)
        if _is_section(value):
            sections.append((key, value, key_where))
            continue
        text = _toml_value(value)
        line = (
            f"{_toml_key(key)} = {text}"
            if text is not None
            else f"# {_toml_key(key)} = {value!r}"
        )
        lines.append(_commented(line, comments.get(key_where)))
    for key, value, key_where in sections:
        keys = (*header, key)
        name = ".".join(_toml_key(part) for part in keys)
        if isinstance(value, Mapping):
            comment = comments.get(key_where)
            # A table holding only tables needs no header of its own
            if comment or not all(_is_section(item) for item in value.values()):
                lines += ["", _commented(f"[{name}]", comment)]
            lines += _toml_table(value, keys, key_where, comments)
            continue
        for index, item in enumerate(value):
            item_where = item_path(key_where, index)
            lines += ["", _commented(f"[[{name}]]", comments.get(item_where))]
            lines += _toml_table(item, keys, item_where, comments)
    return lines


def _is_section(value: Any) -> bool:
    """Return whether a value is written as a table section, not on one line."""
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, Mapping) for item in value)
    if not isinstance(value, Mapping):
        return False
    text = _toml_value(value)
    return (
        text is None
        or len(text) > _INLINE_LENGTH
        or any(isinstance(item, Mapping) for item in value.values())
    )


def _toml_value(value: Any) -> str | None:
    """Return a value on one line of TOML, or None if TOML cannot hold it."""
    if isinstance(value, Mapping):
        items = [(key, _toml_value(item)) for key, item in value.items()]
        if any(text is None for _, text in items):
            return None
        pairs = ", ".join(f"{_toml_key(key)} = {text}" for key, text in items)
        return f"{{ {pairs} }}" if pairs else "{}"
    if isinstance(value, list | tuple):
        texts = [_toml_value(item) for item in value]
        if any(text is None for text in texts):
            return None
        return f"[{', '.join(cast(list[str], texts))}]"
    try:
        return tomli_w.dumps({"x": value}).removeprefix("x = ").rstrip("\n")
    except TypeError:
        return None


def _toml_key(key: str) -> str:
    """Return a key bare if TOML allows it, or quoted."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return tomli_w.dumps({key: 0}).rsplit(" = ", 1)[0]


def _commented(line: str, comment: str | None) -> str:
    """Return a line with a comment after it, if there is one."""
    return f"{line}  # {comment}" if comment else line
