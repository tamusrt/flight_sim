"""Find the values a campaign's dispersions name in the base sim.

A dispersion's key is a key path in the base sim's resolved input, such as
``properties.dry_mass_properties.mass``, with ``[i]`` after a list. An aero
table's key path, a colon and columns of the table name a factor that scales
those columns, such as ``properties.aero_table:Cy+Cz``.
"""

import csv
import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NoReturn, TypeAlias, TypeGuard

from pint import PintError

from edith.errors import EdithConfigError
from flight_sim.config import ConfigError, InputSource
from flight_sim.simulation import Simulation
from flight_sim.units import parse_units
from flight_sim.utilities.data_loader import (
    AERO_COLUMNS,
    POWER_ON_COLUMN,
    aero_table_from_csv,
)

# What Pint raises for units it cannot read, AssertionError among them for
# some malformed ones, such as "m/"
UNREADABLE_UNITS = (
    PintError,
    AssertionError,
    AttributeError,
    SyntaxError,
    TypeError,
    ValueError,
)

# Columns of an aero table that a factor may scale
_SCALABLE_COLUMNS = (*AERO_COLUMNS, POWER_ON_COLUMN)

# The type an aero table read from a CSV names in the input
_AERO_CSV_TYPE = f"{aero_table_from_csv.__module__}.{aero_table_from_csv.__name__}"

# A number and its units, such as "0.25 kg", "-2.08 m" or "1e-5 m/s"
_QUANTITY = re.compile(
    r"(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(?P<units>(?![eE][+-]?\d)[A-Za-z].*)"
)

# One key of a key path, with any list indices after it, such as "value[0]"
_KEY_PART = re.compile(r"(?P<name>[A-Za-z0-9_-]+)(?P<indices>(?:\[\d+\])*)")

# What a key path finds when it leads nowhere, since a value may be None
_MISSING = object()


@dataclass(frozen=True)
class ValueTarget:
    """Number at a key path in the base sim's resolved input."""

    # Table keys and list indices down to the number, such as
    # ("properties", "dry_mass_properties", "cg_location", "value", 0)
    path: tuple[str | int, ...]


@dataclass(frozen=True)
class AeroTarget:
    """Columns of an aero table, which one factor scales."""

    path: tuple[str | int, ...]  # Of the table, such as ("properties", "aero_table")
    file: Path  # Its CSV, absolute
    columns: tuple[str, ...]  # Such as ("Cy", "Cz")


Target: TypeAlias = ValueTarget | AeroTarget


class Nominal(NamedTuple):
    """Value of a target in the base sim."""

    value: float
    units: str  # As the input writes them, such as "kg"; "" for no units
    text: str  # As messages show it, such as '"17 kg"'


def value_path(path: Sequence[str | int]) -> str:
    """Return a key path such as "dry_mass_properties.cg_location.value[0]"."""
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


def parse_quantity(text: str) -> tuple[float, str] | None:
    """Return the number and units of a quantity such as "0.25 kg", or None."""
    match = _QUANTITY.fullmatch(text.strip())
    if match is None:
        return None
    return float(match["number"]), match["units"]


class BaseSim:
    """Resolved input of the base sim, which its dispersions' keys index."""

    def __init__(self, source: InputSource) -> None:
        """Index the resolved input a sim was built from."""
        self.source = source
        self._headers: dict[Path, list[str]] = {}

    @classmethod
    def load(cls, path: Path) -> "BaseSim":
        """Build the base sim from its input file and keep the resolved input.

        Args:
            path (Path): The sim input file.

        Returns:
            BaseSim: The resolved input, with includes and defaults filled in
                and paths made absolute.

        Raises:
            EdithConfigError: If flight_sim cannot read or build the sim.
        """
        try:
            simulation = Simulation.from_file(path)
        except ConfigError as error:
            raise EdithConfigError(f"base_sim: {error}") from error
        assert simulation.source is not None
        return cls(simulation.source)

    def target(self, key: str, where: str) -> tuple[Target, Nominal]:
        """Return the value a dispersion's key names, and its nominal value.

        Args:
            key (str): Such as "properties.dry_mass_properties.mass" or
                "properties.aero_table:Cx".
            where (str): Key path of the dispersion, for messages.

        Returns:
            tuple[Target, Nominal]: Where the value is, and the value.

        Raises:
            EdithConfigError: If the key is malformed or names no number of
                the base sim.
        """
        text, colon, columns = key.partition(":")
        path = _key_parts(text, where)
        node, parents = _walk(self.source.table, path, where)
        if colon:
            if not _is_aero_csv(node):
                raise EdithConfigError(
                    f"{where}: {text} is not an aero table read from a CSV, so "
                    "it has no columns to scale"
                )
            file = Path(str(node["filepath"]))
            target = AeroTarget(path, file, _columns(columns, where))
            header = self._header(target.file, where)
            return target, _aero_nominal(header, target, where)
        if _is_aero_csv(node):
            raise EdithConfigError(
                f"{where}: {text} is an aero table; scale its columns, such as "
                f'"{text}:Cx"'
            )
        return ValueTarget(path), _leaf(node, _units_owner(path, parents), path, where)

    def _header(self, file: Path, where: str) -> list[str]:
        """Return the column names of an aero table's CSV.

        Raises:
            EdithConfigError: If the file cannot be read.
        """
        if file not in self._headers:
            try:
                with file.open(newline="", encoding="utf-8-sig") as handle:
                    row = next(csv.reader(handle), [])
            except (OSError, UnicodeDecodeError, csv.Error) as error:
                raise EdithConfigError(
                    f"{where}: cannot read {file.name}: {error}"
                ) from error
            self._headers[file] = [column.strip() for column in row]
        return self._headers[file]


def _is_aero_csv(node: object) -> TypeGuard[dict[str, object]]:
    """Return whether a node is the table of an aero table read from a CSV."""
    return isinstance(node, dict) and node.get("type") == _AERO_CSV_TYPE


def _key_parts(text: str, where: str) -> tuple[str | int, ...]:
    """Split a key path such as "cg_location.value[0]" into keys and indices.

    Raises:
        EdithConfigError: If the text is not keys joined by dots, each with
            any list indices after it.
    """
    parts: list[str | int] = []
    for piece in text.split("."):
        match = _KEY_PART.fullmatch(piece)
        if match is None:
            raise EdithConfigError(
                f"{where}: cannot read the key path {text!r}; join keys with "
                "dots and put [i] after a list, such as "
                "properties.dry_mass_properties.cg_location.value[0]"
            )
        parts.append(match["name"])
        parts.extend(int(index) for index in re.findall(r"\d+", match["indices"]))
    return tuple(parts)


def _columns(text: str, where: str) -> tuple[str, ...]:
    """Split an aero column selector such as "Cy+Cz" into its columns.

    Raises:
        EdithConfigError: If a column is not one the aero table reads, or
            appears twice.
    """
    columns = tuple(text.split("+"))
    choices = f"{', '.join(_SCALABLE_COLUMNS[:-1])} or {_SCALABLE_COLUMNS[-1]}"
    for column in columns:
        if column not in _SCALABLE_COLUMNS:
            raise EdithConfigError(
                f"{where}: {column!r} is not a column the aero table reads; "
                f"scale {choices}, where Cx holds the drag"
            )
    if len(set(columns)) < len(columns):
        raise EdithConfigError(f"{where}: names a column twice")
    return columns


def _aero_nominal(header: Sequence[str], target: AeroTarget, where: str) -> Nominal:
    """Return the nominal factor on aero columns, checking the table has them.

    Raises:
        EdithConfigError: If the CSV lacks a column.
    """
    missing = [column for column in target.columns if column not in header]
    if missing:
        raise EdithConfigError(
            f"{where}: {target.file.name} has no {' or '.join(missing)} column; "
            f"its columns are {', '.join(header)}"
        )
    columns = " and ".join(target.columns)
    plural = "s" if len(target.columns) > 1 else ""
    return Nominal(1.0, "", f"1, a factor on the {columns} column{plural}")


def _walk(
    table: Mapping[str, object], path: Sequence[str | int], where: str
) -> tuple[object, list[object]]:
    """Return the node at a key path, and the nodes above it, outermost first.

    Raises:
        EdithConfigError: If the path leads nowhere.
    """
    node: object = table
    parents: list[object] = []
    for depth, part in enumerate(path):
        parents.append(node)
        child = _child(node, part)
        if child is _MISSING:
            _no_value(node, path, depth, where)
        node = child
    return node, parents


def _child(node: object, part: str | int) -> object:
    """Return the item a key names in a table, or an index in a list, or _MISSING."""
    if isinstance(part, int):
        return node[part] if isinstance(node, list) and part < len(node) else _MISSING
    return node.get(part, _MISSING) if isinstance(node, dict) else _MISSING


def _units_owner(path: Sequence[str | int], parents: Sequence[object]) -> object:
    """Return the table whose units a path's number takes, or None.

    A number in the ``value`` or ``diag`` list of an array quantity, such as
    ``cg_location.value[0]``, takes the units beside that list.
    """
    names = [index for index, part in enumerate(path) if isinstance(part, str)]
    last = names[-1] if names else -1
    if last < 0 or last == len(path) - 1 or path[last] not in ("value", "diag"):
        return None
    return parents[last]


def _leaf(
    node: object, owner: object, path: Sequence[str | int], where: str
) -> Nominal:
    """Return a key path's nominal value, checking it is a single number.

    Raises:
        EdithConfigError: If the value is not a number, with or without units
            Pint can read.
    """
    if isinstance(node, int | float) and not isinstance(node, bool):
        units = owner.get("units") if isinstance(owner, dict) else None
        if isinstance(units, str):
            return _readable(Nominal(float(node), units, f"{node!r} {units}"), where)
        return Nominal(float(node), "", f"{node!r}, a plain number")
    if isinstance(node, str):
        quantity = parse_quantity(node)
        if quantity is not None:
            return _readable(Nominal(*quantity, f'"{node}"'), where)
        raise EdithConfigError(f'{where}: the value is "{node}", not a number')
    if isinstance(node, dict) and ("value" in node or "diag" in node):
        part = "value" if "value" in node else "diag"
        raise EdithConfigError(
            f"{where}: the value is an array; disperse one number of it, such as "
            f"{value_path(path)}.{part}[0]"
        )
    if isinstance(node, dict):
        raise EdithConfigError(
            f"{where}: the value is a table; disperse one of its values, "
            f"{', '.join(map(str, node))}"
        )
    if isinstance(node, list):
        raise EdithConfigError(
            f"{where}: the value is a list; disperse one number of it, such as "
            f"{value_path(path)}[0]"
        )
    raise EdithConfigError(f"{where}: the value is {node!r}, not a number")


def _readable(nominal: Nominal, where: str) -> Nominal:
    """Return a nominal value, checking Pint can read its units.

    Raises:
        EdithConfigError: If it cannot.
    """
    try:
        parse_units(nominal.units)
    except UNREADABLE_UNITS as error:
        raise EdithConfigError(
            f"{where}: cannot read the units of the nominal value, {nominal.text}"
        ) from error
    return nominal


def _no_value(
    node: object, path: Sequence[str | int], depth: int, where: str
) -> NoReturn:
    """Raise the error for a key path that stops at ``node``, at ``depth``.

    Raises:
        EdithConfigError: Naming the missing part, with the key closest to it
            or the length of the list it indexes.
    """
    found = path[:depth]
    message = f"{where}: the base sim has no {value_path(path[: depth + 1])}"
    if isinstance(node, dict):
        keys = [str(key) for key in node]
        close = difflib.get_close_matches(str(path[depth]), keys, n=1)
        if close:
            message += f" (did you mean {value_path((*found, close[0]))}?)"
    elif isinstance(node, list):
        message += f"; {value_path(found)} has {len(node)} items"
    elif found:
        message += f"; {value_path(found)} is {node!r}"
    raise EdithConfigError(message)
