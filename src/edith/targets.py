"""Find the values a campaign's dispersions name in the base sim's files.

A dispersion's key names a file in the base sim's folder and a value in it: a
key path in a TOML file, such as ``vehicle.toml:dry_mass_properties.mass``, or
columns of an aero table that one factor scales, such as ``aero.csv:Cy+Cz``.
"""

import csv
import difflib
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NoReturn, TypeAlias

from pint import PintError

from edith.errors import EdithConfigError
from flight_sim.units import parse_units
from flight_sim.utilities.data_loader import AERO_COLUMNS, POWER_ON_COLUMN

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

# Columns of an aero table: its grid, and the columns aero_table_from_csv
# reads, which a factor may scale
_GRID_COLUMNS = ("Mach", "Alpha", "Phi")
_SCALABLE_COLUMNS = (*AERO_COLUMNS, POWER_ON_COLUMN)

# A number and its units, such as "0.25 kg", "-2.08 m" or "1e-5 m/s"
_QUANTITY = re.compile(
    r"(?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(?P<units>(?![eE][+-]?\d)[A-Za-z].*)"
)

# One key of a key path, with any list indices after it, such as "value[0]"
_KEY_PART = re.compile(r"(?P<name>[A-Za-z0-9_-]+)(?P<indices>(?:\[\d+\])*)")


@dataclass(frozen=True)
class TomlTarget:
    """Value at a key path in a TOML file of the base sim's folder."""

    file: str  # Such as "vehicle.toml"

    # Table keys and list indices down to the value, such as
    # ("dry_mass_properties", "cg_location", "value", 0)
    path: tuple[str | int, ...]


@dataclass(frozen=True)
class CsvTarget:
    """Columns of an aero table, which one factor scales."""

    file: str  # Such as "aero.csv"
    columns: tuple[str, ...]  # Such as ("Cy", "Cz")


Target: TypeAlias = TomlTarget | CsvTarget


class Nominal(NamedTuple):
    """Value of a target in the base sim."""

    value: float
    units: str  # As the file writes them, such as "kg"; "" for no units
    text: str  # As messages show it, such as '"17 kg"'


def display_path(path: Path) -> str:
    """Return a path relative to the working directory if it is under it."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


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


class BaseFolder:
    """Files of the base sim's folder, each read once."""

    def __init__(self, folder: Path) -> None:
        """List the files of a folder."""
        self.folder = folder
        self.names = sorted(entry.name for entry in folder.iterdir() if entry.is_file())
        self._tables: dict[str, dict[str, object]] = {}
        self._headers: dict[str, list[str]] = {}

    def target(self, key: str, where: str) -> tuple[Target, Nominal]:
        """Return the value a dispersion's key names, and its nominal value.

        Args:
            key (str): Such as "vehicle.toml:dry_mass_properties.mass".
            where (str): Key path of the dispersion, for messages.

        Returns:
            tuple[Target, Nominal]: Where the value is, and the value.

        Raises:
            EdithConfigError: If the key is malformed or names no number of
                the base sim.
        """
        file, colon, rest = key.partition(":")
        if not colon or not file or not rest:
            raise EdithConfigError(
                f"{where}: name a file of the base sim and a value in it, such "
                'as "vehicle.toml:dry_mass_properties.mass" or "aero.csv:Cx"'
            )
        suffix = Path(file).suffix.lower()
        if "/" in file or "\\" in file or suffix not in (".toml", ".csv"):
            raise EdithConfigError(
                f"{where}: expected a TOML or CSV file directly in the base sim's "
                f"folder, got {file!r}"
            )
        if suffix == ".csv":
            csv_target = CsvTarget(file, _columns(rest, where))
            return csv_target, _csv_nominal(
                self._header(file, where), csv_target, where
            )
        toml_target = TomlTarget(file, _key_parts(rest, where))
        return toml_target, _toml_nominal(self._table(file, where), toml_target, where)

    def _file(self, name: str, where: str) -> Path:
        """Return a file of the folder, named as on disk.

        Raises:
            EdithConfigError: If the folder holds no file of that name.
        """
        if name in self.names:
            return self.folder / name
        same = [other for other in self.names if other.lower() == name.lower()]
        if same:
            raise EdithConfigError(f"{where}: names {name}, but the file is {same[0]}")
        files = [other for other in self.names if other.endswith((".toml", ".csv"))]
        raise EdithConfigError(
            f"{where}: {name} is not in {display_path(self.folder)}, the base "
            f"sim's folder, which holds {', '.join(files)}"
        )

    def _table(self, name: str, where: str) -> dict[str, object]:
        """Return the table of a TOML file of the folder.

        Raises:
            EdithConfigError: If the file is missing or not valid TOML.
        """
        if name not in self._tables:
            path = self._file(name, where)
            try:
                with path.open("rb") as file:
                    self._tables[name] = tomllib.load(file)
            except OSError as error:
                raise EdithConfigError(
                    f"{where}: cannot read {name}: {error.strerror or error}"
                ) from error
            except tomllib.TOMLDecodeError as error:
                raise EdithConfigError(f"{where}: {name}: {error}") from error
        return self._tables[name]

    def _header(self, name: str, where: str) -> list[str]:
        """Return the column names of a CSV file of the folder.

        Raises:
            EdithConfigError: If the file is missing or cannot be read.
        """
        if name not in self._headers:
            path = self._file(name, where)
            try:
                with path.open(newline="", encoding="utf-8-sig") as file:
                    row = next(csv.reader(file), [])
            except (OSError, UnicodeDecodeError, csv.Error) as error:
                raise EdithConfigError(
                    f"{where}: cannot read {name}: {error}"
                ) from error
            self._headers[name] = [column.strip() for column in row]
        return self._headers[name]


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
                "dry_mass_properties.cg_location.value[0]"
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
        if column in _GRID_COLUMNS:
            raise EdithConfigError(
                f"{where}: {column} is a grid column of the aero table; scale {choices}"
            )
        if column not in _SCALABLE_COLUMNS:
            raise EdithConfigError(
                f"{where}: {column!r} is not a column the aero table reads; "
                f"scale {choices}, where Cx holds the drag"
            )
    if len(set(columns)) < len(columns):
        raise EdithConfigError(f"{where}: names a column twice")
    return columns


def _csv_nominal(header: Sequence[str], target: CsvTarget, where: str) -> Nominal:
    """Return the nominal factor on aero columns, checking the table has them.

    Raises:
        EdithConfigError: If the CSV is not an aero table, or lacks a column.
    """
    if not all(column in header for column in _GRID_COLUMNS):
        raise EdithConfigError(
            f"{where}: {target.file} is not an aero table, which has Mach, "
            "Alpha and Phi columns"
        )
    missing = [column for column in target.columns if column not in header]
    if missing:
        raise EdithConfigError(
            f"{where}: {target.file} has no {' or '.join(missing)} column; its "
            f"columns are {', '.join(header)}"
        )
    columns = " and ".join(target.columns)
    plural = "s" if len(target.columns) > 1 else ""
    return Nominal(1.0, "", f"1, a factor on the {columns} column{plural}")


def _toml_nominal(
    table: Mapping[str, object], target: TomlTarget, where: str
) -> Nominal:
    """Return the nominal value a key path names in a TOML file's table.

    Raises:
        EdithConfigError: If the path leads nowhere, or to no single number.
    """
    node: object = table
    parents: list[object] = []
    for depth, part in enumerate(target.path):
        parents.append(node)
        child = _child(node, part)
        if child is None:
            _no_value(table, node, target, depth, where)
        node = child
    return _leaf(node, _units_owner(target.path, parents), target, where)


def _child(node: object, part: str | int) -> object:
    """Return the item a key names in a table, or an index in a list, or None."""
    if isinstance(part, int):
        return node[part] if isinstance(node, list) and part < len(node) else None
    return node.get(part) if isinstance(node, dict) else None


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


def _leaf(node: object, owner: object, target: TomlTarget, where: str) -> Nominal:
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
            f"{value_path(target.path)}.{part}[0]"
        )
    if isinstance(node, dict):
        raise EdithConfigError(
            f"{where}: the value is a table; disperse one of its values, "
            f"{', '.join(map(str, node))}"
        )
    if isinstance(node, list):
        raise EdithConfigError(
            f"{where}: the value is a list; disperse one number of it, such as "
            f"{value_path(target.path)}[0]"
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
    table: Mapping[str, object],
    node: object,
    target: TomlTarget,
    depth: int,
    where: str,
) -> NoReturn:
    """Raise the error for a key path that stops at ``node``, at ``depth``.

    Raises:
        EdithConfigError: Naming the missing part, with the file the value
            may be in, or the key closest to it.
    """
    found = target.path[:depth]
    message = f"{where}: {target.file} has no {value_path(target.path[: depth + 1])}"
    if isinstance(node, dict):
        include = node.get("include")
        if isinstance(include, str):
            place = value_path(found) or "its top level"
            rest = value_path(target.path[depth:])
            raise EdithConfigError(
                f'{message}; {place} includes {include}, so disperse "{include}:{rest}"'
            )
        keys = [str(key) for key in node]
        close = difflib.get_close_matches(str(target.path[depth]), keys, n=1)
        if close:
            message += f" (did you mean {value_path((*found, close[0]))}?)"
    elif isinstance(node, list):
        message += f"; {value_path(found)} has {len(node)} items"
    elif found:
        message += f"; {value_path(found)} is {node!r}"
    extends = table.get("extends")
    if isinstance(extends, str):
        message += f"; {target.file} extends {extends}, which may set it"
    raise EdithConfigError(message)
