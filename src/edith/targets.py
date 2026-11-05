"""Find the values a campaign's dispersions name in the base sim.

A dispersion's key is a key path in the base sim's resolved input, the table
``sim.source.table`` holds, such as ``properties.dry_mass_properties.mass``,
with ``[i]`` after a list, such as
``properties.dry_mass_properties.cg_location.value[0]``. flight_sim resolves
the sim's files wherever they sit, so the key path is the same however the
files are laid out.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NoReturn

from edith.errors import EdithConfigError, closest
from flight_sim.config import ConfigError, InputSource
from flight_sim.simulation import Simulation
from flight_sim.units import UNREADABLE_UNITS, parse_quantity, parse_units

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


def load_base_sim(path: Path) -> InputSource:
    """Build the base sim from its input file and return its resolved input.

    Args:
        path (Path): The sim input file.

    Returns:
        InputSource: The resolved input, with includes and defaults filled in
            and paths made absolute.

    Raises:
        EdithConfigError: If flight_sim cannot read or build the sim.
    """
    try:
        simulation = Simulation.from_file(path)
    except ConfigError as error:
        raise EdithConfigError(f"base_sim: {error}") from error
    assert simulation.source is not None
    return simulation.source


def find_target(
    source: InputSource, key: str, where: str
) -> tuple[ValueTarget, Nominal]:
    """Return the value a dispersion's key names, and its nominal value.

    Args:
        source (InputSource): The base sim's resolved input.
        key (str): Such as "properties.dry_mass_properties.mass".
        where (str): Key path of the dispersion, for messages.

    Returns:
        tuple[ValueTarget, Nominal]: Where the value is, and the value.

    Raises:
        EdithConfigError: If the key is malformed or names no number of the
            base sim.
    """
    path = _key_parts(key, where)
    node, parents = _walk(source.table, path, where)
    return ValueTarget(path), _leaf(node, _units_owner(path, parents), path, where)


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
        close = closest(str(path[depth]), [str(key) for key in node])
        if close is not None:
            message += f" (did you mean {value_path((*found, close))}?)"
    elif isinstance(node, list):
        message += f"; {value_path(found)} has {len(node)} items"
    elif found:
        message += f"; {value_path(found)} is {node!r}"
    raise EdithConfigError(message)
