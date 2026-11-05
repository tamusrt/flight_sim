"""Read and check EDITH campaign files, such as edith_config.toml.

A campaign file names the base sim, the number of trials, the master seed and
the values to disperse, each by its key path in the base sim's resolved input,
as ``edith.targets`` describes. Each dispersion's ``dispersion`` table names
the class that draws its values; for now that is always StandardDispersion,
which draws a number from a normal or uniform distribution and is the default
for a dispersion that leaves the table out. A dispersion's
numbers are converted to the units its nominal value is written in.
"""

import contextlib
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Self, TypeAlias, cast, get_args

from pint import DimensionalityError, OffsetUnitCalculusError

from edith.errors import EdithConfigError, closest
from edith.targets import (
    Nominal,
    ValueTarget,
    find_target,
    load_base_sim,
    value_path,
)
from flight_sim.config import InputSource
from flight_sim.config.toml_files import display_path, key_path
from flight_sim.config.writing import toml_key
from flight_sim.units import UNREADABLE_UNITS, convert, parse_quantity

__all__ = [
    "STANDARD_DISPERSION",
    "Dispersion",
    "DispersionClass",
    "Distribution",
    "EdithConfig",
    "Normal",
    "Sampling",
    "Uniform",
]

Sampling: TypeAlias = Literal["lhs", "random"]


def _listed(keys: Sequence[str]) -> str:
    """Return keys as a list in prose, such as "a, b and c"."""
    return " and ".join((", ".join(keys[:-1]), keys[-1])) if len(keys) > 1 else keys[0]


# Keys of a campaign file
_REQUIRED_KEYS = ("N", "master_seed", "base_sim", "output_dir", "dispersions")
_OPTIONAL_KEYS = ("sampling", "workers", "batch_size", "trial_timeout")
_CAMPAIGN_USAGE = (
    f"a campaign file needs {_listed(_REQUIRED_KEYS)}, "
    f"and takes {_listed(_OPTIONAL_KEYS)}"
)


class _Keys(NamedTuple):
    """Keys one dist_type needs and takes besides its own, and how errors say so."""

    required: tuple[str, ...]
    optional: tuple[str, ...]
    usage: str


_DISTRIBUTION_KEYS = {
    "normal": _Keys(
        ("sigma",),
        ("low", "high"),
        "a normal dispersion takes sigma, and optionally low and high",
    ),
    "uniform": _Keys(
        (),
        ("half_width", "low", "high"),
        "a uniform dispersion takes half_width, or low and high",
    ),
}

# Keys of a dispersion that hold its numbers
_SPREAD_KEYS = ("sigma", "half_width", "low", "high")

_DISPERSION_USAGE = (
    "for now every dispersion leaves dispersion out, or names its class as "
    'dispersion = { module = "StandardDispersion", class = "StandardDispersion" }'
)


@dataclass(frozen=True)
class DispersionClass:
    """Class that draws a dispersion's values, as its ``dispersion`` table names it."""

    module: str  # Such as "StandardDispersion"
    class_name: str  # The table's class, such as "StandardDispersion"


# Draws a number of the base sim from a normal or uniform distribution
STANDARD_DISPERSION = DispersionClass("StandardDispersion", "StandardDispersion")

# The classes a dispersion may name; for the CDR, only the standard one
_DISPERSION_CLASSES = (STANDARD_DISPERSION,)


@dataclass(frozen=True)
class Normal:
    """Normal distribution about the dispersion's nominal value.

    ``low`` and ``high``, where given, truncate it on either side of the
    nominal value.
    """

    sigma: float
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution from ``low`` to ``high``."""

    low: float
    high: float


Distribution: TypeAlias = Normal | Uniform


@dataclass(frozen=True)
class Dispersion:
    """One value of the base sim and the distribution its trials draw it from.

    The nominal value and the distribution's numbers are all in ``units``.
    """

    # Key of its table in the campaign file, such as
    # "properties.dry_mass_properties.mass"
    key: str
    target: ValueTarget
    nominal: float  # Value in the base sim
    distribution: Distribution

    # Units of the nominal value as the input writes them, such as "kg"; ""
    # for a plain number
    units: str

    # Class that draws its values, as its dispersion table names it
    dispersion_class: DispersionClass = STANDARD_DISPERSION


@dataclass(frozen=True)
class EdithConfig:
    """Monte Carlo campaign of dispersed flights of one base sim.

    ``from_file`` reads one from a campaign file. Its paths are absolute, and
    the optional settings the file leaves out are None. ``source`` is the base
    sim's resolved input, which the dispersions' keys index and trials are
    written from.
    """

    path: Path  # The campaign file
    trials: int  # N in the campaign file
    master_seed: int
    base_sim: Path
    output_dir: Path
    dispersions: tuple[Dispersion, ...]  # In file order
    source: InputSource = field(repr=False)
    sampling: Sampling = "lhs"
    workers: int | None = None
    batch_size: int | None = None  # Trials per batch
    trial_timeout_s: float | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read and check a campaign file.

        Args:
            path (str | Path): Campaign file, such as edith_config.toml.

        Returns:
            EdithConfig: The campaign the file describes.

        Raises:
            EdithConfigError: If the file cannot be read, is not TOML or does
                not describe a valid campaign. The message names the file and
                the key at fault.
        """
        path = Path(path).resolve()
        table = _read_table(path)
        try:
            return cls.from_table(table, path)
        except EdithConfigError as error:
            raise EdithConfigError(f"{display_path(path)}: {error}") from error

    @classmethod
    def from_table(cls, table: Mapping[str, object], path: str | Path) -> Self:
        """Check a campaign file's table and build the campaign it describes.

        Args:
            table (Mapping[str, object]): The file's table, as ``tomllib``
                reads it.
            path (str | Path): The campaign file, which relative paths in the
                table are taken from.

        Returns:
            EdithConfig: The campaign.

        Raises:
            EdithConfigError: If a key is missing, unknown or invalid, the
                base sim does not build, or a dispersion names no value of it.
                The message names the key at fault.
        """
        path = Path(path).resolve()
        _check_keys(table, "", _REQUIRED_KEYS, _OPTIONAL_KEYS, _CAMPAIGN_USAGE)
        trials = _whole_number(table["N"], "N", 1)
        master_seed = _whole_number(table["master_seed"], "master_seed", 0)
        sampling = _sampling(table.get("sampling", "lhs"))
        workers = _optional_whole_number(table, "workers")
        batch_size = _optional_whole_number(table, "batch_size")
        trial_timeout_s = _optional_timeout_s(table)
        base_sim = _base_sim(table["base_sim"], path.parent)
        output_dir = _output_dir(table["output_dir"], path.parent)
        source = load_base_sim(base_sim)
        return cls(
            path=path,
            trials=trials,
            master_seed=master_seed,
            base_sim=base_sim,
            output_dir=output_dir,
            dispersions=_dispersions(table["dispersions"], source),
            source=source,
            sampling=sampling,
            workers=workers,
            batch_size=batch_size,
            trial_timeout_s=trial_timeout_s,
        )


def _read_table(path: Path) -> dict[str, object]:
    """Read a campaign file's table without checking it.

    Raises:
        EdithConfigError: If the file cannot be read or is not valid TOML.
    """
    # Not load_toml: a campaign file takes no include or extends
    try:
        # utf-8-sig also reads files that editors saved with a byte-order mark
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise EdithConfigError(
            f"cannot read {display_path(path)}: {error.strerror or error}"
        ) from error
    except UnicodeDecodeError as error:
        raise EdithConfigError(f"{display_path(path)} is not UTF-8 text") from error
    try:
        table: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise EdithConfigError(f"{display_path(path)}: {error}") from error
    return table


def _key_path(where: str, key: str) -> str:
    """Return the key path of a key in the table at ``where``, quoted if need be."""
    return key_path(where, toml_key(key))


def _check_keys(
    table: Mapping[str, object],
    where: str,
    required: Sequence[str],
    optional: Sequence[str],
    usage: str,
) -> None:
    """Check a table holds every required key and no key it does not take.

    Raises:
        EdithConfigError: Naming the first unknown key, or the missing keys.
    """
    known = (*required, *optional)
    for key in table:
        if key not in known:
            close = closest(key, known)
            hint = f" (did you mean {close}?)" if close is not None else ""
            raise EdithConfigError(
                f"{_key_path(where, key)}: unknown key{hint}; {usage}"
            )
    missing = [key for key in required if key not in table]
    if missing:
        place = f"{where}: " if where else ""
        raise EdithConfigError(f"{place}missing {' and '.join(missing)}; {usage}")


def _whole_number(value: object, where: str, minimum: int) -> int:
    """Return an integer of at least ``minimum``.

    Raises:
        EdithConfigError: If the value is not an integer, or is too small.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EdithConfigError(
            f"{where}: expected a whole number of at least {minimum}, got {value!r}"
        )
    return value


def _optional_whole_number(table: Mapping[str, object], key: str) -> int | None:
    """Return a positive integer setting, or None if the table leaves it out."""
    return _whole_number(table[key], key, 1) if key in table else None


def _sampling(value: object) -> Sampling:
    """Return the sampling method.

    Raises:
        EdithConfigError: If it is not "lhs" or "random".
    """
    if isinstance(value, str) and value in get_args(Sampling):
        return cast(Sampling, value)
    raise EdithConfigError(f'sampling: expected "lhs" or "random", got {value!r}')


def _optional_timeout_s(table: Mapping[str, object]) -> float | None:
    """Return the trial timeout in seconds, or None if the table leaves it out.

    Raises:
        EdithConfigError: If it is not a time above zero.
    """
    if "trial_timeout" not in table:
        return None
    value = table["trial_timeout"]
    quantity = parse_quantity(value) if isinstance(value, str) else None
    seconds = math.nan
    if quantity is not None:
        with contextlib.suppress(*UNREADABLE_UNITS):
            seconds = convert(quantity[0], quantity[1], "s")
    if not 0.0 < seconds < math.inf:
        raise EdithConfigError(
            f'trial_timeout: expected a time above 0, such as "10 min", got {value!r}'
        )
    return seconds


def _path(value: object, where: str, folder: Path) -> Path:
    """Return a path given relative to ``folder``, made absolute.

    Raises:
        EdithConfigError: If the value is not a path.
    """
    if not isinstance(value, str) or not value.strip():
        raise EdithConfigError(f"{where}: expected a path, got {value!r}")
    return (folder / value).resolve()


def _base_sim(value: object, folder: Path) -> Path:
    """Return the base sim's input file.

    Raises:
        EdithConfigError: If it is not a TOML file that exists.
    """
    path = _path(value, "base_sim", folder)
    if path.suffix.lower() != ".toml":
        raise EdithConfigError(
            f"base_sim: expected a sim input file ending in .toml, got {value!r}"
        )
    if not path.is_file():
        raise EdithConfigError(f"base_sim: {display_path(path)} does not exist")
    return path


def _output_dir(value: object, folder: Path) -> Path:
    """Return the folder the campaign writes to, which need not exist yet.

    Raises:
        EdithConfigError: If the path is a file.
    """
    path = _path(value, "output_dir", folder)
    if path.exists() and not path.is_dir():
        raise EdithConfigError(f"output_dir: {display_path(path)} is a file")
    return path


class _Number(NamedTuple):
    """One of a dispersion's numbers, in its nominal value's units."""

    value: float
    text: str  # As the campaign file writes it, such as "2 cm"


def _dispersions(value: object, source: InputSource) -> tuple[Dispersion, ...]:
    """Read every dispersion table, checking no two change the same value.

    Raises:
        EdithConfigError: If there are none, one is invalid or two change the
            same value.
    """
    if not isinstance(value, dict) or not value:
        raise EdithConfigError(
            "dispersions: expected one or more tables, such as "
            '[dispersions."properties.dry_mass_properties.mass"]'
        )
    dispersions = tuple(_dispersion(key, item, source) for key, item in value.items())
    first_of: dict[ValueTarget, Dispersion] = {}
    for dispersion in dispersions:
        first = first_of.setdefault(dispersion.target, dispersion)
        if first is not dispersion:
            raise EdithConfigError(
                f"{_key_path('dispersions', dispersion.key)}: changes "
                f"{value_path(dispersion.target.path)}, which "
                f"{_key_path('dispersions', first.key)} changes too"
            )
    return dispersions


def _dispersion(key: str, value: object, source: InputSource) -> Dispersion:
    """Read one dispersion table, whose key names the value it disperses.

    Raises:
        EdithConfigError: If the table names no dispersion class EDITH has,
            the key names no value of the base sim, or the table does not
            describe a distribution of it.
    """
    where = _key_path("dispersions", key)
    if not isinstance(value, dict):
        raise EdithConfigError(
            f"{where}: expected a table of dispersion, dist_type and its spread, "
            f"got {value!r}"
        )
    dispersion_class = _dispersion_class(value, where)
    target, nominal = find_target(source, key, where)
    dist_type = _dist_type(value, where)
    keys = _DISTRIBUTION_KEYS[dist_type]
    _check_keys(
        value,
        where,
        ("dist_type", *keys.required),
        ("dispersion", *keys.optional),
        keys.usage,
    )
    numbers = {
        name: _number(value[name], _key_path(where, name), nominal)
        for name in _SPREAD_KEYS
        if name in value
    }
    distribution: Distribution = (
        _normal(numbers, nominal, where)
        if dist_type == "normal"
        else _uniform(numbers, nominal, where, keys.usage)
    )
    return Dispersion(
        key, target, nominal.value, distribution, nominal.units, dispersion_class
    )


def _dispersion_class(table: Mapping[str, object], where: str) -> DispersionClass:
    """Return the class a ``dispersion`` table names, or the standard one.

    Raises:
        EdithConfigError: If the table is malformed, or names a class EDITH
            does not have.
    """
    if "dispersion" not in table:
        return STANDARD_DISPERSION
    block = table["dispersion"]
    block_where = _key_path(where, "dispersion")
    if not isinstance(block, dict):
        raise EdithConfigError(
            f"{block_where}: expected a table of module and class, got {block!r}; "
            f"{_DISPERSION_USAGE}"
        )
    _check_keys(block, block_where, ("module", "class"), (), _DISPERSION_USAGE)
    module_where = _key_path(block_where, "module")
    class_where = _key_path(block_where, "class")
    chosen = DispersionClass(
        _class_name(block["module"], module_where),
        _class_name(block["class"], class_where),
    )
    if chosen.module not in {option.module for option in _DISPERSION_CLASSES}:
        raise EdithConfigError(
            f"{module_where}: EDITH has no dispersion module {chosen.module!r}; "
            f"{_DISPERSION_USAGE}"
        )
    if chosen not in _DISPERSION_CLASSES:
        raise EdithConfigError(
            f"{class_where}: {chosen.module} has no dispersion class "
            f"{chosen.class_name!r}; {_DISPERSION_USAGE}"
        )
    return chosen


def _class_name(value: object, where: str) -> str:
    """Return the name of a dispersion's module or class.

    Raises:
        EdithConfigError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise EdithConfigError(
            f'{where}: expected a name such as "StandardDispersion", got {value!r}'
        )
    return value


def _dist_type(table: Mapping[str, object], where: str) -> str:
    """Return a dispersion's distribution type.

    Raises:
        EdithConfigError: If it is missing, or not "normal" or "uniform".
    """
    if "dist_type" not in table:
        raise EdithConfigError(f'{where}: missing dist_type, "normal" or "uniform"')
    dist_type = table["dist_type"]
    if not isinstance(dist_type, str) or dist_type not in _DISTRIBUTION_KEYS:
        raise EdithConfigError(
            f'{_key_path(where, "dist_type")}: expected "normal" or "uniform", '
            f"got {dist_type!r}"
        )
    return dist_type


def _number(value: object, where: str, nominal: Nominal) -> _Number:
    """Return one of a dispersion's numbers, converted to its nominal value's units.

    A nominal value with units, such as "17 kg", takes a number with units of
    the same kind; a plain number takes a plain number. A nominal value in
    units with an offset, such as degC, takes numbers in those units only.

    Raises:
        EdithConfigError: If the value is not a finite number in that form, or
            its units do not convert to the nominal value's.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = _finite(float(value), where, value)
        if nominal.units:
            raise EdithConfigError(
                f"{where}: needs units, since the nominal value is "
                f'{nominal.text}; write "{value:g} {nominal.units}"'
            )
        return _Number(number, f"{value:g}")
    text = value.strip() if isinstance(value, str) else ""
    quantity = parse_quantity(text)
    if quantity is None:
        example = f'"1 {nominal.units}"' if nominal.units else "0.05"
        raise EdithConfigError(
            f"{where}: expected a number such as {example}, got {value!r}"
        )
    number = _finite(quantity[0], where, value)
    if not nominal.units:
        # No number is suggested: "20 percent" could mean 0.2, or 20 % of
        # the nominal value
        raise EdithConfigError(
            f"{where}: takes a plain number, since the nominal value is "
            f"{nominal.text}; got {value!r}"
        )
    try:
        converted = convert(number, quantity[1], nominal.units)
    except DimensionalityError as error:
        raise EdithConfigError(
            f'{where}: "{text}" does not convert to {nominal.units}, the units '
            f"of the nominal value, {nominal.text}"
        ) from error
    except OffsetUnitCalculusError as error:
        raise EdithConfigError(
            f"{where}: write {text!r} in {nominal.units}, the units of the nominal "
            f"value, {nominal.text}, since a spread and a value convert differently"
        ) from error
    except UNREADABLE_UNITS as error:
        raise EdithConfigError(f"{where}: cannot read the units of {text!r}") from error
    return _Number(_finite(converted, where, value), text)


def _finite(number: float, where: str, value: object) -> float:
    """Return a number, checking it is finite.

    Raises:
        EdithConfigError: If it is infinite or NaN.
    """
    if not math.isfinite(number):
        raise EdithConfigError(f"{where}: expected a finite number, got {value!r}")
    return number


def _above(first: float, second: float) -> bool:
    """Return whether one number is above another by more than rounding."""
    return first > second and not math.isclose(first, second, abs_tol=1e-12)


def _normal(numbers: Mapping[str, _Number], nominal: Nominal, where: str) -> Normal:
    """Return a normal distribution, checking sigma and its bounds.

    Raises:
        EdithConfigError: If sigma is negative, or low and high do not lie on
            either side of the nominal value.
    """
    sigma, low, high = numbers["sigma"], numbers.get("low"), numbers.get("high")
    if sigma.value < 0.0:
        raise EdithConfigError(
            f"{_key_path(where, 'sigma')}: expected 0 or more, got {sigma.text}"
        )
    side = "low and high truncate a normal distribution on either side of it"
    if low is not None and _above(low.value, nominal.value):
        raise EdithConfigError(
            f"{_key_path(where, 'low')}: {low.text} is above the nominal value, "
            f"{nominal.text}; {side}"
        )
    if high is not None and _above(nominal.value, high.value):
        raise EdithConfigError(
            f"{_key_path(where, 'high')}: {high.text} is below the nominal "
            f"value, {nominal.text}; {side}"
        )
    _check_order(low, high, where)
    return Normal(
        sigma.value,
        None if low is None else low.value,
        None if high is None else high.value,
    )


def _uniform(
    numbers: Mapping[str, _Number], nominal: Nominal, where: str, usage: str
) -> Uniform:
    """Return a uniform distribution of half_width about the nominal, or low to high.

    Raises:
        EdithConfigError: If it has both forms or neither, half_width is
            negative, or low is not below high.
    """
    half_width = numbers.get("half_width")
    low, high = numbers.get("low"), numbers.get("high")
    if half_width is not None:
        if low is not None or high is not None:
            raise EdithConfigError(
                f"{where}: give half_width, or low and high, not both; {usage}"
            )
        if half_width.value < 0.0:
            raise EdithConfigError(
                f"{_key_path(where, 'half_width')}: expected 0 or more, got "
                f"{half_width.text}"
            )
        return Uniform(
            nominal.value - half_width.value, nominal.value + half_width.value
        )
    if low is None and high is None:
        raise EdithConfigError(f"{where}: missing half_width, or low and high; {usage}")
    if low is None or high is None:
        raise EdithConfigError(
            f"{where}: missing {'low' if low is None else 'high'}; {usage}"
        )
    _check_order(low, high, where)
    return Uniform(low.value, high.value)


def _check_order(low: _Number | None, high: _Number | None, where: str) -> None:
    """Check low is below high where both are given.

    Raises:
        EdithConfigError: If it is not.
    """
    if low is not None and high is not None and not low.value < high.value:
        raise EdithConfigError(
            f"{where}: low ({low.text}) must be below high ({high.text})"
        )
