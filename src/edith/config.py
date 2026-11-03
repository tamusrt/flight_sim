"""Read and check EDITH campaign files, such as edith_config.toml.

A campaign file names the base sim, the number of trials, the master seed and
the values to disperse, each by the dispersion's key, as ``edith.targets``
describes. A dispersion's numbers are converted to the units its nominal value
is written in.
"""

import contextlib
import difflib
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Self, TypeAlias

from pint import DimensionalityError

from edith.errors import EdithConfigError
from edith.targets import (
    UNREADABLE_UNITS,
    BaseFolder,
    CsvTarget,
    Nominal,
    Target,
    TomlTarget,
    display_path,
    parse_quantity,
    value_path,
)
from flight_sim.units import scalar, ureg

__all__ = [
    "CsvTarget",
    "Dispersion",
    "Distribution",
    "EdithConfig",
    "EdithConfigError",
    "Normal",
    "Sampling",
    "Target",
    "TomlTarget",
    "Uniform",
    "read_edith_config",
]

Sampling: TypeAlias = Literal["lhs", "random"]
_SAMPLINGS: dict[str, Sampling] = {"lhs": "lhs", "random": "random"}

# Keys of a campaign file
_REQUIRED_KEYS = ("N", "master_seed", "base_sim", "output_dir", "dispersions")
_OPTIONAL_KEYS = ("sampling", "workers", "batch_size", "trial_timeout")
_CAMPAIGN_USAGE = (
    "a campaign file needs N, master_seed, base_sim, output_dir and "
    "dispersions, and takes sampling, workers, batch_size and trial_timeout"
)


class _Keys(NamedTuple):
    """Keys a dispersion of one type needs and takes, and how errors say so."""

    required: tuple[str, ...]
    optional: tuple[str, ...]
    usage: str


_DISTRIBUTION_KEYS = {
    "normal": _Keys(
        ("dist_type", "sigma"),
        ("low", "high"),
        "a normal dispersion takes sigma, and optionally low and high",
    ),
    "uniform": _Keys(
        ("dist_type",),
        ("half_width", "low", "high"),
        "a uniform dispersion takes half_width, or low and high",
    ),
}

# Keys of a dispersion that hold its numbers, and those that are differences
# from the nominal value rather than values of it
_SPREAD_KEYS = ("sigma", "half_width", "low", "high")
_DIFFERENCE_KEYS = ("sigma", "half_width")

# Sigmas below its nominal value of 1 that an aero factor's normal
# distribution without low stays above 0
_FACTOR_SIGMAS = 5.0

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


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
    # "vehicle.toml:dry_mass_properties.mass"
    key: str
    target: Target
    nominal: float  # Value in the base sim; 1 for an aero factor
    distribution: Distribution

    # Units of the nominal value as its file writes them, such as "kg"; ""
    # for a plain number or an aero factor
    units: str


@dataclass(frozen=True)
class EdithConfig:
    """Monte Carlo campaign of dispersed flights of one base sim.

    ``from_file`` reads one from a campaign file. Its paths are absolute, and
    the optional settings the file leaves out are None.
    """

    path: Path  # The campaign file
    trials: int  # N in the campaign file
    master_seed: int
    base_sim: Path
    output_dir: Path
    dispersions: tuple[Dispersion, ...]  # In file order
    sampling: Sampling = "lhs"
    workers: int | None = None
    batch_size: int | None = None  # Trials per batch
    trial_timeout_s: float | None = None

    @property
    def base_folder(self) -> Path:
        """Return the base sim's folder, which holds every file it reads."""
        return self.base_sim.parent

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
        table = read_edith_config(path)
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
            EdithConfigError: If a key is missing, unknown or invalid, or a
                dispersion names no value of the base sim. The message names
                the key at fault.
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
        _check_outside(path, base_sim.parent, output_dir)
        return cls(
            path=path,
            trials=trials,
            master_seed=master_seed,
            base_sim=base_sim,
            output_dir=output_dir,
            dispersions=_dispersions(table["dispersions"], BaseFolder(base_sim.parent)),
            sampling=sampling,
            workers=workers,
            batch_size=batch_size,
            trial_timeout_s=trial_timeout_s,
        )


def read_edith_config(path: str | Path) -> dict[str, object]:
    """Read a campaign file's table without checking it.

    Args:
        path (str | Path): Campaign file.

    Returns:
        dict[str, object]: The file's top-level table.

    Raises:
        EdithConfigError: If the file cannot be read or is not valid TOML.
    """
    path = Path(path)
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
    if not _BARE_KEY.fullmatch(key):
        key = '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return f"{where}.{key}" if where else key


def _did_you_mean(key: str, known: Sequence[str]) -> str:
    """Return a hint naming the known key closest to a misspelt one, if any."""
    same = [option for option in known if option.lower() == key.lower()]
    close = same or difflib.get_close_matches(key, known, n=1)
    return f" (did you mean {close[0]}?)" if close else ""


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
            hint = _did_you_mean(key, known)
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
    if isinstance(value, str) and value in _SAMPLINGS:
        return _SAMPLINGS[value]
    raise EdithConfigError(f'sampling: expected "lhs" or "random", got {value!r}')


def _convert(number: float, units: str, to_units: str, difference: bool) -> float:
    """Return a number in other units of the same kind.

    Units are of the same kind when their root units are the same, which keeps
    angles apart from plain numbers, though Pint converts "5 percent" to
    0.05 rad. A difference, such as a sigma, converts without the units'
    offsets: a sigma of 18 degF is 10 degC, or 10 delta_degC, where the
    temperature 18 degF is -7.8 degC.

    Raises:
        DimensionalityError: If the units are of different kinds.
    """
    root = ureg.get_root_units(units)[1]
    if root != ureg.get_root_units(to_units)[1]:
        raise DimensionalityError(units, to_units)
    if not difference:
        return float(scalar(number, units).m_as(to_units))
    return number * (_step(units, str(root)) / _step(to_units, str(root)))


def _step(units: str, root: str) -> float:
    """Return the size of one of ``units`` in its root units, without offset."""
    return float(scalar(1.0, units).m_as(root)) - float(scalar(0.0, units).m_as(root))


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
            seconds = _convert(quantity[0], quantity[1], "s", False)
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


def _check_outside(campaign: Path, base_folder: Path, output_dir: Path) -> None:
    """Check the campaign file and output folder lie outside the base sim's folder.

    Every trial copies the base sim's folder, so neither may be inside it, and
    the output folder may not hold it either.

    Raises:
        EdithConfigError: Naming the key whose path breaks this.
    """
    folder = display_path(base_folder)
    if campaign.is_relative_to(base_folder):
        raise EdithConfigError(
            f"base_sim: the campaign file is inside {folder}, the base sim's "
            "folder, which every trial copies; move the campaign file to a "
            "folder of its own"
        )
    if output_dir.is_relative_to(base_folder):
        raise EdithConfigError(
            f"output_dir: {display_path(output_dir)} is inside {folder}, the "
            "base sim's folder, which every trial copies"
        )
    if base_folder.is_relative_to(output_dir):
        raise EdithConfigError(
            f"output_dir: {display_path(output_dir)} holds {folder}, the base "
            "sim's folder; pick a folder beside it"
        )


class _Number(NamedTuple):
    """One of a dispersion's numbers, in its nominal value's units."""

    value: float
    text: str  # As the campaign file writes it, such as "2 cm"


def _dispersions(value: object, base: BaseFolder) -> tuple[Dispersion, ...]:
    """Read every dispersion table, checking no two change the same value.

    Raises:
        EdithConfigError: If there are none, one is invalid or two change the
            same value.
    """
    if not isinstance(value, dict) or not value:
        raise EdithConfigError(
            "dispersions: expected one or more tables, such as "
            '[dispersions."vehicle.toml:dry_mass_properties.mass"]'
        )
    dispersions = tuple(_dispersion(key, item, base) for key, item in value.items())
    for index, first in enumerate(dispersions):
        for second in dispersions[index + 1 :]:
            shared = _shared(first.target, second.target)
            if shared:
                raise EdithConfigError(
                    f"{_key_path('dispersions', second.key)}: changes {shared}, "
                    f"which {_key_path('dispersions', first.key)} changes too"
                )
    return dispersions


def _shared(first: Target, second: Target) -> str:
    """Return what two targets both change, or "" if nothing."""
    if first.file != second.file:
        return ""
    if isinstance(first, TomlTarget) and isinstance(second, TomlTarget):
        same = first.path == second.path
        return f"{first.file}:{value_path(first.path)}" if same else ""
    if isinstance(first, CsvTarget) and isinstance(second, CsvTarget):
        columns = [column for column in first.columns if column in second.columns]
        return f"the {columns[0]} column of {first.file}" if columns else ""
    return ""


def _dispersion(key: str, value: object, base: BaseFolder) -> Dispersion:
    """Read one dispersion table, whose key names the value it disperses.

    Raises:
        EdithConfigError: If the key names no value of the base sim, or the
            table does not describe a distribution of it.
    """
    where = _key_path("dispersions", key)
    if not isinstance(value, dict):
        raise EdithConfigError(
            f"{where}: expected a table of dist_type and its spread, got {value!r}"
        )
    target, nominal = base.target(key, where)
    dist_type = _dist_type(value, where)
    keys = _DISTRIBUTION_KEYS[dist_type]
    _check_keys(value, where, keys.required, keys.optional, keys.usage)
    numbers = {
        name: _number(value[name], name, _key_path(where, name), nominal)
        for name in _SPREAD_KEYS
        if name in value
    }
    distribution: Distribution = (
        _normal(numbers, nominal, where)
        if dist_type == "normal"
        else _uniform(numbers, nominal, where, keys.usage)
    )
    if isinstance(target, CsvTarget):
        _check_factor(distribution, where)
    return Dispersion(key, target, nominal.value, distribution, nominal.units)


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


def _number(value: object, name: str, where: str, nominal: Nominal) -> _Number:
    """Return one of a dispersion's numbers, converted to its nominal value's units.

    A nominal value with units, such as "17 kg", takes a number with units of
    the same kind; a plain number, or an aero factor, takes a plain number.

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
        converted = _convert(
            number, quantity[1], nominal.units, name in _DIFFERENCE_KEYS
        )
    except DimensionalityError as error:
        raise EdithConfigError(
            f'{where}: "{text}" does not convert to {nominal.units}, the units '
            f"of the nominal value, {nominal.text}"
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


def _check_factor(distribution: Distribution, where: str) -> None:
    """Check an aero factor's distribution stays above 0.

    Raises:
        EdithConfigError: If its low end is 0 or less, or a normal without low
            comes within five sigma of 0.
    """
    if isinstance(distribution, Normal) and distribution.low is None:
        if _FACTOR_SIGMAS * distribution.sigma >= 1.0:
            raise EdithConfigError(
                f"{_key_path(where, 'sigma')}: {distribution.sigma:g} puts 0 "
                f"within {_FACTOR_SIGMAS:g} sigma of the factor's nominal value "
                "of 1; set low above 0 as well, such as low = 0.5"
            )
    elif distribution.low is not None and distribution.low <= 0.0:
        raise EdithConfigError(
            f"{where}: the factor's low end is {distribution.low:g}; keep it "
            "above 0, or the coefficients vanish or change sign"
        )
