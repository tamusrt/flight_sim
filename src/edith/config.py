"""Read and check EDITH campaign files, such as edith_config.toml.

A campaign file names the base sim, the number of trials, the master seed and
the values to disperse, each by its key path in the base sim's resolved input,
as ``edith.targets`` describes. flight_sim's own loader reads the file and
builds its settings, as it builds a sim from a sim input file, so both kinds of
file are read, and their errors worded, the same way. A dispersion's optional
``dispersion`` key names the class that draws its values; for now that is
always "StandardDispersion", the default, which draws a number from a normal or
uniform distribution. A dispersion's numbers are converted to the units its
nominal value is written in. ``edith.dispersion`` holds what is read, and the
dispersion and distribution classes a campaign may name, which this module
finds there rather than listing them itself.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

from pint import DimensionalityError, OffsetUnitCalculusError

from edith.dispersion import (
    DISPERSION_CLASSES,
    DISTRIBUTIONS,
    STANDARD_DISPERSION,
    Dispersion,
    Parameter,
    ParameterError,
)
from edith.errors import EdithConfigError, closest
from edith.targets import Nominal, find_target, load_base_sim
from flight_sim.config import ConfigError, InputSource, build, load_toml
from flight_sim.config.toml_files import display_path, key_path
from flight_sim.config.writing import toml_key
from flight_sim.units import (
    UNREADABLE_UNITS,
    Scalar,
    UnitChecked,
    convert,
    parse_quantity,
)

__all__ = ["CampaignSettings", "EdithConfig", "Sampling"]

Sampling: TypeAlias = Literal["lhs", "random"]

# The dist_type values a dispersion may take, as messages list them
_DIST_TYPES = " or ".join(f'"{name}"' for name in DISTRIBUTIONS)

_DISPERSION_USAGE = (
    "for now every dispersion leaves dispersion out, or writes dispersion = "
    '"StandardDispersion"'
)

# Smallest value of each whole-number setting
_MINIMUMS = {"num_sims": 1, "master_seed": 0, "workers": 1, "batch_size": 1}


@dataclass(frozen=True)
class CampaignSettings(UnitChecked):
    """Settings of a campaign file: every top-level key but ``dispersions``.

    flight_sim's loader builds this from the file as it builds a sim from a sim
    input file, checking each key's type, units and path and naming the key at
    fault. Paths are relative to the campaign file. ``__post_init__`` checks
    what the types alone do not.
    """

    num_sims: int  # Trials to run
    master_seed: int
    base_sim: Path  # The sim input file every trial disperses
    output_dir: Path  # Gets the trials and their results; need not exist yet
    sampling: Sampling = "lhs"
    workers: int | None = None
    batch_size: int | None = None  # Trials per batch
    trial_timeout: Annotated[Scalar | None, "s"] = None

    def __post_init__(self) -> None:
        """Check the settings' ranges, and the base sim and output folder.

        Raises:
            ValueError: Naming the setting at fault.
        """
        super().__post_init__()
        for name, minimum in _MINIMUMS.items():
            value = getattr(self, name)
            if value is not None and value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
        timeout = self.trial_timeout
        if timeout is not None and not 0.0 < timeout.m_as("s") < math.inf:
            raise ValueError(
                f'trial_timeout must be above 0, such as "10 min", got {timeout:~}'
            )
        for name in ("base_sim", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path, got {getattr(self, name)!r}")
        if self.base_sim.suffix.lower() != ".toml":
            raise ValueError(
                "base_sim must be a sim input file ending in .toml, got "
                f"{self.base_sim.name!r}"
            )
        if not self.base_sim.is_file():
            raise ValueError(
                f"base_sim {display_path(self.base_sim.resolve())} does not exist"
            )
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(
                f"output_dir {display_path(self.output_dir.resolve())} is a file"
            )


@dataclass(frozen=True, kw_only=True)
class EdithConfig(CampaignSettings):
    """Monte Carlo campaign of dispersed flights of one base sim.

    ``from_file`` reads one from a campaign file: its settings, with their
    paths made absolute, the values it disperses, and ``source``, the base
    sim's resolved input, which the dispersions' keys index and trials are
    written from.
    """

    path: Path  # The campaign file
    dispersions: tuple[Dispersion, ...]  # In file order
    source: InputSource = field(repr=False)

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
        try:
            table = load_toml(path)
        except ConfigError as error:  # Its message names the file already
            raise EdithConfigError(str(error)) from error
        try:
            settings = _settings(table)
            source = load_base_sim(settings.base_sim)
            dispersions = _dispersions(table["dispersions"], source)
        except EdithConfigError as error:
            raise EdithConfigError(f"{display_path(path)}: {error}") from error
        return cls(
            **{item.name: getattr(settings, item.name) for item in fields(settings)},
            path=path,
            dispersions=dispersions,
            source=source,
        )


def _settings(table: Mapping[str, Any]) -> CampaignSettings:
    """Build a campaign file's settings, with their paths made absolute.

    Args:
        table (Mapping[str, Any]): The file's table, as
            ``flight_sim.config.load_toml`` reads it, whose strings remember
            the file's folder for relative paths.

    Returns:
        CampaignSettings: Every top-level key but ``dispersions``.

    Raises:
        EdithConfigError: If ``dispersions`` is missing, the table has a
            ``type`` or ``ref``, or a setting is missing, unknown or invalid.
            The message names the key at fault.
    """
    if "dispersions" not in table:
        raise EdithConfigError(
            "missing dispersions, a table of each value to disperse, such as "
            '[dispersions."properties.dry_mass_properties.mass"]'
        )
    for key in ("type", "ref"):
        if key in table:
            raise EdithConfigError(f"{key}: a campaign file takes no {key}")
    try:
        settings: CampaignSettings = build(
            {key: item for key, item in table.items() if key != "dispersions"},
            CampaignSettings,
        )
    except ConfigError as error:
        raise EdithConfigError(str(error)) from error
    return replace(
        settings,
        base_sim=settings.base_sim.resolve(),
        output_dir=settings.output_dir.resolve(),
    )


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
    dispersions = []
    first_of: dict[str, str] = {}  # Campaign key of each key path's dispersion
    for key, item in value.items():
        dispersion = _dispersion(key, item, source)
        first = first_of.setdefault(dispersion.key, key)
        if first != key:
            raise EdithConfigError(
                f"{_key_path('dispersions', key)}: changes {dispersion.key}, "
                f"which {_key_path('dispersions', first)} changes too"
            )
        dispersions.append(dispersion)
    return tuple(dispersions)


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
    key_path_, nominal = find_target(source, key, where)
    distribution_class = DISTRIBUTIONS[_dist_type(value, where)]
    names = (*distribution_class.REQUIRED, *distribution_class.OPTIONAL)
    _check_keys(
        value,
        where,
        ("dist_type", *distribution_class.REQUIRED),
        ("dispersion", *distribution_class.OPTIONAL),
        distribution_class.USAGE,
    )
    parameters = {
        name: _number(value[name], _key_path(where, name), nominal)
        for name in names
        if name in value
    }
    try:
        distribution = distribution_class.from_parameters(parameters, nominal)
    except ParameterError as error:
        place = where if error.key is None else _key_path(where, error.key)
        raise EdithConfigError(f"{place}: {error}") from error
    return Dispersion(
        key_path_, nominal.value, distribution, nominal.units, dispersion_class
    )


def _dispersion_class(table: Mapping[str, object], where: str) -> str:
    """Return the dispersion class a dispersion names, or the standard one.

    Raises:
        EdithConfigError: If ``dispersion`` is not a string, or not a key of
            ``DISPERSION_CLASSES``.
    """
    if "dispersion" not in table:
        return STANDARD_DISPERSION
    name = table["dispersion"]
    where = _key_path(where, "dispersion")
    if not isinstance(name, str):
        raise EdithConfigError(
            f"{where}: expected a string, got {name!r}; {_DISPERSION_USAGE}"
        )
    if name not in DISPERSION_CLASSES:
        raise EdithConfigError(
            f"{where}: EDITH has no dispersion class {name!r}; {_DISPERSION_USAGE}"
        )
    return str(name)  # A plain string: load_toml's strings do not pickle


def _dist_type(table: Mapping[str, object], where: str) -> str:
    """Return a dispersion's distribution type, a key of ``DISTRIBUTIONS``.

    Raises:
        EdithConfigError: If it is missing, or names no distribution EDITH has.
    """
    if "dist_type" not in table:
        raise EdithConfigError(f"{where}: missing dist_type, {_DIST_TYPES}")
    dist_type = table["dist_type"]
    if not isinstance(dist_type, str) or dist_type not in DISTRIBUTIONS:
        raise EdithConfigError(
            f"{_key_path(where, 'dist_type')}: expected {_DIST_TYPES}, "
            f"got {dist_type!r}"
        )
    return dist_type


def _number(value: object, where: str, nominal: Nominal) -> Parameter:
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
        return Parameter(number, f"{value:g}")
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
    return Parameter(_finite(converted, where, value), text)


def _finite(number: float, where: str, value: object) -> float:
    """Return a number, checking it is finite.

    Raises:
        EdithConfigError: If it is infinite or NaN.
    """
    if not math.isfinite(number):
        raise EdithConfigError(f"{where}: expected a finite number, got {value!r}")
    return number
