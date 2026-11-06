"""Read and check EDITH campaign files, such as edith_config.toml.

A campaign file names the base sim, the number of trials, the master seed and
the values to disperse, each by its key path in the base sim's resolved input,
as ``edith.targets`` describes. flight_sim's own loader reads the file and
builds its settings, as it builds a sim from a sim input file, so both kinds of
file are read, and their errors worded, the same way. A dispersion's optional
``dispersion`` key names the class in ``edith.dispersion`` that reads the rest
of its table and draws its values; for now that is always StandardDispersion,
the default.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

from edith.dispersion import DISPERSION_CLASSES, BaseDispersion, StandardDispersion
from edith.errors import EdithConfigError, check_keys, quoted_key_path
from edith.targets import load_base_sim
from flight_sim.config import ConfigError, InputSource, build, load_toml
from flight_sim.config.toml_files import display_path, untagged
from flight_sim.output import OutputConfiguration, fill
from flight_sim.simulation import Simulation
from flight_sim.units import Scalar, UnitChecked

__all__ = ["CampaignSettings", "EdithConfig", "InputMode", "Sampling", "output_values"]

Sampling: TypeAlias = Literal["lhs", "random"]
InputMode: TypeAlias = Literal["normal", "compressed", "lightweight"]

_DISPERSION_USAGE = (
    "for now every dispersion leaves dispersion out, or writes dispersion = "
    '"StandardDispersion"'
)

# Top-level tables of a campaign file read apart from its settings
_TABLES = ("dispersions", "output")

# Keys a campaign's output table takes: the base sim's output settings but
# the directory, which is each trial's run directory
_OUTPUT_KEYS = tuple(
    item.name for item in fields(OutputConfiguration) if item.name != "directory"
)
_OUTPUT_USAGE = (
    "output takes the keys of a sim's [output] table but directory: "
    f"{', '.join(_OUTPUT_KEYS)}"
)

# Smallest value of each whole-number setting
_MINIMUMS = {"num_sims": 1, "master_seed": 0, "batch_size": 1}


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
    # Gets the trials and their results; need not exist yet. Takes {name},
    # the base sim's, and {datetime}, as a sim's output directory does.
    output_dir: Path
    sampling: Sampling = "lhs"
    batch_size: int | None = None  # Trials per batch
    trial_timeout: Annotated[Scalar | None, "s"] = None
    input_mode: InputMode = "normal"

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
        try:
            fill(str(self.output_dir), "", datetime.now())
        except (AttributeError, IndexError, KeyError, ValueError) as error:
            raise ValueError(
                f"output_dir {self.output_dir} has a bad placeholder ({error}); "
                "use {name}, {datetime} or {datetime:%Y%m%d}"
            ) from error
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
    paths made absolute and ``output_dir``'s placeholders filled, the values
    it disperses, ``source``, the base sim's
    resolved input, which the dispersions' keys index and trials are written
    from, and ``output``, the file's output table, which every trial takes
    over the base sim's.
    """

    path: Path  # The campaign file
    dispersions: tuple[BaseDispersion, ...]  # In file order
    source: InputSource = field(repr=False)

    # flight_sim output settings of every trial, by key, such as {"plots": []}
    output: dict[str, Any] = field(default_factory=dict)

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
            settings = replace(
                settings, output_dir=_filled(settings.output_dir, source)
            )
            dispersions = _dispersions(table["dispersions"], source)
            output = _output(table["output"], source) if "output" in table else {}
        except EdithConfigError as error:
            raise EdithConfigError(f"{display_path(path)}: {error}") from error
        return cls(
            **{item.name: getattr(settings, item.name) for item in fields(settings)},
            path=path,
            dispersions=dispersions,
            source=source,
            output=output,
        )


def _filled(output_dir: Path, source: InputSource) -> Path:
    """Return the output directory with its {name} and {datetime} filled."""
    name = str(source.table.get("name", ""))
    return Path(fill(str(output_dir), name, datetime.now())).resolve()


def output_values(output: Mapping[str, Any]) -> dict[str, Any]:
    """Return output settings as the key paths ``InputSource.table_with`` takes."""
    return {f"output.{key}": value for key, value in output.items()}


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
            {key: item for key, item in table.items() if key not in _TABLES},
            CampaignSettings,
        )
    except ConfigError as error:
        raise EdithConfigError(str(error)) from error
    return replace(
        settings,
        base_sim=settings.base_sim.resolve(),
        output_dir=settings.output_dir.resolve(),
    )


def _dispersions(value: object, source: InputSource) -> tuple[BaseDispersion, ...]:
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
                f"{quoted_key_path('dispersions', key)}: changes {dispersion.key}, "
                f"which {quoted_key_path('dispersions', first)} changes too"
            )
        dispersions.append(dispersion)
    return tuple(dispersions)


def _dispersion(key: str, value: object, source: InputSource) -> BaseDispersion:
    """Read one dispersion table with the class its ``dispersion`` key names.

    Raises:
        EdithConfigError: If the table names no dispersion class EDITH has, or
            that class cannot read it.
    """
    where = quoted_key_path("dispersions", key)
    if not isinstance(value, dict):
        raise EdithConfigError(
            f"{where}: expected a table of dispersion, dist_type and its spread, "
            f"got {value!r}"
        )
    return _dispersion_class(value, where).from_table(key, value, source, where)


def _dispersion_class(table: Mapping[str, object], where: str) -> type[BaseDispersion]:
    """Return the dispersion class a dispersion names, or the standard one.

    Raises:
        EdithConfigError: If ``dispersion`` is not a string, or not a key of
            ``DISPERSION_CLASSES``.
    """
    if "dispersion" not in table:
        return StandardDispersion
    name = table["dispersion"]
    where = quoted_key_path(where, "dispersion")
    if not isinstance(name, str):
        raise EdithConfigError(
            f"{where}: expected a string, got {name!r}; {_DISPERSION_USAGE}"
        )
    if name not in DISPERSION_CLASSES:
        raise EdithConfigError(
            f"{where}: EDITH has no dispersion class {name!r}; {_DISPERSION_USAGE}"
        )
    return DISPERSION_CLASSES[name]


def _output(value: object, source: InputSource) -> dict[str, Any]:
    """Return a campaign's output table, checked by building the base sim with it.

    Raises:
        EdithConfigError: If it is not a table, has a key a sim's output table
            does not take, names a directory, or the base sim does not build
            with it.
    """
    if not isinstance(value, dict):
        raise EdithConfigError(
            "output: expected a table of flight_sim output settings, such as "
            f"plots = [], got {value!r}"
        )
    if "directory" in value:
        raise EdithConfigError(
            "output.directory: each trial writes to its own run directory"
        )
    check_keys(value, "output", (), _OUTPUT_KEYS, _OUTPUT_USAGE)
    output: dict[str, Any] = untagged(value)
    try:
        Simulation.from_table(source.table_with(output_values(output)), source.path)
    except ConfigError as error:
        raise EdithConfigError(str(error)) from error
    return output
