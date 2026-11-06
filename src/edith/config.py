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
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

from edith.dispersion import DISPERSION_CLASSES, BaseDispersion, StandardDispersion
from edith.errors import EdithConfigError, quoted_key_path
from edith.targets import load_base_sim
from flight_sim.config import ConfigError, InputSource, build, load_toml
from flight_sim.config.toml_files import display_path
from flight_sim.units import Scalar, UnitChecked

__all__ = ["CampaignSettings", "EdithConfig", "Sampling"]

Sampling: TypeAlias = Literal["lhs", "random"]

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
    dispersions: tuple[BaseDispersion, ...]  # In file order
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
