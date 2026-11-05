"""Dispersions of a campaign: which value of the base sim, and how trials draw it.

``edith.config`` builds these from a campaign file. A dispersion's numbers are
all in the units its nominal value is written in.
"""

from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "STANDARD_DISPERSION",
    "Dispersion",
    "Distribution",
    "Normal",
    "Uniform",
]

# The dispersion class that draws a number of the base sim from a normal or
# uniform distribution; for now the only one, and the default
STANDARD_DISPERSION = "StandardDispersion"


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

    # Key path of the value in the base sim's resolved input, as
    # InputSource.table_with takes it, such as
    # "properties.dry_mass_properties.mass"
    key: str
    nominal: float  # Value in the base sim
    distribution: Distribution

    # Units of the nominal value as the input writes them, such as "kg"; ""
    # for a plain number
    units: str

    # Class that draws its values, as its dispersion key names it; the
    # standard one when the key is left out
    dispersion_class: str = STANDARD_DISPERSION
