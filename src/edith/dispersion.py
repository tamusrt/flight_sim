"""Dispersions of a campaign: which value of the base sim, and how trials draw it.

``edith.config`` builds these from a campaign file. A campaign names its
dispersion class by the name of a ``BaseDispersion`` subclass in this module,
and its distribution by the ``DIST_TYPE`` of a class in
``edith.statistics.distributions``, which this module re-exports. A
dispersion's numbers are all in the units its nominal value is written in.
"""

from dataclasses import dataclass

from edith.statistics.distributions import (
    DISTRIBUTIONS,
    Distribution,
    Normal,
    Parameter,
    ParameterError,
    Uniform,
)

__all__ = [
    "DISTRIBUTIONS",
    "STANDARD_DISPERSION",
    "BaseDispersion",
    "Dispersion",
    "Distribution",
    "Normal",
    "Parameter",
    "ParameterError",
    "StandardDispersion",
    "Uniform",
]


class BaseDispersion:
    """Base of the dispersion classes a campaign may name.

    A campaign's ``dispersion = "<name>"`` must name a subclass of this in
    this module, such as StandardDispersion.
    """


class StandardDispersion(BaseDispersion):
    """Dispersion class that draws a number from a normal or uniform distribution.

    The default, and for now the only one.
    """


# The default dispersion class's name, as a campaign file writes it
STANDARD_DISPERSION = StandardDispersion.__name__


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

    # Units of the nominal value as the input writes them, such as "kg"; "" for
    # a plain number
    units: str

    # Class that draws its values, as its dispersion key names it; the
    # standard one when the key is left out
    dispersion_class: str = STANDARD_DISPERSION
