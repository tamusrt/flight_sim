"""Dispersions of a campaign: which value of the base sim, and how trials draw it.

``edith.config`` builds these from a campaign file. A campaign names its
dispersion class by the name of a ``BaseDispersion`` subclass in this module,
and its distribution by the ``DIST_TYPE`` of a class in ``Distribution``. Each
distribution class lists the parameters it takes and checks them itself, so a
new distribution is one new class here, added to ``Distribution``. A
dispersion's numbers are all in the units its nominal value is written in.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, Self, TypeAlias, get_args

from edith.targets import Nominal

__all__ = [
    "DISPERSION_CLASSES",
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

# Each dispersion class, by the name a campaign writes; adding a subclass of
# BaseDispersion adds it here
DISPERSION_CLASSES: dict[str, type[BaseDispersion]] = {
    cls.__name__: cls for cls in BaseDispersion.__subclasses__()
}


class Parameter(NamedTuple):
    """One of a distribution's parameters, in its nominal value's units."""

    value: float
    text: str  # As the campaign file writes it, such as "2 cm"


class ParameterError(ValueError):
    """Error in a distribution's parameters, naming the one at fault if one is."""

    def __init__(self, message: str, key: str | None = None) -> None:
        """Keep the message, and the key of the parameter at fault, if any."""
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class Normal:
    """Normal distribution about the dispersion's nominal value.

    ``low`` and ``high``, where given, truncate it on either side of the
    nominal value.
    """

    # How a campaign names it, the parameters it needs and takes, and how
    # errors say so
    DIST_TYPE: ClassVar[str] = "normal"
    REQUIRED: ClassVar[tuple[str, ...]] = ("sigma",)
    OPTIONAL: ClassVar[tuple[str, ...]] = ("low", "high")
    USAGE: ClassVar[str] = (
        "a normal dispersion takes sigma, and optionally low and high"
    )

    sigma: float
    low: float | None = None
    high: float | None = None

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution its parameters describe, checking them.

        Args:
            parameters (Mapping[str, Parameter]): sigma, and any of low and
                high, in the nominal value's units.
            nominal (Nominal): The value it disperses.

        Returns:
            Normal: The distribution.

        Raises:
            ParameterError: If sigma is negative, or low and high do not lie on
                either side of the nominal value.
        """
        sigma = parameters["sigma"]
        low, high = parameters.get("low"), parameters.get("high")
        if sigma.value < 0.0:
            raise ParameterError(f"expected 0 or more, got {sigma.text}", "sigma")
        side = "low and high truncate a normal distribution on either side of it"
        if low is not None and _above(low.value, nominal.value):
            raise ParameterError(
                f"{low.text} is above the nominal value, {nominal.text}; {side}", "low"
            )
        if high is not None and _above(nominal.value, high.value):
            raise ParameterError(
                f"{high.text} is below the nominal value, {nominal.text}; {side}",
                "high",
            )
        _check_order(low, high)
        return cls(
            sigma.value,
            None if low is None else low.value,
            None if high is None else high.value,
        )


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution from ``low`` to ``high``."""

    # How a campaign names it, the parameters it takes, and how errors say so
    DIST_TYPE: ClassVar[str] = "uniform"
    REQUIRED: ClassVar[tuple[str, ...]] = ()
    OPTIONAL: ClassVar[tuple[str, ...]] = ("half_width", "low", "high")
    USAGE: ClassVar[str] = "a uniform dispersion takes half_width, or low and high"

    low: float
    high: float

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Parameter], nominal: Nominal
    ) -> Self:
        """Return the distribution of half_width about the nominal, or low to high.

        Args:
            parameters (Mapping[str, Parameter]): half_width, or low and high,
                in the nominal value's units.
            nominal (Nominal): The value it disperses.

        Returns:
            Uniform: The distribution.

        Raises:
            ParameterError: If it has both forms or neither, half_width is
                negative, or low is not below high.
        """
        half_width = parameters.get("half_width")
        low, high = parameters.get("low"), parameters.get("high")
        if half_width is not None:
            if low is not None or high is not None:
                raise ParameterError(
                    f"give half_width, or low and high, not both; {cls.USAGE}"
                )
            if half_width.value < 0.0:
                raise ParameterError(
                    f"expected 0 or more, got {half_width.text}", "half_width"
                )
            return cls(
                nominal.value - half_width.value, nominal.value + half_width.value
            )
        if low is None and high is None:
            raise ParameterError(f"missing half_width, or low and high; {cls.USAGE}")
        if low is None or high is None:
            raise ParameterError(
                f"missing {'low' if low is None else 'high'}; {cls.USAGE}"
            )
        _check_order(low, high)
        return cls(low.value, high.value)


Distribution: TypeAlias = Normal | Uniform

# Each distribution class, by the dist_type a campaign names it with; adding a
# class to Distribution adds it here
DISTRIBUTIONS: dict[str, type[Distribution]] = {
    cls.DIST_TYPE: cls for cls in get_args(Distribution)
}


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


def _above(first: float, second: float) -> bool:
    """Return whether one number is above another by more than rounding."""
    return first > second and not math.isclose(first, second, abs_tol=1e-12)


def _check_order(low: Parameter | None, high: Parameter | None) -> None:
    """Check low is below high where both are given.

    Raises:
        ParameterError: If it is not.
    """
    if low is not None and high is not None and not low.value < high.value:
        raise ParameterError(f"low ({low.text}) must be below high ({high.text})")
