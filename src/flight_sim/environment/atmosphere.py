"""Atmosphere models giving the air conditions at an altitude."""

import bisect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, NamedTuple

from flight_sim.units import Scalar, UnitChecked, scalar


class AtmosphereConditions(NamedTuple):
    """Air conditions at one altitude, as plain floats in the annotated units."""

    temperature: Annotated[float, "K"]
    pressure: Annotated[float, "Pa"]
    air_density: Annotated[float, "kg/m**3"]
    speed_of_sound: Annotated[float, "m/s"]


class AtmosphereModel(ABC):
    """Atmosphere the integrator samples at each altitude it visits."""

    @abstractmethod
    def conditions(self, altitude_m: float) -> AtmosphereConditions:
        """Return the air conditions at an altitude.

        Args:
            altitude_m (float): Geometric altitude in metres.

        Returns:
            AtmosphereConditions: Temperature, pressure, density and speed
                of sound at that altitude.
        """


# US Standard Atmosphere 1976 constants
_G0 = 9.80665  # m/s**2
_GAS_CONSTANT = 8.31432  # J/(mol*K)
_MOLAR_MASS = 0.0289644  # kg/mol
_AIR_GAS_CONSTANT = _GAS_CONSTANT / _MOLAR_MASS  # J/(kg*K)
_HEAT_CAPACITY_RATIO = 1.4
_EARTH_RADIUS_M = 6356766.0  # For the geometric to geopotential conversion
_SEA_LEVEL_TEMPERATURE_K = 288.15
_SEA_LEVEL_PRESSURE_PA = 101325.0

# Geopotential altitude at the base of each layer and the layer's lapse rate.
# The troposphere formula extends below the table down to the minimum
# altitude, and the table ends at the maximum altitude, 86 km geometric.
_LAYER_BASE_M = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
_LAYER_LAPSE_K_PER_M = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002)
_MIN_ALTITUDE_M = -5000.0
_MAX_ALTITUDE_M = 86000.0


def _layer_bases() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Integrate the layer lapse rates up from sea level.

    Returns:
        tuple[tuple[float, ...], tuple[float, ...]]: Temperature in K and
            pressure in Pa at the base of each layer.
    """
    temperatures = [_SEA_LEVEL_TEMPERATURE_K]
    pressures = [_SEA_LEVEL_PRESSURE_PA]
    for index in range(len(_LAYER_BASE_M) - 1):
        thickness = _LAYER_BASE_M[index + 1] - _LAYER_BASE_M[index]
        temperature = temperatures[index] + _LAYER_LAPSE_K_PER_M[index] * thickness
        temperatures.append(temperature)
        pressures.append(
            _layer_pressure(
                temperature, thickness, pressures[index], temperatures[index], index
            )
        )
    return tuple(temperatures), tuple(pressures)


def _layer_pressure(
    temperature: float,
    height_above_base: float,
    base_pressure: float,
    base_temperature: float,
    layer: int,
) -> float:
    """Return the pressure within one layer in Pa.

    Args:
        temperature (float): Temperature at the point in K.
        height_above_base (float): Geopotential height above the layer base
            in metres.
        base_pressure (float): Pressure at the layer base in Pa.
        base_temperature (float): Temperature at the layer base in K.
        layer (int): Index of the layer.

    Returns:
        float: Pressure in Pa from the gradient formula, or the isothermal
            formula in a layer with zero lapse rate.
    """
    lapse = _LAYER_LAPSE_K_PER_M[layer]
    if lapse == 0.0:
        return base_pressure * math.exp(
            -_G0 * _MOLAR_MASS * height_above_base / (_GAS_CONSTANT * base_temperature)
        )
    return base_pressure * math.pow(
        temperature / base_temperature, -_G0 * _MOLAR_MASS / (_GAS_CONSTANT * lapse)
    )


_LAYER_BASE_TEMPERATURE_K, _LAYER_BASE_PRESSURE_PA = _layer_bases()


def _standard_conditions(altitude_m: float) -> AtmosphereConditions:
    """Return the standard day's air conditions at a geometric altitude in metres."""
    altitude = min(max(altitude_m, _MIN_ALTITUDE_M), _MAX_ALTITUDE_M)
    height = _EARTH_RADIUS_M * altitude / (_EARTH_RADIUS_M + altitude)
    layer = max(bisect.bisect_right(_LAYER_BASE_M, height) - 1, 0)

    base_temperature = _LAYER_BASE_TEMPERATURE_K[layer]
    height_above_base = height - _LAYER_BASE_M[layer]
    temperature = base_temperature + _LAYER_LAPSE_K_PER_M[layer] * height_above_base
    pressure = _layer_pressure(
        temperature,
        height_above_base,
        _LAYER_BASE_PRESSURE_PA[layer],
        base_temperature,
        layer,
    )
    return AtmosphereConditions(
        temperature=temperature,
        pressure=pressure,
        air_density=pressure / (_AIR_GAS_CONSTANT * temperature),
        speed_of_sound=math.sqrt(
            _HEAT_CAPACITY_RATIO * _AIR_GAS_CONSTANT * temperature
        ),
    )


@dataclass
class StandardAtmosphere1976(AtmosphereModel, UnitChecked):
    """US Standard Atmosphere 1976 up to 86 km geometric altitude.

    Conditions are held at their values at the ends of that range. As in
    RASAero II, temperature and pressure follow the standard variation scaled
    to pass through the launch site's values, and humidity lowers the density.
    The defaults give the standard day. ``anchor`` sets the site's altitude,
    which ``TruthConfiguration`` takes from its launch elevation.
    """

    # None takes the standard day's temperature at the launch site
    launch_temperature: Annotated[Scalar | None, "K"] = None

    # Site pressure reduced to sea level along the standard atmosphere, as
    # weather stations and airfields report it
    barometric_pressure: Annotated[Scalar, "Pa"] = field(
        default_factory=lambda: scalar(_SEA_LEVEL_PRESSURE_PA, "Pa")
    )

    # Relative humidity at the launch site, from 0 to 1
    relative_humidity: float = 0.0

    # Geometric altitude of the launch site in m
    _launch_altitude_m: float = field(default=0.0, init=False, repr=False)

    # Site value over the standard value, for temperature and pressure
    _scales: tuple[float, float] = field(init=False, repr=False, compare=False)
    _vapour_pressure: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the humidity and anchor the site conditions at sea level.

        Raises:
            ValueError: If the relative humidity is outside 0 to 1.
        """
        super().__post_init__()
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative_humidity must be from 0 to 1, not a percentage")
        self.anchor(self._launch_altitude_m)

    def anchor(self, launch_altitude_m: float) -> None:
        """Take the site conditions as being at a launch altitude in metres."""
        self._launch_altitude_m = launch_altitude_m
        standard = _standard_conditions(launch_altitude_m)
        temperature = (
            standard.temperature
            if self.launch_temperature is None
            else float(self.launch_temperature.m_as("K"))
        )
        pressure = (
            float(self.barometric_pressure.m_as("Pa"))
            * standard.pressure
            / _SEA_LEVEL_PRESSURE_PA
        )
        # Water vapour pressure at the site from the Tetens formula
        celsius = temperature - 273.15
        self._vapour_pressure = (
            self.relative_humidity
            * 610.78
            * 10.0 ** (7.5 * celsius / (celsius + 237.3))
        )
        self._scales = (
            temperature / standard.temperature,
            pressure / standard.pressure,
        )

    def conditions(self, altitude_m: float) -> AtmosphereConditions:
        """Return the air conditions at a geometric altitude in metres."""
        standard = _standard_conditions(altitude_m)
        temperature = standard.temperature * self._scales[0]
        pressure = standard.pressure * self._scales[1]
        # Hann's formula: vapour pressure falls tenfold every 6.3 km, faster
        # than air pressure. Vapour weighs 0.622 of dry air at the same pressure.
        vapour_pressure = self._vapour_pressure * 10.0 ** (
            -(altitude_m - self._launch_altitude_m) / 6300.0
        )
        return AtmosphereConditions(
            temperature=temperature,
            pressure=pressure,
            air_density=(pressure - 0.378 * vapour_pressure)
            / (_AIR_GAS_CONSTANT * temperature),
            speed_of_sound=math.sqrt(
                _HEAT_CAPACITY_RATIO * _AIR_GAS_CONSTANT * temperature
            ),
        )


@dataclass
class VacuumAtmosphere(AtmosphereModel):
    """Atmosphere with no air, so the vehicle feels no aerodynamic loads."""

    # A finite speed of sound keeps the Mach number finite
    speed_of_sound_m_s: float = 340.29

    def conditions(self, altitude_m: float) -> AtmosphereConditions:
        """Return zero pressure and density at any altitude."""
        return AtmosphereConditions(
            temperature=0.0,
            pressure=0.0,
            air_density=0.0,
            speed_of_sound=self.speed_of_sound_m_s,
        )
