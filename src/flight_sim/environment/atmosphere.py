"""Atmosphere models giving the air conditions at an altitude."""

import bisect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple


class AtmosphereConditions(NamedTuple):
    """Air conditions at one altitude, as plain SI values."""

    temperature: float  # K
    pressure: float  # Pa
    air_density: float  # kg/m**3
    speed_of_sound: float  # m/s


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


@dataclass
class StandardAtmosphere1976(AtmosphereModel):
    """US Standard Atmosphere 1976 up to 86 km geometric altitude.

    Conditions are held at their values at the ends of that range.
    """

    def conditions(self, altitude_m: float) -> AtmosphereConditions:
        """Return the air conditions at a geometric altitude in metres."""
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


# Standard tropopause: geopotential altitude in m and temperature in K
_TROPOPAUSE_M = _LAYER_BASE_M[1]
_TROPOPAUSE_TEMPERATURE_K = _LAYER_BASE_TEMPERATURE_K[1]


@dataclass
class LaunchSiteAtmosphere(AtmosphereModel):
    """Atmosphere built from the temperature and pressure measured at the pad.

    Altitudes passed in are above sea level, as for any atmosphere model; the
    pad elevation only anchors the measured pad temperature and pressure. Up to
    the standard tropopause (11 km geopotential) the temperature falls
    linearly from the pad temperature to the standard tropopause
    temperature, and the pressure follows hydrostatically from the pad
    pressure. Above it the US Standard Atmosphere 1976 is used, with its
    pressure scaled to stay continuous. This is OpenRocket's "extended ISA"
    model, so a hot pad warms the low air without heating the whole column.
    """

    # Height of the pad above sea level in metres
    pad_elevation_m: float = 0.0

    # Air temperature in K and pressure in Pa at the pad
    pad_temperature_k: float = _SEA_LEVEL_TEMPERATURE_K
    pad_pressure_pa: float = _SEA_LEVEL_PRESSURE_PA

    def _troposphere(self, height_m: float) -> tuple[float, float]:
        """Return temperature in K and pressure in Pa at a geopotential height."""
        pad_height = _geopotential(self.pad_elevation_m)
        lapse = (_TROPOPAUSE_TEMPERATURE_K - self.pad_temperature_k) / (
            _TROPOPAUSE_M - pad_height
        )
        temperature = self.pad_temperature_k + lapse * (height_m - pad_height)
        if abs(lapse) < 1e-12:
            pressure = self.pad_pressure_pa * math.exp(
                -_G0 * (height_m - pad_height) / (_AIR_GAS_CONSTANT * temperature)
            )
        else:
            pressure = self.pad_pressure_pa * (
                temperature / self.pad_temperature_k
            ) ** (-_G0 / (_AIR_GAS_CONSTANT * lapse))
        return temperature, pressure

    def conditions(self, altitude_m: float) -> AtmosphereConditions:
        """Return the air conditions at a geometric altitude above sea level."""
        sea_level_altitude = min(max(altitude_m, _MIN_ALTITUDE_M), _MAX_ALTITUDE_M)
        height = _geopotential(sea_level_altitude)
        if height <= _TROPOPAUSE_M:
            temperature, pressure = self._troposphere(height)
        else:
            standard = StandardAtmosphere1976().conditions(sea_level_altitude)
            _, model_tropopause = self._troposphere(_TROPOPAUSE_M)
            temperature = standard.temperature
            pressure = standard.pressure * (
                model_tropopause / _LAYER_BASE_PRESSURE_PA[1]
            )
        return AtmosphereConditions(
            temperature=temperature,
            pressure=pressure,
            air_density=pressure / (_AIR_GAS_CONSTANT * temperature),
            speed_of_sound=math.sqrt(
                _HEAT_CAPACITY_RATIO * _AIR_GAS_CONSTANT * temperature
            ),
        )


def _geopotential(altitude_m: float) -> float:
    """Return the geopotential altitude in m for a geometric one."""
    return _EARTH_RADIUS_M * altitude_m / (_EARTH_RADIUS_M + altitude_m)
