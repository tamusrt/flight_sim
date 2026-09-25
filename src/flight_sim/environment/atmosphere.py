"""Environment module containing atmospheric data structures."""

import math
from dataclasses import dataclass, field

from flight_sim.units import Scalar, UnitChecked, Vector, scalar, zero_vector


@dataclass
class AtmosphereData(UnitChecked):
    """Atmospheric conditions at a specific altitude."""

    air_density: Scalar = field(default_factory=lambda: scalar(0.0, "kg/m**3"))
    speed_of_sound: Scalar = field(default_factory=lambda: scalar(0.0, "m/s"))
    temperature: Scalar = field(default_factory=lambda: scalar(0.0, "K"))
    pressure: Scalar = field(default_factory=lambda: scalar(0.0, "Pa"))
    wind_velocity: Vector = field(default_factory=lambda: zero_vector("m/s"))


_SEA_LEVEL_TEMPERATURE_K = 288.15
_SEA_LEVEL_PRESSURE_PA = 101325.0
_LAPSE_RATE_K_PER_M = -0.0065
_GAS_CONSTANT = 287.0
_GRAVITY = 9.81
_TROPOPAUSE_ALTITUDE_M = 11000.0
_PRESSURE_EXPONENT = -_GRAVITY / (_LAPSE_RATE_K_PER_M * _GAS_CONSTANT)

# Conditions at the 11km boundary, from the gradient equations
_TEMP_TROPOPAUSE = (
    _SEA_LEVEL_TEMPERATURE_K + _LAPSE_RATE_K_PER_M * _TROPOPAUSE_ALTITUDE_M
)
_PRESS_TROPOPAUSE = _SEA_LEVEL_PRESSURE_PA * (
    (_TEMP_TROPOPAUSE / _SEA_LEVEL_TEMPERATURE_K) ** _PRESSURE_EXPONENT
)


def get_atmosphere(altitude: Scalar) -> AtmosphereData:
    """Calculate atmospheric properties to first isothermal region (25 km)."""

    alt_m = max(0.0, float(altitude.m_as("m")))

    if alt_m < _TROPOPAUSE_ALTITUDE_M:
        # Gradient region math (Troposphere)
        temp = _SEA_LEVEL_TEMPERATURE_K + _LAPSE_RATE_K_PER_M * alt_m
        pressure = _SEA_LEVEL_PRESSURE_PA * (
            (temp / _SEA_LEVEL_TEMPERATURE_K) ** _PRESSURE_EXPONENT
        )
    else:
        # Isothermal region math (Lower Stratosphere and above fallback)
        temp = _TEMP_TROPOPAUSE
        pressure = _PRESS_TROPOPAUSE * math.exp(
            -(_GRAVITY / (_GAS_CONSTANT * temp)) * (alt_m - _TROPOPAUSE_ALTITUDE_M)
        )

    return AtmosphereData(
        air_density=scalar(pressure / (_GAS_CONSTANT * temp), "kg/m**3"),
        speed_of_sound=scalar((1.4 * _GAS_CONSTANT * temp) ** 0.5, "m/s"),
        temperature=scalar(temp, "K"),
        pressure=scalar(pressure, "Pa"),
        wind_velocity=zero_vector("m/s"),
    )
