"""Environment module containing atmospheric data structures."""

from dataclasses import dataclass, field

import numpy as np

from flight_sim.units import Scalar, UnitChecked, Vector, scalar, zero_vector


@dataclass
class AtmosphereData(UnitChecked):
    """Atmospheric conditions at a specific altitude."""

    air_density: Scalar = field(default_factory=lambda: scalar(0.0, "kg/m**3"))
    speed_of_sound: Scalar = field(default_factory=lambda: scalar(0.0, "m/s"))
    temperature: Scalar = field(default_factory=lambda: scalar(0.0, "K"))
    pressure: Scalar = field(default_factory=lambda: scalar(0.0, "Pa"))
    wind_velocity: Vector = field(default_factory=lambda: zero_vector("m/s"))


def get_atmosphere(altitude: Scalar) -> AtmosphereData:
    """Calculate atmospheric properties to first isothermal region (25 km)."""

    alt_m = max(0.0, float(altitude.m_as("m")))

    # Calculate conditions at the 11km boundary using gradient equations
    temp_tropopause = 288.15 + (-0.0065 * 11000.0)
    press_tropopause = 101325.0 * (
        (temp_tropopause / 288.15) ** (-9.81 / (-0.0065 * 287.0))
    )

    if alt_m < 11000.0:
        # Gradient region math (Troposphere)
        temp = 288.15 + (-0.0065 * alt_m)
        pressure = 101325.0 * ((temp / 288.15) ** (-9.81 / (-0.0065 * 287.0)))
    else:
        # Isothermal region math (Lower Stratosphere and above fallback)
        temp = temp_tropopause
        pressure = press_tropopause * np.exp(
            -(9.81 / (287.0 * temp)) * (alt_m - 11000.0)
        )

    return AtmosphereData(
        air_density=scalar(pressure / (287.0 * temp), "kg/m**3"),
        speed_of_sound=scalar((1.4 * 287.0 * temp) ** 0.5, "m/s"),
        temperature=scalar(temp, "K"),
        pressure=scalar(pressure, "Pa"),
        wind_velocity=zero_vector("m/s"),
    )
