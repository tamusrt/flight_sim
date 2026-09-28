"""Gravity models giving the gravitational acceleration at a location."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

_EARTH_ROTATION_RATE = 7.292115e-5
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = 6.6943799901413165e-3
_WGS84_EQUATORIAL_GRAVITY = 9.7803253359
_WGS84_SOMIGLIANA_CONSTANT = 1.93185265241e-3
_WGS84_GM = 3.986004418e14
_WGS84_SEMI_MINOR_AXIS_M = _WGS84_SEMI_MAJOR_AXIS_M * (1.0 - _WGS84_FLATTENING)
_ROTATION_PARAMETER = (
    _EARTH_ROTATION_RATE**2
    * _WGS84_SEMI_MAJOR_AXIS_M**2
    * _WGS84_SEMI_MINOR_AXIS_M
    / _WGS84_GM
)


class GravityModel(ABC):
    """Gravity the integrator samples at each location it visits."""

    @abstractmethod
    def magnitude(self, latitude_rad: float, altitude_m: float) -> float:
        """Return the gravity magnitude in m/s**2.

        Args:
            latitude_rad (float): Geodetic latitude in radians.
            altitude_m (float): Height above the reference ellipsoid in metres.

        Returns:
            float: Gravity magnitude in m/s**2.
        """


@dataclass
class WGS84Gravity(GravityModel):
    """WGS84 normal gravity, varying with latitude and altitude."""

    def magnitude(self, latitude_rad: float, altitude_m: float) -> float:
        """Return the WGS84 normal gravity magnitude in m/s**2.

        Args:
            latitude_rad (float): Geodetic latitude in radians.
            altitude_m (float): Height above the reference ellipsoid in metres.

        Returns:
            float: Gravity magnitude in m/s**2.
        """
        sin_latitude_squared = math.sin(latitude_rad) ** 2
        surface_gravity = (
            _WGS84_EQUATORIAL_GRAVITY
            * (1.0 + _WGS84_SOMIGLIANA_CONSTANT * sin_latitude_squared)
            / math.sqrt(1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude_squared)
        )
        height_factor = (
            1.0
            - (
                2.0
                / _WGS84_SEMI_MAJOR_AXIS_M
                * (
                    1.0
                    + _WGS84_FLATTENING
                    + _ROTATION_PARAMETER
                    - 2.0 * _WGS84_FLATTENING * sin_latitude_squared
                )
                * altitude_m
            )
            + 3.0 * altitude_m**2 / _WGS84_SEMI_MAJOR_AXIS_M**2
        )
        return surface_gravity * height_factor


@dataclass
class ConstantGravity(GravityModel):
    """Gravity of one magnitude everywhere."""

    magnitude_m_s2: float

    def magnitude(self, latitude_rad: float, altitude_m: float) -> float:
        """Return the configured gravity magnitude in m/s**2."""
        return self.magnitude_m_s2
