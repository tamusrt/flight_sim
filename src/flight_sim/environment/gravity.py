"""Environment module containing gravity data"""

from flight_sim.units import Scalar, scalar
import numpy as np

from flight_sim.units import Scalar, Vector, scalar

_EARTH_ROTATION_RATE = 7.292115e-5
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = 6.6943799901413165e-3
_WGS84_EQUATORIAL_GRAVITY = 9.7803253359
_WGS84_SOMIGLIANA_CONSTANT = 1.93185265241e-3
_WGS84_GM = 3.986004418e14


def get_gravity(latitude: Scalar, longitude: Scalar, altitude: Scalar) -> Scalar:
    """Return gravity for the integrator.

    Args:
        latitude (Scalar): Geodetic latitude.
        longitude (Scalar): Geodetic longitude.
        altitude (Scalar): Height above the reference ellipsoid.

    Returns:
        Scalar: Cached gravity magnitude or WGS84 normal gravity when no cache
            is active.
    """
    latitude_rad = float(latitude.m_as("rad"))
    height_m = float(altitude.m_as("m"))
    sin_latitude_squared = np.sin(latitude_rad) ** 2
    surface_gravity = _WGS84_EQUATORIAL_GRAVITY * (
        1.0 + _WGS84_SOMIGLIANA_CONSTANT * sin_latitude_squared
    ) / np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude_squared)
    semi_minor_axis_m = _WGS84_SEMI_MAJOR_AXIS_M * (1.0 - _WGS84_FLATTENING)
    rotation_parameter = (
        _EARTH_ROTATION_RATE**2
        * _WGS84_SEMI_MAJOR_AXIS_M**2
        * semi_minor_axis_m
        / _WGS84_GM
    )
    height_factor = 1.0 - (
        2.0
        / _WGS84_SEMI_MAJOR_AXIS_M
        * (
            1.0
            + _WGS84_FLATTENING
            + rotation_parameter
            - 2.0 * _WGS84_FLATTENING * sin_latitude_squared
        )
        * height_m
    ) + 3.0 * height_m**2 / _WGS84_SEMI_MAJOR_AXIS_M**2
    return scalar(float(surface_gravity * height_factor), "m/s**2")
