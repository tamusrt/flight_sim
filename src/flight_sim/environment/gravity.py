"""Environment module containing gravity data"""

from flight_sim.units import Scalar, scalar


def get_gravity(latitude: Scalar, longitude: Scalar, altitude: Scalar) -> Scalar:
    """Calculate the acceleration of gravity at a point.

    Args:
        latitude (Scalar): Geodetic latitude of the vehicle.
        longitude (Scalar): Geodetic longitude of the vehicle.
        altitude (Scalar): Height of the vehicle above the reference surface.

    Returns:
        Scalar: Magnitude of the gravitational acceleration.
    """
    # pylint: disable=unused-argument
    return scalar(9.81, "m/s**2")  # Placeholder for now, will implement later
