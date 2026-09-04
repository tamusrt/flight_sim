"""Environment module containing gravity data"""


# pylint: disable=unused-argument
def get_gravity(latitude: float, longitude: float, altitude: float) -> float:
    """Calculates acceleration of gravity based on latitude, longitude, and altitude"""
    return 9.81  # Placeholder for now, will implement later
