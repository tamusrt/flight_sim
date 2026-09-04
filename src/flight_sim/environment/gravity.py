"""Environment module containing gravity data"""

def current_gravity(latitude: float, longitude: float, altitude: float) -> float:
    """Calculates acceleration of gravoty based on latitude, longitude, and altitude"""
    return 9.81 #Placeholder for now, will implement later