"""Environment module containing atmospheric data structures."""

from dataclasses import dataclass, field

import numpy as np


def zero_array() -> np.ndarray:
    """Return a zero-valued three-dimensional array."""
    return np.zeros(3)


@dataclass
class AtmosphereData:
    """Dataclass representing atmospheric conditions at a specific altitude."""

    air_density: float = 0.0
    speed_of_sound: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0
    wind_velocity: np.ndarray = field(default_factory=zero_array)
