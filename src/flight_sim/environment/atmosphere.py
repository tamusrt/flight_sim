"""Environment module containing atmospheric data structures."""

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
