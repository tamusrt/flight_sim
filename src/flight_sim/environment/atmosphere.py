from dataclasses import dataclass

@dataclass
class AtmosphereData:
    air_density: float = 0.0
    speed_of_sound: float = 0.0
    x_wind_vel: float = 0.0
    y_wind_vel: float = 0.0
    z_wind_vel: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0
