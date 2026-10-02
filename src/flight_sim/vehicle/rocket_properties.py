"""Data structures defining physical vehicle properties"""

from dataclasses import dataclass, field

from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine, solid_engine_from_csv


@dataclass
class RocketProperties:
    """Aerodynamic and motor properties of the rocket"""

    aero_table: AeroTable
    motor_file_path: str
    propellant_mass: float  # Total weight of solid fuel in kg

    engine: Engine = field(init=False)

    def __post_init__(self) -> None:
        self.engine = solid_engine_from_csv(self.motor_file_path, self.propellant_mass)
