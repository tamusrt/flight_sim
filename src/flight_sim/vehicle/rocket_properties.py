"""Data structures defining physical vehicle properties"""

from dataclasses import dataclass, field

from flight_sim.units import Scalar, Vector, vector
from flight_sim.utilities.data_loader import AeroTable, aero_table_from_csv
from flight_sim.vehicle.engine import Engine, solid_engine_from_csv


@dataclass
class RocketProperties:
    """Aerodynamic and motor properties of the rocket"""

    aero_file_path: str
    motor_file_path: str
    propellant_mass: float  # Total weight of solid fuel in kg

    reference_area: Scalar
    reference_diameter: Scalar

    reference_point: Vector = field(
        default_factory=lambda: vector((0.0, 0.0, 0.0), "m")
    )

    # Drag, lift (normal), side force, roll, pitch and yaw coefficients
    aero_coefficients: AeroTable = field(init=False)

    engine: Engine = field(init=False)

    def __post_init__(self) -> None:
        self.aero_coefficients = aero_table_from_csv(self.aero_file_path)
        self.engine = solid_engine_from_csv(self.motor_file_path, self.propellant_mass)
