"""Data structures defining physical vehicle properties"""

from dataclasses import dataclass

from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)


@dataclass
class RocketProperties:
    """Aerodynamics, motor and dry mass properties of the rocket."""

    aero_table: AeroTable
    engine: Engine

    # Everything but the motor
    dry_mass_properties: MassProperties

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the whole rocket's mass properties at a simulation time in s."""
        return combine(self.dry_mass_properties.si, self.engine.mass_properties(time))
