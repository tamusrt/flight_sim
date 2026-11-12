"""Aerodynamic, engine, mass and shape properties of the rocket."""

from dataclasses import dataclass

from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)
from flight_sim.vehicle.reynolds_correction import ReynoldsCorrection
from flight_sim.vehicle.rocket_shape import RocketShape


@dataclass
class RocketProperties:
    """Aerodynamics, motor, dry mass properties and shape of the rocket."""

    aero_table: AeroTable
    engine: Engine

    # Everything but the motor
    dry_mass_properties: MassProperties

    shape: RocketShape

    # Adds FS1.0's pitch and yaw damping, the cross-flow drag of the body and
    # fins as the rocket turns, to the aero table's torque
    enable_pitch_damping: bool

    # Corrects the aero table's Cx for the flight's Reynolds number; None
    # flies the table as it is
    reynolds_correction: ReynoldsCorrection | None = None

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the whole rocket's mass properties at a simulation time in s."""
        return combine(self.dry_mass_properties.si, self.engine.mass_properties(time))
