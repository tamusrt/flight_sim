"""Data structures defining physical vehicle properties"""

import math
from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import Scalar, UnitChecked
from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)
from flight_sim.vehicle.reynolds_correction import ReynoldsCorrection


@dataclass
class RocketProperties(UnitChecked):
    """Aerodynamics, motor and dry mass properties of the rocket."""

    aero_table: AeroTable
    engine: Engine

    # Everything but the motor
    dry_mass_properties: MassProperties

    enable_pitch_damping: bool
    length: Annotated[Scalar, "m"]
    num_fins: int
    fin_area: Annotated[Scalar, "m**2"]
    fin_pos: Annotated[Scalar, "m"]

    # Corrects the aero table's Cx for the flight's Reynolds number; None
    # flies the table as it is
    reynolds_correction: ReynoldsCorrection | None = None

    length_m: float = field(init=False, repr=False, compare=False)
    fin_area_m2: float = field(init=False, repr=False, compare=False)
    fin_pos_m: float = field(init=False, repr=False, compare=False)

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the whole rocket's mass properties at a simulation time in s."""
        return combine(self.dry_mass_properties.si, self.engine.mass_properties(time))

    def __post_init__(self) -> None:
        """Check the shape and cache it in SI units."""
        super().__post_init__()
        length_m = float(self.length.m_as("m"))
        fin_area_m2 = float(self.fin_area.m_as("m**2"))
        fin_pos_m = float(self.fin_pos.m_as("m"))
        if not (math.isfinite(length_m) and length_m > 0.0):
            raise ValueError(f"length must be finite and positive, got {length_m} m")
        if self.num_fins < 0:
            raise ValueError(f"num_fins must be 0 or more, got {self.num_fins}")
        if not (math.isfinite(fin_area_m2) and fin_area_m2 >= 0.0):
            raise ValueError(
                f"fin_area must be finite and 0 or more, got {fin_area_m2} m**2"
            )
        if not -length_m <= fin_pos_m <= 0.0:
            raise ValueError(
                f"fin_pos must lie on the body, from 0 m at the nose tip to"
                f" -{length_m} m at its aft end (aft is -X), got {fin_pos_m} m"
            )
        self.length_m = length_m
        self.fin_area_m2 = fin_area_m2
        self.fin_pos_m = fin_pos_m
