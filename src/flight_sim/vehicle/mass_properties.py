"""Mass, centre of gravity and inertia of the vehicle through the burn."""

from dataclasses import dataclass

import numpy as np

from flight_sim.units import vector
from flight_sim.vehicle.rocket_state import RocketState


@dataclass(frozen=True)
class MassPropertiesTable:
    """Centre of gravity and inertia at ignition and at burnout.

    Between the two, both are interpolated linearly in the mass, which
    holds while the propellant burns down roughly uniformly in place.
    Lengths are in metres from the nose tip along the body X axis (negative
    aft), inertias are the body-axis diagonal in kg*m**2.
    """

    launch_mass_kg: float
    launch_cg_m: float
    launch_inertia_kg_m2: tuple[float, float, float]
    burnout_mass_kg: float
    burnout_cg_m: float
    burnout_inertia_kg_m2: tuple[float, float, float]

    def burned_fraction(self, mass_kg: float) -> float:
        """Return the share of the propellant burned at a mass, from 0 to 1."""
        span = self.launch_mass_kg - self.burnout_mass_kg
        if span <= 0.0:
            return 1.0
        return float(np.clip((self.launch_mass_kg - mass_kg) / span, 0.0, 1.0))

    def apply(self, state: RocketState) -> None:
        """Set the state's CG and inertia from its current mass, in place.

        Args:
            state (RocketState): State to update.
        """
        fraction = self.burned_fraction(float(state.current_mass.m_as("kg")))
        cg = self.launch_cg_m + fraction * (self.burnout_cg_m - self.launch_cg_m)
        launch = np.asarray(self.launch_inertia_kg_m2)
        burnout = np.asarray(self.burnout_inertia_kg_m2)
        state.cg_location = vector((cg, 0.0, 0.0), "m")
        state.inertia = vector(launch + fraction * (burnout - launch), "kg*m**2")
