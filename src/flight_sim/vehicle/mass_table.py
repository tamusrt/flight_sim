"""A rocket's mass, CG and inertia at ignition and at burnout, as engine parts.

OpenRocket gives the whole rocket's numbers at two moments: on the pad and
when the motor has burned out. The engine model wants the rocket as parts
instead: everything but the motor, the motor casing, and the propellant
grain that burns away. ``MassPropertiesTable`` turns the two moments into
those parts, so that the rocket weighs exactly what OpenRocket says at
ignition and at burnout, and in between the grain burns as the engine model
says it does.
"""

import math
from dataclasses import dataclass

import numpy as np

from flight_sim.units import matrix, scalar, vector
from flight_sim.vehicle.engine import PropellantGrain
from flight_sim.vehicle.mass_properties import MassProperties

# Share of the burnout mass that is called the casing. The engine always has a
# casing, since a motor with nothing left in it would have no mass at all; the
# total of the parts is the burnout mass whatever the share.
_CASING_SHARE = 0.01

# The core of a grain is never quite the whole of it, nor its length nothing
_MIN_LENGTH_M = 0.01
_MAX_CORE_SHARE = 0.99


@dataclass(frozen=True)
class MassPropertiesTable:
    """Centre of gravity and inertia at ignition and at burnout.

    Lengths are in metres from the nose tip along the body X axis (negative
    aft), inertias are the body-axis diagonal in kg*m**2. The inertia is
    about the rocket's CG at that moment.
    """

    launch_mass_kg: float
    launch_cg_m: float
    launch_inertia_kg_m2: tuple[float, float, float]
    burnout_mass_kg: float
    burnout_cg_m: float
    burnout_inertia_kg_m2: tuple[float, float, float]

    @property
    def propellant_mass_kg(self) -> float:
        """Mass of the propellant, which is what burns off."""
        return self.launch_mass_kg - self.burnout_mass_kg

    def parts(
        self, motor_diameter_m: float
    ) -> tuple[MassProperties, MassProperties, PropellantGrain]:
        """Split the rocket into the dry mass, the motor casing and the grain.

        The grain is where the propellant sat, so the CG is right at ignition
        and at burnout. It fills the motor's diameter. Its core and length are
        the ones whose inertia, with the parallel-axis terms, takes the burnout
        inertia to the ignition inertia: the pitch inertia is matched exactly,
        and the roll inertia as closely as a grain of that diameter can be.

        Args:
            motor_diameter_m (float): Outer diameter of the motor.

        Returns:
            tuple[MassProperties, MassProperties, PropellantGrain]: Everything
                but the motor, the casing, and the propellant. The dry mass
                and the casing together are the rocket at burnout.

        Raises:
            ValueError: If the rocket does not get lighter through the burn.
        """
        burnout = self.burnout_mass_kg
        propellant = self.propellant_mass_kg
        if propellant <= 0.0 or burnout <= 0.0:
            raise ValueError(
                f"The rocket must lose mass through the burn, got "
                f"{self.launch_mass_kg} kg at launch and {burnout} kg at burnout"
            )
        grain_cg = (
            self.launch_mass_kg * self.launch_cg_m - burnout * self.burnout_cg_m
        ) / propellant
        # Moving the grain's mass out to its own CG adds to the burnout inertia
        reduced = burnout * propellant / self.launch_mass_kg
        shift_sq = (self.burnout_cg_m - grain_cg) ** 2
        roll = self.launch_inertia_kg_m2[0] - self.burnout_inertia_kg_m2[0]
        pitch = (
            self.launch_inertia_kg_m2[1]
            - self.burnout_inertia_kg_m2[1]
            - reduced * shift_sq
        )
        outer_sq = (motor_diameter_m / 2.0) ** 2
        core_sq = min(
            max(2.0 * roll / propellant - outer_sq, 0.0), _MAX_CORE_SHARE * outer_sq
        )
        length_sq = max(
            12.0 * pitch / propellant - 3.0 * (outer_sq + core_sq), _MIN_LENGTH_M**2
        )
        grain = PropellantGrain(
            mass=scalar(propellant, "kg"),
            length=scalar(math.sqrt(length_sq), "m"),
            outer_diameter=scalar(motor_diameter_m, "m"),
            core_diameter=scalar(2.0 * math.sqrt(core_sq), "m"),
            cg_location=vector((grain_cg, 0.0, 0.0), "m"),
        )
        casing_cg = vector((self.burnout_cg_m, 0.0, 0.0), "m")
        casing = MassProperties(
            mass=scalar(_CASING_SHARE * burnout, "kg"),
            cg_location=casing_cg,
            inertia=matrix(np.zeros((3, 3)), "kg*m**2"),
        )
        dry = MassProperties(
            mass=scalar((1.0 - _CASING_SHARE) * burnout, "kg"),
            cg_location=casing_cg,
            inertia=matrix(np.diag(self.burnout_inertia_kg_m2), "kg*m**2"),
        )
        return dry, casing, grain
