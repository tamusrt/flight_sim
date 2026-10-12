"""Settings for one rocket. Copy this file to rockets/<name>.py and edit.

Run it with:  uv run python -m flight_sim <name>

All three functions are optional: leave one out and main's built-in value is used.
Body-frame CG values are negative when aft of the nose tip (-2.5 = 2.5 m aft).
"""

from typing import Any

import numpy as np

from flight_sim.environment.atmosphere import StandardAtmosphere1976
from flight_sim.environment.gravity import WGS84Gravity
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import UniformWind
from flight_sim.units import matrix, scalar, vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties


def get_properties() -> RocketProperties:
    """Aerodynamics, motor (.rse file) and dry mass for this rocket."""
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "data/aero/estimated_aero.csv",  # Mach,Alpha,Phi,Cx,Cy,Cz,CMx,CMy,CMz
            reference_area=scalar(0.0182414692, "m**2"),  # OR: pi * diameter**2 / 4
            reference_length=scalar(0.1524, "m"),  # OR: body tube diameter
            reference_point=vector((0.0, 0.0, 0.0), "m"),
            frame="missile",
        ),
        engine=SolidEngine.from_rse("data/motors/Loki_M3400-LB.rse"),
        dry_mass_properties=MassProperties(
            mass=scalar(17.0, "kg"),  # OR: mass with no motor
            cg_location=vector((-1.0, 0.0, 0.0), "m"),  # OR: CG, made negative
            # OR: (rotational, longitudinal, longitudinal) moments of inertia
            inertia=matrix(np.diag((2.4, 135.0, 135.0)), "kg*m**2"),
        ),
    )


def get_rail() -> LaunchRail:
    """Launch rail length, elevation and azimuth."""
    return LaunchRail(
        length=scalar(17.0, "ft"),  # OR: launch rod length
        elevation=scalar(85.0, "deg"),  # OR: 90 - launch rod angle
        azimuth=scalar(0.0, "deg"),
    )


def get_truth_options() -> dict[str, Any]:
    """Wind, atmosphere, gravity and launch site (match OR launch conditions)."""
    return {
        "wind": UniformWind(),  # calm air
        "atmosphere": StandardAtmosphere1976(),  # same ISA as OpenRocket's default
        "gravity": WGS84Gravity(),
        "launch_elevation": scalar(0.0, "m"),  # OR: launch altitude
        # "launch_latitude": scalar(35.0, "deg"),  # OR: latitude
    }