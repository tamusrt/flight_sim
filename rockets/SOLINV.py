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

# Motor: IgnisSET5.eng, copied from
# srt_13/3_Dynamics/1_Aeroshaping & Flight Profiles/1_IREC/RasAero/Latest

def get_properties() -> RocketProperties:
    """Aerodynamics, motor (.eng file) and dry mass for this rocket."""
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "data/ras_alpha_files/converted_aerodynamics-SOLINV.csv",  # Mach,Alpha,Phi,Cx,Cy,Cz,CMx,CMy,CMz
            reference_area=scalar(0.0182414692, "m**2"),  # OR: pi * diameter**2 / 4
            reference_length=scalar(0.1524, "m"),  # OR: body tube diameter
            reference_point=vector((-1.27, 0.0, 0.0), "m"),  # Converter REFPOINT: 50 in aft of nose
            frame="missile",
        ),
        engine=SolidEngine.from_eng(
            # .eng files carry no CG, so the motor's CG is its geometric center
            "data/motors/IgnisSET5.eng",

            # Replace 3.10 with the actual distance from the rocket nose
            # to the FORWARD end of the installed motor.
            vector((-2.4003, 0.0, 0.0), "m"),
        ),
        # RASAero sustainer is 141.31 lb loaded with its CG 129 in aft of the
        # nose; less the motor (43.68 kg, CG 3.7095 m aft) that leaves:
        dry_mass_properties=MassProperties(
            mass=scalar(20.417, "kg"),
            cg_location=vector((-2.3505, 0.0, 0.0), "m"),  # 92.54 in aft
            # OR: (rotational, longitudinal, longitudinal) moments of inertia
            inertia=matrix(np.diag((5.23, 2602.07, 2602.07)), "lb*ft**2"),  
        ),
    )


def get_rail() -> LaunchRail:
    """Launch rail length, elevation and azimuth."""
    return LaunchRail(
        length=scalar(39.37, "inch"),  # OR: launch rod length
        elevation=scalar(90.0, "deg"),  # OR: 90 - launch rod angle
        azimuth=scalar(0.0, "deg"),
    )


def get_truth_options() -> dict[str, Any]:
    """Wind, atmosphere, gravity and launch site (match OR launch conditions)."""
    return {
        "wind": UniformWind(),  # calm air
        "atmosphere": StandardAtmosphere1976(),  # same ISA as OpenRocket's default
        "gravity": WGS84Gravity(),
        "launch_elevation": scalar(0.0, "m"),  # OR: launch altitude
        "launch_latitude": scalar(28.61, "deg"),  # OR: latitude
    }