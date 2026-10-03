"""Build a flyable rocket from an OpenRocket design and a RASAero CSV.

The geometry, masses and launch conditions come from the ``.ork``; the
aerodynamics come from the RASAero CSV the team made for the same geometry; the
motor is the ``.eng`` or ``.rse`` given (its thrust curve and its propellant
and total mass, so a motor change needs no new ``.ork``). Pass the same ``.ork``
that the CSV was made from: the sim cannot tell whether the two still agree.

The saved OpenRocket simulation named ``sim`` supplies the launch rod, wind
and site, since those are not part of the rocket itself.

An ``.ork`` that carries no parachutes gets the original descent: the recovery
Sol Invictus flies (a reefed 120 in main on one separation), with the body
drag taken from this rocket's own size. Replace it once the rocket has a
recovery design of its own.
"""

from __future__ import annotations

import atexit
import math
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from flight_sim.__main__ import RECOVERY, RocketProfile
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import UniformWind
from flight_sim.motor_file import MotorFile, eng_for_sim, load_motor
from flight_sim.ork import (
    Motor,
    OrkRocket,
    load_ork,
    mass_properties,
    saved_motor,
)
from flight_sim.recovery_extension import EJECTION_CHARGE, FLIGHT_COMPUTER
from flight_sim.recovery_systems import ReefedSingleSeparation
from flight_sim.units import scalar
from flight_sim.vehicle.mass_properties import MassPropertiesTable
from flight_sim.vehicle.rocket_properties import TrapezoidFinSet

_LAPSE_K_PER_M = 0.0065
# Drag coefficient of the rocket falling roughly nose-first, as in __main__
_FALLING_BODY_CD = 0.55
_ISA_EXPONENT = 5.2559


def pad_pressure_pa(sea_level_pa: float, pad_k: float, elevation_m: float) -> float:
    """Pressure at the pad, from the sea-level pressure and the pad temperature.

    OpenRocket's "extended ISA" gives a base temperature and a sea-level
    pressure; the standard lapse rate carries them up to the pad.
    """
    sea_level_k = pad_k + _LAPSE_K_PER_M * elevation_m
    return float(
        sea_level_pa
        * (1.0 - _LAPSE_K_PER_M * elevation_m / sea_level_k) ** _ISA_EXPONENT
    )


def table_from(ork: OrkRocket, motor: Motor, roll_ratio: float) -> MassPropertiesTable:
    """Mass, CG and inertia at ignition and burnout, from the component masses.

    Args:
        ork (OrkRocket): The design.
        motor (Motor): The motor's mass and station.
        roll_ratio (float): Roll inertia over mass, in m**2.
    """
    mass0, cg0, iyy0 = mass_properties(ork, motor, 0.0)
    mass1, cg1, iyy1 = mass_properties(ork, motor, 1.0)
    return MassPropertiesTable(
        launch_mass_kg=mass0,
        launch_cg_m=-cg0,
        launch_inertia_kg_m2=(roll_ratio * mass0, iyy0, iyy0),
        burnout_mass_kg=mass1,
        burnout_cg_m=-cg1,
        burnout_inertia_kg_m2=(roll_ratio * mass1, iyy1, iyy1),
    )


def motor_with_file(saved: Motor, motor_file: MotorFile) -> Motor:
    """The motor a thrust-curve file describes, where OpenRocket had the saved one.

    The file's propellant and total mass win over the ones inferred from the
    saved simulation, so a newer motor file is flown with its own masses.
    """
    return replace(
        saved,
        mass_kg=motor_file.total_kg or saved.mass_kg,
        propellant_kg=motor_file.propellant_kg or saved.propellant_kg,
    )


def _sim_eng(motor_path: str | Path) -> str:
    """An ``.eng`` for the sim: the file itself, or a copy made from an ``.rse``."""
    if Path(motor_path).suffix.lower() == ".eng":
        return str(motor_path)
    scratch = Path(tempfile.mkdtemp(prefix="flight_sim_motor_"))
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    return str(eng_for_sim(motor_path, scratch))


def original_descent(reference_area_m2: float) -> ReefedSingleSeparation:
    """The Sol Invictus descent, with the falling body's drag for this rocket.

    Args:
        reference_area_m2 (float): The rocket's body cross-section.

    Returns:
        ReefedSingleSeparation: One reefed canopy out 3 s after apogee, cut
        open at 2000 ft, on a single separation.
    """
    recovery = replace(RECOVERY, body_drag_area_m2=_FALLING_BODY_CD * reference_area_m2)
    return ReefedSingleSeparation(recovery, FLIGHT_COMPUTER, EJECTION_CHARGE)


def profile_from_ork(  # pylint: disable=too-many-locals
    ork_path: str | Path,
    aero_csv: str,
    motor_eng: str,
    *,
    sim: str | None = None,
    name: str | None = None,
    fin_misalignment_deg: float = 0.0,
    motor: Motor | None = None,
) -> RocketProfile:
    """Make a rocket profile from an ``.ork`` and a RASAero CSV.

    Args:
        ork_path (str | Path): The OpenRocket file.
        aero_csv (str): Aero CSV in the sim's format, made for this geometry.
        motor_eng (str): Motor file, RASP ``.eng`` or RockSim ``.rse``.
        sim (str | None): Saved simulation whose launch conditions to use;
            the first one in the file when None.
        name (str | None): Name for the viewer; the design's own when None.
        fin_misalignment_deg (float): Assumed fin twist that spins the rocket.
        motor (Motor | None): The motor's masses; from the motor file when
            None (the saved simulation's where the file gives none).

    Returns:
        RocketProfile: Ready to fly, down to the ground under the original descent.
    """
    ork = load_ork(ork_path)
    sim_name = sim if sim is not None else next(iter(ork.sims))
    saved = ork.sims[sim_name]
    cond = saved.conditions
    motor = motor or motor_with_file(saved_motor(ork, sim_name), load_motor(motor_eng))
    roll_ratio = float(saved.series["Rotational moment of inertia"][0]) / float(
        saved.series["Mass"][0]
    )
    fins = ork.fins
    pad_k = cond.get("basetemperature", 288.15)
    return RocketProfile(
        name=name or ork.name,
        aero_file=aero_csv,
        motor_file=_sim_eng(motor_eng),
        propellant_mass_kg=motor.propellant_kg,
        mass_properties=table_from(ork, motor, roll_ratio),
        reference_area_m2=ork.reference_area_m2,
        reference_length_m=ork.reference_diameter_m,
        fins=TrapezoidFinSet(
            fin_count=fins.count,
            root_chord_m=fins.root_chord_m,
            tip_chord_m=fins.tip_chord_m,
            span_m=fins.span_m,
            sweep_length_m=fins.sweep_m,
            body_radius_m=0.5 * ork.reference_diameter_m,
            misalignment_rad=math.radians(fin_misalignment_deg),
        ),
        rail=LaunchRail(
            length=scalar(cond["launchrodlength"], "m"),
            elevation=scalar(90.0 - cond["launchrodangle"], "deg"),
            azimuth=scalar(cond["launchroddirection"], "deg"),
        ),
        scheme=original_descent(ork.reference_area_m2),
        wind=UniformWind(
            speed=scalar(cond["windaverage"], "m/s"),
            from_azimuth=scalar(math.degrees(cond["winddirection"]), "deg"),
        ),
        pad_temperature_k=pad_k,
        pad_elevation_m=cond["launchaltitude"],
        pad_pressure_pa=pad_pressure_pa(
            cond.get("basepressure", 101325.0), pad_k, cond["launchaltitude"]
        ),
    )
