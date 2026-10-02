"""Core FS runner: flies Sol Invictus from the pad to the ground and plots it.

Vehicle, launch site and recovery values come from the team's OpenRocket
model (dynamics repo, aero_modeling/SOL_INVICTUS/OpenRocket/SOL_4_30.ork,
simulation "best") unless a comment says otherwise. That simulation
predicts a 6445 m apogee at 39.0 s and 30.3 m/s off the rail, for
comparison. The recovery system follows the recovery team's IREC info doc
instead of the drogue and main in that file.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.descent import (
    DescentResult,
    RecoverySystem,
    ReefedParachute,
    simulate_descent,
)
from flight_sim.environment.atmosphere import LaunchSiteAtmosphere
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import UniformWind
from flight_sim.events import (
    APOGEE,
    IMPACT,
    FlightEvent,
    peak_vertical_velocity,
    rail_exit,
)
from flight_sim.integration import (
    IntegrationConfiguration,
    TruthConfiguration,
    adaptive_step,
)
from flight_sim.units import scalar, vector, zero_vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.mass_properties import MassPropertiesTable
from flight_sim.vehicle.rocket_properties import RocketProperties, TrapezoidFinSet
from flight_sim.vehicle.rocket_state import RocketState

# Mass properties at ignition and burnout. The propellant mass is the one in
# the IgnisSET5.eng header, which matches OpenRocket's mass drop.
PROPELLANT_MASS_KG = 22.135
MASS_PROPERTIES = MassPropertiesTable(
    launch_mass_kg=65.19,
    launch_cg_m=-3.298,  # Behind the nose tip
    launch_inertia_kg_m2=(0.237, 97.06, 97.06),
    burnout_mass_kg=65.19 - PROPELLANT_MASS_KG,
    burnout_cg_m=-3.109,
    burnout_inertia_kg_m2=(0.174, 82.74, 82.74),
)

# Launch site: pad elevation, temperature and pressure, and latitude
PAD_ELEVATION_M = 890.016
PAD_TEMPERATURE_K = 303.15
PAD_PRESSURE_PA = 91432.755
LAUNCH_LATITUDE_DEG = 31.0311

# Steady wind of 3.57632 m/s blowing toward +Y, which is east, so it comes
# from the west (azimuth 270 degrees)
WIND = UniformWind(speed=scalar(3.57632, "m/s"), from_azimuth=scalar(270.0, "deg"))

# 10 m rail tilted 4 degrees from vertical into the wind, so leaning west
LAUNCH_RAIL = LaunchRail(
    length=scalar(10.0, "m"),
    elevation=scalar(86.0, "deg"),
    azimuth=scalar(270.0, "deg"),
)

# Four trapezoidal fins. The 0.02 degree misalignment is an assumed small
# manufacturing defect that slowly rolls the rocket; set it to 0 for a
# perfectly built fin can.
FIN_MISALIGNMENT_DEG = 0.02
FINS = TrapezoidFinSet(
    fin_count=4,
    root_chord_m=0.381,
    tip_chord_m=0.0762,
    span_m=0.1905,
    sweep_length_m=0.3302,
    body_radius_m=0.0762,
    misalignment_rad=math.radians(FIN_MISALIGNMENT_DEG),
)

REFERENCE_AREA_M2 = 0.0182414692  # 6 in body tube

# Recovery: single separation, dual deploy with one reefed main, from the
# recovery team's IREC info doc. A Fruity Chutes 120 in canopy (Cd 2.2 on
# its area less the 21.12 in spill hole) comes out reefed to a 3.98 ft mouth
# (reefed Cd 0.7, the team's estimate pending testing), and the reef line is
# cut at 2000 ft above the pad. It comes out 3 s after apogee by assumption.
# The body drag area assumes the rocket falls roughly nose-first with the
# axial drag coefficient OpenRocket gives near apogee (0.55).
FT_TO_M = 0.3048
RECOVERY = RecoverySystem(
    parachutes=(
        ReefedParachute(
            "main",
            diameter_m=120.0 * 0.0254,
            drag_coefficient=2.2,
            reefed_opening_diameter_m=3.9796 * FT_TO_M,
            reefed_drag_coefficient=0.7,
            disreef_altitude_m=2000.0 * FT_TO_M,
            deploy_delay_s=3.0,
            spill_hole_diameter_m=21.12 * 0.0254,
        ),
    ),
    body_drag_area_m2=0.55 * REFERENCE_AREA_M2,
)


def get_default_state() -> RocketState:
    """Generate the standard launchpad initial state for Sol Invictus."""
    return RocketState(
        position=vector((0.0, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
        current_mass=scalar(MASS_PROPERTIES.launch_mass_kg, "kg"),
        inertia=vector(MASS_PROPERTIES.launch_inertia_kg_m2, "kg*m**2"),
        cg_location=vector((MASS_PROPERTIES.launch_cg_m, 0.0, 0.0), "m"),
        angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
        orientation=Quaternion(q_x=0.0, q_y=0.0, q_z=0.0, q_w=1.0),
        on_rail=True,
    )


def get_launch_state() -> RocketState:
    """Return the launchpad state with the rocket lying along the launch rail."""
    state = get_default_state()
    state.orientation = LAUNCH_RAIL.orientation()
    return state


def get_default_properties() -> RocketProperties:
    """Build the Sol Invictus properties, including the fins for roll."""
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "data/ras_alpha_files/invictus_aero.csv",
            reference_area=scalar(REFERENCE_AREA_M2, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=zero_vector("m"),
        ),
        motor_file_path="data/motors/IgnisSET5.eng",
        propellant_mass=PROPELLANT_MASS_KG,
        fins=FINS,
    )


def get_default_config() -> IntegrationConfiguration:
    """Build the integration settings for the launch site, wind and rail."""
    return IntegrationConfiguration(
        truth=TruthConfiguration(
            atmosphere=LaunchSiteAtmosphere(
                pad_elevation_m=PAD_ELEVATION_M,
                pad_temperature_k=PAD_TEMPERATURE_K,
                pad_pressure_pa=PAD_PRESSURE_PA,
            ),
            wind=WIND,
            launch_latitude=scalar(LAUNCH_LATITUDE_DEG, "deg"),
            launch_elevation=scalar(PAD_ELEVATION_M, "m"),
            launch_rail=LAUNCH_RAIL,
        )
    )


def telemetry_row(time_s: float, state: RocketState) -> dict[str, object]:
    """Return the plotted values of one state."""
    return {
        "Time (s)": time_s,
        "Inertial Position (m)": state.position.m_as("m"),
        "Inertial Velocity (m/s)": state.velocity.m_as("m/s"),
        "Angular Rate (rad/s)": state.angular_velocity.m_as("rad/s"),
        "Mass (kg)": float(state.current_mass.m_as("kg")),
    }


def apply_mass_properties(state: RocketState) -> None:
    """Set the state's CG and inertia for the propellant left."""
    MASS_PROPERTIES.apply(state)


def descend(
    apogee_time_s: float, state: RocketState, config: IntegrationConfiguration
) -> DescentResult:
    """From apogee, fall under the parachute as a point mass.

    Args:
        apogee_time_s (float): Time since ignition at apogee, in seconds.
        state (RocketState): State at apogee.
        config (IntegrationConfiguration): Truth models and settings.

    Returns:
        DescentResult: The descent samples and parachute events.
    """
    descent = simulate_descent(apogee_time_s, state, config, RECOVERY)
    for deployment in descent.deployments:
        print(f"{deployment.name} at {deployment.time_s:.2f} seconds")
    if descent.landed:
        print(f"impact at {descent.times_s[-1]:.2f} seconds")
    return descent


def main() -> None:  # pylint: disable=too-many-statements
    """Main function to run entire flight"""
    properties = get_default_properties()
    state = get_launch_state()
    config = get_default_config()
    events = (
        rail_exit(LAUNCH_RAIL),
        peak_vertical_velocity(properties, config),
        APOGEE,
        IMPACT,
    )

    dt = scalar(0.01, "s")  # First step length to try
    current_time = 0.0
    telemetry = []
    step_lengths = []
    hit: FlightEvent | None = None

    while True:
        telemetry.append(telemetry_row(current_time, state))

        if hit is not None:
            print(f"{hit.name} at {current_time:.2f} seconds")
        if hit is APOGEE or hit is IMPACT:
            break

        state, dt_taken, dt, hit = adaptive_step(
            current_time, state, properties, config, dt, events=events
        )
        current_time += float(dt_taken.m_as("s"))
        step_lengths.append(float(dt_taken.m_as("s")))
        apply_mass_properties(state)

    if hit is APOGEE:
        descent = descend(current_time, state, config)
        telemetry.extend(map(telemetry_row, descent.times_s, descent.states))

    df = pd.DataFrame(telemetry)
    time = df["Time (s)"]
    position = np.vstack(df["Inertial Position (m)"])
    velocity = np.vstack(df["Inertial Velocity (m/s)"])
    angular_rate = np.vstack(df["Angular Rate (rad/s)"])

    print(f"Apogee: {position[:, 0].max():.1f} m")
    print(f"Max Vertical Velocity: {velocity[:, 0].max():.1f} m/s")

    world_axes = ["X (up)", "Y (east)", "Z (north)"]
    plt.figure(figsize=(10, 12))

    # Plot 1: Position
    plt.subplot(3, 1, 1)
    plt.plot(time, position, linewidth=2)
    plt.title("Flight Profile - Airmail")
    plt.ylabel("Position (m)")
    plt.legend(world_axes)
    plt.grid(True)

    # Plot 2: Velocity
    plt.subplot(3, 1, 2)
    plt.plot(time, velocity, linewidth=2)
    plt.ylabel("Velocity (m/s)")
    plt.legend(world_axes)
    plt.grid(True)

    # Plot 3: Body angular rate
    plt.subplot(3, 1, 3)
    plt.plot(time, angular_rate, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Rate (rad/s)")
    plt.legend(["Body X (roll)", "Body Y", "Body Z"])
    plt.grid(True)

    plt.tight_layout()

    plt.figure()
    plt.plot(step_lengths, linewidth=2)
    plt.xlabel("Step (-)")
    plt.ylabel("dt taken (s)")

    if "PYTEST_CURRENT_TEST" not in os.environ:
        plt.show()


if __name__ == "__main__":
    main()
