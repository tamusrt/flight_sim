"""Core FS runner"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.events import APOGEE, IMPACT, peak_vertical_velocity
from flight_sim.flight_event import FlightEvent
from flight_sim.integration import (
    IntegrationConfiguration,
    TruthConfiguration,
    adaptive_step,
)
from flight_sim.units import matrix, scalar, vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.engine import PropellantGrain, solid_engine_from_csv
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


def get_default_state() -> RocketState:
    """Generate the standard initial state for Sol Invictus, at rest off the rail."""
    return RocketState(
        position=vector((0.0, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
        angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
        orientation=Quaternion(q_x=0.0, q_y=0.0, q_z=0.0, q_w=1.0),
    )


def get_default_properties() -> RocketProperties:
    """Build the standard aerodynamic, motor and mass properties for Sol Invictus.

    The mass properties are placeholders: 25 kg with its CG about 1.5 m aft of
    the nose tip at ignition, burning down to 20 kg.
    """
    grain = PropellantGrain(
        mass=scalar(5.0, "kg"),
        length=scalar(0.75, "m"),
        outer_diameter=scalar(0.0762, "m"),
        core_diameter=scalar(0.03, "m"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
    )
    # Thin tube of radius 0.045 m and length 0.9 m
    casing = MassProperties(
        mass=scalar(3.0, "kg"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
        inertia=matrix(np.diag((0.0061, 0.21, 0.21)), "kg*m**2"),
    )
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "data/aero/estimated_aero.csv",
            reference_area=scalar(0.0182414692, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=vector((0.0, 0.0, 0.0), "m"),
            frame="missile",
        ),
        engine=solid_engine_from_csv(
            "tests/test_data/standard_motor.csv", grain, casing
        ),
        dry_mass_properties=MassProperties(
            mass=scalar(17.0, "kg"),
            cg_location=vector((-1.0, 0.0, 0.0), "m"),
            inertia=matrix(np.diag((2.4, 135.0, 135.0)), "kg*m**2"),
        ),
    )


def main() -> None:  # pylint: disable=too-many-statements
    """Main function to run entire flight"""
    properties = get_default_properties()
    rail = LaunchRail(
        length=scalar(17.0, "ft"),
        elevation=scalar(85.0, "deg"),
        azimuth=scalar(0.0, "deg"),
    )
    state = rail.mount(get_default_state())
    config = IntegrationConfiguration(truth=TruthConfiguration(launch_rail=rail))
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)

    dt = scalar(0.01, "s")  # First step length to try
    current_time = 0.0
    telemetry = []
    step_lengths = []
    hit: FlightEvent | None = None

    while True:
        pos = state.position.m_as("m")
        vel = state.velocity.m_as("m/s")
        ang_rate = state.angular_velocity.m_as("rad/s")

        telemetry.append(
            {
                "Time (s)": current_time,
                "Inertial Position (m)": pos,
                "Inertial Velocity (m/s)": vel,
                "Angular Rate (rad/s)": ang_rate,
                "Mass (kg)": properties.mass_properties(current_time).mass,
            }
        )

        if hit is not None:
            print(f"{hit.name} at {current_time:.2f} seconds")
        if hit is IMPACT:
            break

        state, dt_taken, dt, hit = adaptive_step(
            current_time, state, properties, config, dt, events=events
        )
        current_time += float(dt_taken.m_as("s"))
        step_lengths.append(float(dt_taken.m_as("s")))

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
