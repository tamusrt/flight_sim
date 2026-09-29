"""Core FS runner"""

import os

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.events import APOGEE, IMPACT, FlightEvent, peak_vertical_velocity
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import scalar, vector
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


def get_default_state() -> RocketState:
    """Generate the standard launchpad initial state for Sol Invictus."""
    return RocketState(
        position=vector((0.0, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
        current_mass=scalar(25.0, "kg"),
        inertia=vector((150.0, 150.0, 2.5), "kg*m**2"),
        cg_location=vector((0.0, 0.0, -1.5), "m"),
        angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
        orientation=Quaternion(q_x=0.0, q_y=0.0, q_z=0.0, q_w=1.0),
    )


def main() -> None:
    """Main function to run entire flight"""
    properties = RocketProperties(
        aero_file_path="tests/test_data/standard_aero.csv",
        motor_file_path="tests/test_data/standard_motor.csv",
        propellant_mass=5.0,
        reference_area=scalar(0.0182414692, "m**2"),
        reference_diameter=scalar(0.1524, "m"),
    )
    state = get_default_state()
    config = IntegrationConfiguration()
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)

    dt = scalar(0.01, "s")  # First step length to try
    current_time = 0.0
    telemetry = []
    step_lengths = []
    hit: FlightEvent | None = None

    while True:
        pos_z = float(state.position.m_as("m")[2])
        vel_z = float(state.velocity.m_as("m/s")[2])
        roll_rate = float(state.angular_velocity.m_as("rad/s")[2])

        telemetry.append(
            {
                "Time (s)": current_time,
                "Altitude (m)": pos_z,
                "Vertical Velocity (m/s)": vel_z,
                "Roll Rate (rad/s)": roll_rate,
                "Mass (kg)": float(state.current_mass.m_as("kg")),
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
    apogee = df["Altitude (m)"].max()
    max_vel = df["Vertical Velocity (m/s)"].max()
    print(f"Apogee: {apogee:.1f} m")
    print(f"Max Vertical Velocity: {max_vel:.1f} m/s")

    plt.figure(figsize=(10, 12))

    # Plot 1: Altitude
    plt.subplot(3, 1, 1)
    plt.plot(df["Time (s)"], df["Altitude (m)"], color="blue", linewidth=2)
    plt.title("Flight Profile - Airmail")
    plt.ylabel("Altitude (m)")
    plt.grid(True)

    # Plot 2: Vertical Velocity
    plt.subplot(3, 1, 2)
    plt.plot(df["Time (s)"], df["Vertical Velocity (m/s)"], color="red", linewidth=2)
    plt.ylabel("Vertical Velocity (m/s)")
    plt.grid(True)

    # Plot 3: Roll Rate
    plt.subplot(3, 1, 3)
    plt.plot(df["Time (s)"], df["Roll Rate (rad/s)"], color="green", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Roll Rate (rad/s)")
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
