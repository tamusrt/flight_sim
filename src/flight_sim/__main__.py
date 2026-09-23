"""Core FS runner"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from flight_sim.environment.atmosphere import get_atmosphere
from flight_sim.integration import step
from flight_sim.units import scalar, vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import Quaternion, RocketState


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
    )
    state = get_default_state()

    dt = 0.01
    current_time = 0.0
    telemetry = []

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

        atmosphere = get_atmosphere(state.position[2])
        state = step(current_time, state, atmosphere, properties, scalar(dt, "s"))
        current_time += dt

        if current_time > 0.1 and pos_z <= 0.0:
            print(f"Rocket has experienced impact at {current_time:.2f} seconds.")
            break

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
    if "PYTEST_CURRENT_TEST" not in os.environ:
        plt.show()


if __name__ == "__main__":
    main()
