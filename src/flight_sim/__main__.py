"""Core FS runner"""

import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.integration import adaptive_step, derivative_computation, locate_event
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


def _telemetry_row(time: float, state: RocketState) -> dict[str, float]:
    """Return the telemetry recorded for one sample of the flight."""
    return {
        "Time (s)": time,
        "Altitude (m)": _altitude(state),
        "Vertical Velocity (m/s)": _vertical_velocity(state),
        "Roll Rate (rad/s)": float(state.angular_velocity.m_as("rad/s")[2]),
        "Mass (kg)": float(state.current_mass.m_as("kg")),
    }


def _altitude(state: RocketState) -> float:
    """Return the altitude in metres."""
    return float(state.position.m_as("m")[2])


def _vertical_velocity(state: RocketState) -> float:
    """Return the vertical velocity in m/s."""
    return float(state.velocity.m_as("m/s")[2])


def _vertical_acceleration(
    properties: RocketProperties, time: float, state: RocketState
) -> float:
    """Return the vertical acceleration in m/s**2, which is zero at max velocity."""
    derivative = derivative_computation(time, state, properties)
    return float(derivative.acceleration.m_as("m/s**2")[2])


def _apogee_event(_time: float, state: RocketState) -> float:
    """Return the vertical velocity, which falls through zero at apogee."""
    return _vertical_velocity(state)


def _impact_event(_time: float, state: RocketState) -> float:
    """Return the altitude, which falls through zero at impact."""
    return _altitude(state)


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

    # Length of the first step to try; the integrator adapts it from there
    dt = scalar(0.01, "s")
    current_time = 0.0
    telemetry = [_telemetry_row(current_time, state)]

    # Peak vertical velocity, then apogee, in the order the flight meets them
    flight_events = (partial(_vertical_acceleration, properties), _apogee_event)

    dt_arr = np.array([])

    while True:
        next_state, dt_taken, dt = adaptive_step(current_time, state, properties, dt)
        dt_arr = np.append(dt_arr, dt_taken)
        # End this step exactly at the first flight event it crosses, so the
        # telemetry holds the integrated state at that event
        for event in flight_events:
            end_time = current_time + float(dt_taken.m_as("s"))
            if event(current_time, state) > 0.0 >= event(end_time, next_state):
                next_state, dt_taken = locate_event(
                    current_time, state, next_state, properties, dt_taken, event
                )
                break

        step_length = float(dt_taken.m_as("s"))
        landed = current_time + step_length > 0.1 and _altitude(next_state) <= 0.0
        if landed:
            # End this step exactly at impact
            next_state, dt_taken = locate_event(
                current_time, state, next_state, properties, dt_taken, _impact_event
            )
            step_length = float(dt_taken.m_as("s"))

        state = next_state
        current_time += step_length
        telemetry.append(_telemetry_row(current_time, state))

        if landed:
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

    plt.figure()
    plt.plot(dt_arr, linewidth=2)
    plt.xlabel("Step (-)")
    plt.ylabel("dt taken (delta sec)")

    plt.tight_layout()
    if "PYTEST_CURRENT_TEST" not in os.environ:
        plt.show()


if __name__ == "__main__":
    main()
