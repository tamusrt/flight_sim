"""Core FS runner: fly the sim input file given on the command line."""

import argparse
import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from flight_sim.config import load
from flight_sim.simulation import Simulation

DEFAULT_SIM = "configs/sims/sol_invictus_nominal.toml"


def main(argv: Sequence[str] | None = None) -> None:
    """Fly a sim input file, then print its apogee and plot the flight.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.
    """
    parser = argparse.ArgumentParser(description="Fly a sim input file.")
    parser.add_argument(
        "sim", nargs="?", default=DEFAULT_SIM, help=f"Defaults to {DEFAULT_SIM}"
    )
    simulation = load(parser.parse_args(argv).sim, Simulation)
    record = simulation.run()

    df = record.telemetry
    step_lengths = record.step_lengths
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
    plt.title(f"Flight Profile - {simulation.name}")
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
