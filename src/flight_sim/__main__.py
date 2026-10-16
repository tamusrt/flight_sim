"""Core FS runner: fly the sim input file given on the command line."""

import argparse
import cProfile
import pstats
import sys
import time as clock
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from flight_sim import output
from flight_sim.config import ConfigError
from flight_sim.simulation import Flight, Simulation

# Functions listed by --profile
PROFILE_LINES = 20


def main(argv: Sequence[str] | None = None) -> None:
    """Fly a sim input file, print its headline numbers, then write and plot it.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.
    """
    parser = argparse.ArgumentParser(
        prog="flight_sim", description="Fly a sim input file."
    )
    parser.add_argument(
        "sim", help="Sim input file, such as examples/sol_invictus/sim.toml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help='Write here instead of [output] directory; "{name}" is the sim name',
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save the plots without showing them"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Time each function during the run, and save profile.prof",
    )
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    args = parser.parse_args(argv)

    try:
        simulation = Simulation.from_file(args.sim)
    except ConfigError as error:
        raise SystemExit(f"Cannot build {args.sim}:\n  {error}") from None
    if args.output_dir is not None:
        simulation.output = replace(simulation.output, directory=args.output_dir)
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    flight = simulation.run()
    if profiler is not None:
        profiler.disable()

    print_headlines(flight)
    start = clock.perf_counter()
    figures = flight.figures()
    plots_s = clock.perf_counter() - start
    start = clock.perf_counter()
    directory = output.write(flight, figures)
    write_s = clock.perf_counter() - start
    print_diagnostics(flight, plots_s, write_s)
    if directory is not None:
        print(f"Wrote {directory}")
    if profiler is not None:
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(r"src[\\/]flight_sim", PROFILE_LINES)
        if directory is not None:
            stats.dump_stats(directory / "profile.prof")
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


def print_headlines(flight: Flight) -> None:
    """Print when each event happened and the flight's summary values.

    Args:
        flight (Flight): The flight that was run.
    """
    times = flight.time.m_as("s")
    for name, row in flight.events.items():
        print(f"{name} at {times[row]:.2f} seconds")
    for key, value in flight.summary().items():
        print(f"{key}: {value:.4g}")


def print_diagnostics(flight: Flight, plots_s: float, write_s: float) -> None:
    """Print the integrator's work and where the wall time went.

    Args:
        flight (Flight): The flight that was run.
        plots_s (float): Wall time drawing the figures took, in seconds.
        write_s (float): Wall time writing the output took, in seconds.
    """
    diagnostics = flight.diagnostics()
    load_s = diagnostics.get("load_time_s", 0.0)
    run_s = diagnostics["run_time_s"]
    print(
        f"steps: {diagnostics['steps']} accepted, "
        f"{diagnostics['rejected_steps']} rejected, "
        f"{diagnostics['rkf45_evaluations']} RKF45 evaluations, "
        f"{diagnostics['event_iterations']} event iterations"
    )
    print(
        f"step length: {diagnostics['min_step_s']:.3g} to "
        f"{diagnostics['max_step_s']:.3g} s"
    )
    print(
        f"wall time: load {load_s:.2f} s, run {run_s:.2f} s "
        f"({diagnostics['real_time_factor']:.3g}x real time), "
        f"plots {plots_s:.2f} s, write {write_s:.2f} s, "
        f"total {load_s + run_s + plots_s + write_s:.2f} s"
    )


if __name__ == "__main__":
    main()
