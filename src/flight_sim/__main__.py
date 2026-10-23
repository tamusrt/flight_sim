"""Core FS runner: prepare and fly the sim input file given on the command line."""

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
from flight_sim.config import ConfigError, load_toml, load_with_source
from flight_sim.mission import Mission
from flight_sim.simulation import Flight, Simulation

# Functions listed by --profile
PROFILE_LINES = 20

# Commands that take one step of a run on their own
COMMANDS = ("prepare", "run")

_DESCRIPTIONS = {
    None: "Prepare a sim's run directory, then fly it from there. "
    '"flight_sim prepare SIM" and "flight_sim run DIR" take each step alone.',
    "prepare": "Create a sim's run directory, holding its resolved input and "
    "copies of its data files.",
    "run": "Fly a prepared run directory, writing its results into it.",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare a sim's run directory, fly it from there, and write and plot it.

    A first argument of "prepare" or "run" takes only that step.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in COMMANDS else None
    parser = _parser(command)
    if not argv:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    args = parser.parse_args(argv)

    if command == "run":
        fly(_load(Path(args.directory) / output.RESOLVED_INPUT), args)
        return
    simulation = _load(args.sim)
    if args.output_dir is not None:
        simulation.output = replace(simulation.output, directory=args.output_dir)
    if command is None and simulation.output.directory is None:
        # Nothing to write, so nothing to prepare
        fly(simulation, args)
        return
    try:
        directory = output.prepare(simulation, overwrite=args.overwrite)
    except FileExistsError as error:
        raise SystemExit(f"{error}; pass --overwrite to write into it") from None
    except ValueError as error:
        raise SystemExit(
            f"Cannot prepare {args.sim}: {error}; set [output] directory"
            " or pass --output-dir"
        ) from None
    print(f"Prepared {directory}")
    if command is None:
        fly(_load(directory / output.RESOLVED_INPUT), args)


def _parser(command: str | None) -> argparse.ArgumentParser:
    """Return the argument parser of a command, or of preparing and flying."""
    parser = argparse.ArgumentParser(
        prog="flight_sim" if command is None else f"flight_sim {command}",
        description=_DESCRIPTIONS[command],
    )
    if command == "run":
        parser.add_argument("directory", help='Run directory "flight_sim prepare" made')
    else:
        parser.add_argument(
            "sim", help="Sim input file, such as examples/sol_invictus/sim.toml"
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            help="Run directory to prepare instead of [output] directory; it takes"
            ' the same "{name}" and "{datetime}" placeholders',
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Write into the run directory even if it already exists",
        )
    if command != "prepare":
        parser.add_argument(
            "--no-show",
            action="store_true",
            help="Save the plots without showing them",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Time each function during the run, and save profile.prof",
        )
    return parser


def _load(path: str | Path) -> Simulation | Mission:
    """Build a sim, or a mission if it has segments, exiting if it cannot."""
    start = clock.perf_counter()
    try:
        table = load_toml(path)
        cls: type[Simulation | Mission] = Mission if "segments" in table else Simulation
        built, source = load_with_source(path, cls, table)
    except ConfigError as error:
        raise SystemExit(f"Cannot build {path}:\n  {error}") from None
    built.source = replace(source, load_time_s=clock.perf_counter() - start)
    return built


def fly(simulation: Simulation | Mission, args: argparse.Namespace) -> None:
    """Fly a sim, print its headline numbers, then write and plot it.

    Args:
        simulation (Simulation | Mission): The sim to fly.
        args (argparse.Namespace): Parsed arguments, for --no-show and
            --profile.
    """
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
