"""Core FS runner: fly the sim input file given on the command line."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from flight_sim.simulation import Simulation

DEFAULT_SIM = "configs/sims/sol_invictus_nominal.toml"


def main(argv: Sequence[str] | None = None) -> None:
    """Fly a sim input file, print its headline numbers, then write and plot it.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.
    """
    parser = argparse.ArgumentParser(description="Fly a sim input file.")
    parser.add_argument(
        "sim", nargs="?", default=DEFAULT_SIM, help=f"Defaults to {DEFAULT_SIM}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help='Write here instead of [output] directory; "{name}" is the sim name',
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save the plots without showing them"
    )
    args = parser.parse_args(argv)

    simulation = Simulation.from_file(args.sim)
    if args.output_dir is not None:
        simulation.output = replace(simulation.output, directory=args.output_dir)
    flight = simulation.run()

    for key, value in flight.summary().items():
        print(f"{key}: {value:.4g}")

    figures = flight.figures()
    directory = flight.write(figures)
    if directory is not None:
        print(f"Wrote {directory}")
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
