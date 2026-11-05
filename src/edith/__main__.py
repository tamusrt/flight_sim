"""EDITH runner: read the Monte Carlo campaign given on the command line."""

import argparse
import sys
from collections.abc import Sequence

from edith.config import EdithConfig
from edith.errors import EdithConfigError
from flight_sim.config.toml_files import display_path


def main(argv: Sequence[str] | None = None) -> None:
    """Read a campaign file from the command line and print what it describes.

    Running the campaign is not built yet, so the command ends after printing
    the campaign's settings and dispersions.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.

    Raises:
        SystemExit: With the usage and code 2 if no file is given, or with the
            error if the file does not describe a valid campaign.
    """
    parser = argparse.ArgumentParser(
        prog="edith", description="Run a Monte Carlo campaign of a sim."
    )
    parser.add_argument("config", help="Campaign input file, such as edith_config.toml")
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    args = parser.parse_args(argv)

    try:
        config = EdithConfig.from_file(args.config)
    except EdithConfigError as error:
        raise SystemExit(f"Cannot read {args.config}:\n  {error}") from None
    print(
        f"{display_path(config.path)}: {config.trials} trials of "
        f"{display_path(config.base_sim)}, {config.sampling} sampling, "
        f"seed {config.master_seed}"
    )
    for dispersion in config.dispersions:
        nominal = f"{dispersion.nominal:g} {dispersion.units}".rstrip()
        print(f"  {dispersion.key}: {nominal}, {dispersion.distribution!r}")
    print("Running the campaign is not built yet.")


if __name__ == "__main__":
    main()
