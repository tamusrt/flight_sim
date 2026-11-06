"""EDITH runner: read the Monte Carlo campaign given on the command line."""

import argparse
import sys
from collections.abc import Sequence

from edith.config import EdithConfig
from edith.errors import EdithConfigError
from edith.runner import run_campaign
from flight_sim.config.toml_files import display_path


def main(argv: Sequence[str] | None = None) -> None:
    """Read a campaign file from the command line, print it, and run it.

    The command prints the campaign's settings and dispersions, then flies its
    trials in batches until the stopping rule is met or ``num_sims`` trials
    have run, and prints the estimate. ``--dry-run`` stops after printing.

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
    parser.add_argument(
        "--dry-run", action="store_true", help="Read and print the campaign only"
    )
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
        f"{display_path(config.path)}: {config.num_sims} trials of "
        f"{display_path(config.base_sim)}, {config.sampling} sampling, "
        f"seed {config.master_seed}"
    )
    for dispersion in config.dispersions:
        print(f"  {dispersion!r}")
    if args.dry_run:
        print("Dry run: no trials were flown.")
        return
    result = run_campaign(config)
    if result.stopped_by == "num_sims" and config.stopping.rule != "fixed":
        print(f"Warning: stopped at num_sims before the {config.stopping.rule} target")
    print(result.summary())
    print(f"Trials: {display_path(result.trials_path)}")


if __name__ == "__main__":
    main()
