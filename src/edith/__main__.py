"""EDITH runner: run the Monte Carlo campaign given on the command line."""

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Read a campaign input file from the command line.

    Running the campaign it describes is not built yet, so a named file ends
    the program with a message saying so.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.
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

    raise SystemExit(f"Cannot run {args.config}: campaigns are not built yet")


if __name__ == "__main__":
    main()
