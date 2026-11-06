"""EDITH runner: sample, prepare and fly the campaign given on the command line."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence

from tqdm import tqdm  # type: ignore[import-untyped]

from edith import parallel, runner
from edith.config import EdithConfig
from edith.errors import EdithConfigError
from edith.parallel import Status
from edith.sampler import sample
from flight_sim.config.toml_files import display_path

# Commands that take the campaign only so far
COMMANDS = ("prepare",)

_DESCRIPTIONS = {
    None: "Run a Monte Carlo campaign of a sim: prepare a run directory per "
    'trial, then fly them in parallel. "edith prepare FILE" stops after the '
    "directories.",
    "prepare": "Prepare a campaign's run directories, one dispersed sim per "
    "trial, without flying them.",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Read a campaign file, prepare its trials and fly them in parallel.

    A first argument of "prepare" stops after the run directories.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.

    Raises:
        SystemExit: With the usage and code 2 if no file is given, with the
            error if the campaign cannot be read or prepared, or with code 1
            if a trial failed or timed out.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in COMMANDS else None
    if not argv:
        _parser(command).print_help(sys.stderr)
        raise SystemExit(2)
    args = _parser(command).parse_args(argv)

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

    prepared = runner.prepare(config, sample(config), overwrite=args.overwrite)
    try:
        directories = list(
            tqdm(
                prepared,
                total=config.num_sims,
                desc="prepare",
                unit="trial",
                disable=None,
            )
        )
    except FileExistsError as error:
        raise SystemExit(f"{error}; pass --overwrite to write into it") from None
    except EdithConfigError as error:
        raise SystemExit(f"Cannot prepare {args.config}:\n  {error}") from None
    runs = display_path(config.output_dir / runner.RUNS)
    print(f"Prepared {len(directories)} run directories in {runs}")
    if command == "prepare":
        return

    timeout = config.trial_timeout
    results = parallel.run(
        directories,
        args.workers,
        None if timeout is None else float(timeout.m_as("s")),
        config.output_dir / parallel.CAMPAIGN_LOG,
    )
    counts = Counter(result.status for result in results)
    tally = ", ".join(
        f"{counts[status]} {status}" for status in Status if counts[status]
    )
    log = display_path(config.output_dir / parallel.CAMPAIGN_LOG)
    print(f"{len(results)} trials: {tally}; see {log}")
    if counts[Status.OK] != len(results):
        raise SystemExit(1)


def _parser(command: str | None) -> argparse.ArgumentParser:
    """Return the argument parser of a command, or of the whole campaign."""
    parser = argparse.ArgumentParser(
        prog="edith" if command is None else f"edith {command}",
        description=_DESCRIPTIONS[command],
    )
    parser.add_argument("config", help="Campaign input file, such as edith_config.toml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write into the runs folder even if it already exists",
    )
    if command is None:
        parser.add_argument(
            "-n",
            "--workers",
            type=_positive,
            default=parallel.default_workers(),
            help="Worker processes, each flying one trial at a time; "
            "default: the CPUs less one",
        )
    return parser


def _positive(text: str) -> int:
    """Read a command-line integer of at least 1.

    Raises:
        argparse.ArgumentTypeError: If it is not one.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number, got {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {value}")
    return value


if __name__ == "__main__":
    main()
