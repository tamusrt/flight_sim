"""EDITH runner: prepare and fly the campaign given on the command line."""

import argparse
import sys
import time as clock
from collections import Counter
from collections.abc import Sequence
from statistics import mean

from rich.progress import Progress, TaskID

from edith import parallel, runner
from edith.config import EdithConfig
from edith.dispersion import BaseDispersion, Distribution, Normal, StandardDispersion
from edith.errors import EdithConfigError
from edith.parallel import Status, progress_bars
from edith.runner import CampaignResult, Trial
from edith.statistics.estimators import BatchStats
from edith.zarr_output import STORE, initialize
from flight_sim.config.toml_files import display_path

# Commands that take the campaign only so far
COMMANDS = ("prepare",)

_DESCRIPTIONS = {
    None: "Run a Monte Carlo campaign of a sim: prepare a run directory per "
    "trial, then fly them in parallel, batch by batch, until the stopping rule "
    'is met. "edith prepare FILE" stops after the directories.',
    "prepare": "Prepare a campaign's run directories, one dispersed sim per "
    "trial, without flying them.",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Read a campaign file, prepare its trials and fly them in parallel.

    A first argument of "prepare" stops after the run directories. Each step
    prints what it did; the end prints the apogee estimate and the wall time
    of each step, as flight_sim prints a run's.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments.
            Defaults to those the program was run with.

    Raises:
        SystemExit: With the usage and code 2 if no file is given, with the
            error if the campaign cannot be read, prepared or flown, or with
            code 1 if a trial failed or timed out.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in COMMANDS else None
    if not argv:
        _parser(command).print_help(sys.stderr)
        raise SystemExit(2)
    args = _parser(command).parse_args(argv)

    start = clock.perf_counter()
    config = _read(args.config)
    phases = {"read": _since(start)}
    if command == "prepare":
        _prepare(config, args.config, args.overwrite)
        phases["prepare"] = _since(start) - phases["read"]
        print(_wall_time(**phases, total=_since(start)))
        return

    # Started now, the workers import flight_sim while the trials are prepared
    count = min(args.workers, config.num_sims)
    with parallel.Workers(
        count, config.output_dir / parallel.WORKER_LOGS, config.batch_size
    ) as workers:
        trials = _prepare(config, args.config, args.overwrite, workers)
        phases["prepare"] = _since(start) - phases["read"]
        initialize(config.output_dir / STORE, config.batch_size)
        result = _fly(config, workers, trials)
    phases["fly"] = _since(start) - sum(phases.values())
    if result.stopped_by == "num_sims" and config.stopping.rule != "fixed":
        print(f"Warning: stopped at num_sims before the {config.stopping.rule} target")
    print(result.summary())
    print(f"Trials: {display_path(result.trials_path)}")
    print(_wall_time(**phases, total=_since(start)))
    seconds = [flown.seconds for flown in result.results]
    print(
        f"trials: {mean(seconds):.1f} s each on average, {min(seconds):.1f} s to "
        f"{max(seconds):.1f} s; {len(seconds) / phases['fly']:.2f} trials/s on "
        f"{count} worker{'s' if count != 1 else ''}"
    )
    if any(flown.status is not Status.OK for flown in result.results):
        raise SystemExit(1)


def _read(path: str) -> EdithConfig:
    """Read a campaign file and print its settings and dispersions.

    Raises:
        SystemExit: With the error, if the file does not describe a campaign.
    """
    try:
        config = EdithConfig.from_file(path)
    except EdithConfigError as error:
        raise SystemExit(f"Cannot read {path}:\n  {error}") from None
    print(
        f"{display_path(config.path)}: {config.num_sims} trials of "
        f"{display_path(config.base_sim)}, {config.sampling} sampling, "
        f"seed {config.master_seed}"
    )
    header = ("key", "nominal", "distribution")
    rows = [_row(dispersion) for dispersion in config.dispersions]
    widths = [
        max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)
    ]
    for row in (header, *rows):
        cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        print("  " + "  ".join(cells).rstrip())
    return config


def _row(dispersion: BaseDispersion) -> tuple[str, str, str]:
    """Return a dispersion's row of the table: key, nominal and distribution."""
    if not isinstance(dispersion, StandardDispersion):
        return dispersion.key, "", type(dispersion).__name__
    units = dispersion.units
    return (
        dispersion.key,
        _quantity(dispersion.nominal, units),
        _describe(dispersion.distribution, units),
    )


def _describe(distribution: Distribution, units: str) -> str:
    """Return a distribution in words, such as "normal, sigma 0.25 kg"."""
    if isinstance(distribution, Normal):
        parts = [f"normal, sigma {_quantity(distribution.sigma, units)}"]
        if distribution.low is not None:
            parts.append(f"low {_quantity(distribution.low, units)}")
        if distribution.high is not None:
            parts.append(f"high {_quantity(distribution.high, units)}")
        return ", ".join(parts)
    return f"uniform from {distribution.low:g} to {_quantity(distribution.high, units)}"


def _quantity(value: float, units: str) -> str:
    """Return a number with its units, such as "0.25 kg", or bare for none."""
    return f"{value:g} {units}".rstrip()


def _prepare(
    config: EdithConfig,
    path: str,
    overwrite: bool,
    workers: parallel.Workers | None = None,
) -> list[Trial]:
    """Prepare the campaign's run directories behind a progress bar.

    Given workers, a row above the bar counts those ready, with a check once
    all are, and the bars stay up until they are.

    Raises:
        SystemExit: With the error, if the runs folder exists and ``overwrite``
            is not set.
    """
    prepared = runner.prepare(config, overwrite=overwrite)
    try:
        with progress_bars() as bars:
            readiness = None
            if workers is not None:
                readiness = bars.add_task("workers ready", total=workers.count)
                _show_ready(bars, readiness, workers)
            task = bars.add_task("prepare", total=config.num_sims)
            trials = []
            for trial in prepared:
                trials.append(trial)
                bars.update(task, advance=1)
                if workers is not None and readiness is not None:
                    _show_ready(bars, readiness, workers)
            while (
                workers is not None and readiness is not None and not workers.settled()
            ):
                clock.sleep(0.1)
                _show_ready(bars, readiness, workers)
    except FileExistsError as error:
        raise SystemExit(f"{error}; pass --overwrite to write into it") from None
    except EdithConfigError as error:
        raise SystemExit(f"Cannot prepare {path}:\n  {error}") from None
    runs = display_path(config.output_dir / runner.RUNS)
    print(f"Prepared {len(trials)} run directories in {runs}")
    return trials


def _fly(
    config: EdithConfig, workers: parallel.Workers, trials: Sequence[Trial]
) -> CampaignResult:
    """Fly the trials batch by batch on the workers, printing each batch's estimate.

    Raises:
        SystemExit: With the error, if most of a batch's trials fail.
    """
    spans = runner.batches(config)
    print(
        f"Flying {len(trials)} trials on {workers.count} "
        f"worker{'s' if workers.count != 1 else ''} in {len(spans)} "
        f"batch{'es' if len(spans) != 1 else ''} of up to "
        f"{spans[0].stop - spans[0].start}"
    )

    def report(batch: int, stats: BatchStats) -> None:
        half_width = stats.ci_half_width(config.stopping.confidence)
        print(
            f"batch {batch}: {stats.trials} trials, apogee {stats.mean:.6g} "
            f"+/- {half_width:.3g} m"
        )

    try:
        result = runner.fly_campaign(config, workers, trials, report)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    counts = Counter(flown.status for flown in result.results)
    tally = ", ".join(
        f"{counts[status]} {status}" for status in Status if counts[status]
    )
    log = display_path(config.output_dir / parallel.CAMPAIGN_LOG)
    print(f"{len(result.results)} trials: {tally}; see {log}")
    return result


def _show_ready(bars: Progress, task: TaskID, workers: parallel.Workers) -> None:
    """Show how many workers are ready: a red cross until all are, then a check."""
    ready = workers.ready()
    utf = "utf" in bars.console.encoding.lower()
    if ready == workers.count:
        mark = "[green]" + ("✓" if utf else "+") + "[/]"
    else:
        mark = "[red]" + ("✗" if utf else "x") + "[/]"
    bars.update(task, completed=ready, description=f"{mark} workers ready")


def _since(start: float) -> float:
    """Return the seconds since a ``perf_counter`` reading."""
    return clock.perf_counter() - start


def _wall_time(**phases: float) -> str:
    """Return the wall time of each phase, as flight_sim prints a run's."""
    return "wall time: " + ", ".join(f"{name} {s:.1f} s" for name, s in phases.items())


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
