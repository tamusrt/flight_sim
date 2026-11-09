"""Prepare a campaign's run directories, one dispersed sim per trial.

Each trial's directory is a flight_sim run directory: ``resolved_input.toml``
with the trial's drawn values and the campaign's output settings in place of
the base sim's, and copies of the data files, so ``flight_sim run`` flies it
on its own.
"""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from edith.config import EdithConfig, output_values
from edith.dispersion import Trial
from edith.sampler import draw_slices
from flight_sim import output
from flight_sim.config.toml_files import display_path

# Folder of a campaign's output directory holding a run directory per trial
RUNS = "runs"


def run_directory(config: EdithConfig, index: int) -> Path:
    """Return the run directory of a trial, such as ``<output_dir>/runs/0007``."""
    width = max(4, len(str(config.num_sims - 1)))
    return config.output_dir / RUNS / f"{index:0{width}d}"


def prepare(
    config: EdithConfig, samples: np.ndarray, *, overwrite: bool = False
) -> Iterator[Path]:
    """Create a run directory per trial, holding its dispersed input.

    Args:
        config (EdithConfig): The campaign.
        samples (np.ndarray): The trials' uniform numbers, as ``sample`` draws
            them.
        overwrite (bool, optional): Write into a runs folder that already
            exists. Defaults to refusing.

    Yields:
        Path: Each trial's run directory, in trial order, as it is written.

    Raises:
        FileExistsError: If the runs folder exists and ``overwrite`` is not set.
    """
    runs = config.output_dir / RUNS
    if runs.exists() and not overwrite:
        raise FileExistsError(f"{display_path(runs)} already exists")
    slices = draw_slices(config.dispersions)
    runs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parquet-", dir=runs) as cache:
        source = (
            output.parquet_source(config.source, Path(cache))
            if config.input_mode == "compressed"
            else config.source
        )
        for index in range(config.num_sims):
            folder = run_directory(config, index)
            folder.mkdir(parents=True, exist_ok=True)
            trial = Trial(folder)
            for dispersion, columns in zip(config.dispersions, slices, strict=True):
                dispersion.apply(samples[index, columns], trial)
            trial.values.update(output_values(config.output))
            trial.values["output.directory"] = folder.as_posix()
            if config.input_mode == "lightweight":
                lightweight_values = dict(trial.values)
                lightweight_values["output.directory"] = "."
                yield output.prepare_lightweight(
                    config.source, lightweight_values, folder
                )
            else:
                yield output.prepare_source(
                    source, trial.values, folder, overwrite=True
                )
