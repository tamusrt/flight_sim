"""EDITH runner tests: the run directories a campaign prepares."""

import tomllib
from pathlib import Path

import pytest

from edith.config import EdithConfig
from edith.runner import prepare, run_directory
from edith.sampler import sample


def test_prepare_writes_a_dispersed_run_directory_per_trial(campaign: Path) -> None:
    """Each trial gets a run directory whose input holds its drawn values."""
    config = EdithConfig.from_file(campaign)

    directories = list(prepare(config, sample(config)))

    assert directories == [run_directory(config, index) for index in range(3)]
    assert directories[0] == config.output_dir / "runs" / "0000"
    masses = []
    for directory in directories:
        text = (directory / "resolved_input.toml").read_text(encoding="utf-8")
        table = tomllib.loads(text)
        assert table["output"]["directory"] == "."
        assert (directory / "inputs" / "aero.parquet").is_file()
        assert not (directory / "inputs" / "aero.csv").exists()
        mass = table["properties"]["dry_mass_properties"]["mass"]
        assert mass.endswith(" kg")
        masses.append(float(mass.split()[0]))
        cg_x = table["properties"]["dry_mass_properties"]["cg_location"]["value"][0]
        assert isinstance(cg_x, float) and abs(cg_x + 1.0) <= 0.02
        friction = table["config"]["truth"]["launch_rail"]["friction_coefficient"]
        assert 0.0 <= friction <= 0.2
    assert len(set(masses)) == 3
    assert all(abs(mass - 17.0) < 1.5 for mass in masses)


def test_prepare_refuses_an_existing_runs_folder_unless_overwritten(
    campaign: Path,
) -> None:
    """A second prepare needs overwrite, and then writes into the same folders."""
    config = EdithConfig.from_file(campaign)
    samples = sample(config)
    directories = list(prepare(config, samples))

    with pytest.raises(FileExistsError, match="runs already exists"):
        list(prepare(config, samples))

    assert list(prepare(config, samples, overwrite=True)) == directories


def test_prepare_writes_the_campaigns_output_settings(campaign: Path) -> None:
    """A trial's input takes the campaign's output keys over the base sim's."""
    text = campaign.read_text(encoding="utf-8")
    campaign.write_text(text + "\n[output]\nplots = []\n", encoding="utf-8")
    config = EdithConfig.from_file(campaign)

    directory = next(prepare(config, sample(config)))

    text = (directory / "resolved_input.toml").read_text(encoding="utf-8")
    output = tomllib.loads(text)["output"]
    assert output["plots"] == []
    assert output["directory"] == "."
    assert "formats" in output


def test_prepare_can_leave_csv_inputs_uncompressed(campaign: Path) -> None:
    """The campaign flag can preserve the source CSVs in prepared runs."""
    campaign.write_text(
        campaign.read_text(encoding="utf-8").replace(
            "num_sims = 3", "num_sims = 3\ncompress_inputs = false"
        ),
        encoding="utf-8",
    )
    config = EdithConfig.from_file(campaign)

    directory = next(prepare(config, sample(config)))

    assert (directory / "inputs" / "aero.csv").is_file()
    assert not (directory / "inputs" / "aero.parquet").exists()
