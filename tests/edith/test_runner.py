"""EDITH runner tests: the run directories a campaign prepares."""

import tomllib
from pathlib import Path

import pytest

from edith.config import EdithConfig
from edith.runner import draw_batch, prepare, run_directory
from flight_sim.simulation import Simulation


def test_prepare_writes_a_dispersed_run_directory_per_trial(campaign: Path) -> None:
    """Normal mode copies each source file into every run directory."""
    campaign.write_text(
        campaign.read_text(encoding="utf-8").replace(
            'base_sim = "../base/sim.toml"',
            'base_sim = "../base/sim.toml"\ninput_mode = "normal"',
        ),
        encoding="utf-8",
    )
    config = EdithConfig.from_file(campaign)

    directories = [trial.folder for trial in prepare(config)]

    assert directories == [run_directory(config, index) for index in range(3)]
    assert directories[0] == config.output_dir / "runs" / "0000"
    masses = []
    for directory in directories:
        text = (directory / "resolved_input.toml").read_text(encoding="utf-8")
        table = tomllib.loads(text)
        assert table["output"]["directory"] == "."
        assert (directory / "inputs" / "aero.csv").is_file()
        assert not (directory / "inputs" / "aero.parquet").exists()
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
    directories = [trial.folder for trial in prepare(config)]

    with pytest.raises(FileExistsError, match="runs already exists"):
        list(prepare(config))

    again = [trial.folder for trial in prepare(config, overwrite=True)]
    assert again == directories


def test_prepare_writes_the_campaigns_output_settings(campaign: Path) -> None:
    """A trial's input takes the campaign's output keys over the base sim's."""
    text = campaign.read_text(encoding="utf-8")
    campaign.write_text(text + "\n[output]\nplots = []\n", encoding="utf-8")
    config = EdithConfig.from_file(campaign)

    directory = next(prepare(config)).folder

    text = (directory / "resolved_input.toml").read_text(encoding="utf-8")
    output = tomllib.loads(text)["output"]
    assert output["plots"] == []
    assert output["directory"] == "."
    assert "formats" in output


def test_prepare_compresses_csv_inputs(campaign: Path) -> None:
    """Compressed mode converts CSV inputs before copying them."""
    campaign.write_text(
        campaign.read_text(encoding="utf-8").replace(
            "num_batches = 1", 'num_batches = 1\ninput_mode = "compressed"'
        ),
        encoding="utf-8",
    )
    config = EdithConfig.from_file(campaign)

    directory = next(prepare(config)).folder

    assert (directory / "inputs" / "aero.parquet").is_file()
    assert not (directory / "inputs" / "aero.csv").exists()


def test_prepare_lightweight_writes_only_trial_overrides(campaign: Path) -> None:
    """Lightweight mode extends the base sim without copying its inputs."""
    campaign.write_text(
        campaign.read_text(encoding="utf-8").replace(
            "num_batches = 1", 'num_batches = 1\ninput_mode = "lightweight"'
        ),
        encoding="utf-8",
    )
    config = EdithConfig.from_file(campaign)

    directory = next(prepare(config)).folder
    text = (directory / "resolved_input.toml").read_text(encoding="utf-8")
    table = tomllib.loads(text)

    assert not (directory / "inputs").exists()
    assert table["extends"] == config.base_sim.as_posix()
    assert Path(table["extends"]).is_absolute()
    assert "properties" in table
    assert "config" in table
    assert table["output"]["directory"] == "."
    assert "aero_table" not in table["properties"]
    simulation = Simulation.from_file(directory / "resolved_input.toml")
    assert simulation.name
    assert simulation.properties.dry_mass_properties.mass.m_as("kg") == pytest.approx(
        float(table["properties"]["dry_mass_properties"]["mass"].split()[0])
    )
    assert simulation.config.truth.launch_rail is not None
    assert simulation.config.truth.launch_rail.friction_coefficient == pytest.approx(
        table["config"]["truth"]["launch_rail"]["friction_coefficient"]
    )


def test_prepare_draws_each_trial_from_senais_batch_streams(campaign: Path) -> None:
    """A trial's numbers are the campaign's draw_batch values for its batch."""
    text = campaign.read_text(encoding="utf-8")
    campaign.write_text(
        text.replace(
            "num_batches = 1\nflight_per_batch = 3",
            "num_batches = 2\nflight_per_batch = 2",
        ),
        encoding="utf-8",
    )
    config = EdithConfig.from_file(campaign)

    trials = list(prepare(config))

    assert len(trials) == 4
    for batch in (0, 1):
        drawn = draw_batch(config, batch, 2)
        for index in (0, 1):
            trial = trials[2 * batch + index]
            expected = {key: float(values[index]) for key, values in drawn.items()}
            assert trial.drawn == pytest.approx(expected)
