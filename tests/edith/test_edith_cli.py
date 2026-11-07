"""EDITH command line tests."""

from pathlib import Path

import pytest

from edith.__main__ import main


def test_no_arguments_prints_usage_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run with no arguments, the command prints its usage to stderr and exits 2."""
    with pytest.raises(SystemExit) as exit_info:
        main([])

    captured = capsys.readouterr()

    assert exit_info.value.code == 2
    assert captured.err.startswith("usage: edith")
    assert captured.out == ""


def test_prepare_describes_the_campaign_and_writes_its_run_directories(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "edith prepare" prints the settings and dispersions, then the directories."""
    main(["prepare", str(campaign)])

    lines = capsys.readouterr().out.splitlines()
    runs = campaign.parent.parent / "out" / "runs"

    assert lines[0].endswith(
        "3 trials of "
        + (campaign.parent.parent / "base" / "sim.toml").as_posix()
        + ", lhs sampling, seed 7"
    )
    assert lines[1].split() == ["key", "nominal", "distribution"]
    assert lines[2].split()[0] == "properties.dry_mass_properties.mass"
    assert "  17 kg  " in lines[2] and lines[2].endswith("normal, sigma 0.25 kg")
    assert lines[4].endswith("uniform from 0 to 0.2")
    assert lines[-2] == f"Prepared 3 run directories in {runs.as_posix()}"
    assert lines[-1].startswith("wall time: read ")
    assert lines[-1].endswith(" s") and ", prepare " in lines[-1]
    assert sorted(path.name for path in runs.iterdir()) == ["0000", "0001", "0002"]
    with pytest.raises(SystemExit, match="already exists; pass --overwrite"):
        main(["prepare", str(campaign)])


def test_an_invalid_campaign_file_exits_with_its_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A campaign file that does not read ends the command with the error."""
    path = tmp_path / "edith_config.toml"
    path.write_text("num_sims = 0\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        main([str(path)])

    message = str(exit_info.value.code)

    assert message.startswith(f"Cannot read {path}:\n  {path.as_posix()}: missing")
    assert capsys.readouterr().out == ""


def test_a_campaign_flies_on_the_workers_given(
    campaign: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "edith FILE -n 1" prepares and flies every trial on one worker, then tallies."""
    main([str(campaign), "-n", "1"])

    lines = capsys.readouterr().out.splitlines()
    out = campaign.parent.parent / "out"

    assert f"3 trials: 3 OK; see {(out / 'edith.log').as_posix()}" in lines
    assert any(line.startswith("batch 0: 3 trials, apogee ") for line in lines)
    assert lines[-4].startswith("apogee ")
    assert lines[-4].endswith("3 trials in 1 batches)")
    assert lines[-3] == f"Trials: {(out / 'trials.csv').as_posix()}"
    assert lines[-2].startswith("wall time: read ")
    assert ", fly " in lines[-2] and ", total " in lines[-2]
    assert lines[-1].startswith("trials: ")
    assert lines[-1].endswith(" trials/s on 1 worker")
    assert (out / "trials.csv").read_text(encoding="utf-8").count("OK") == 3
    assert len((out / "edith.log").read_text(encoding="utf-8").splitlines()) == 3
    assert all((out / "runs" / f"000{i}" / "run.toml").is_file() for i in range(3))


@pytest.mark.parametrize("workers", ["0", "-2", "two"])
def test_a_bad_worker_count_is_a_usage_error(campaign: Path, workers: str) -> None:
    """-n takes a whole number of at least 1, and exits with the usage otherwise."""
    with pytest.raises(SystemExit) as exit_info:
        main([str(campaign), "-n", workers])

    assert exit_info.value.code == 2
