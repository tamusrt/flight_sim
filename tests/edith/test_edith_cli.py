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
    assert lines[1].startswith(
        "  StandardDispersion(key='properties.dry_mass_properties.mass', "
        "nominal=17.0, distribution=Normal(mean=17.0, sigma=0.25, "
    )
    assert lines[-1] == f"Prepared 3 run directories in {runs.as_posix()}"
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
