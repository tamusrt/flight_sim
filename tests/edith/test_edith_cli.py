"""EDITH command line tests."""

from pathlib import Path

import pytest

from edith.__main__ import main

_CAMPAIGN = "examples/sol_invictus_mc/edith_config.toml"


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


def test_a_campaign_file_is_read_and_described(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given the example, the command prints its settings and every dispersion."""
    main([_CAMPAIGN])

    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == (
        f"{_CAMPAIGN}: 500 trials of examples/sol_invictus/sim.toml, "
        "lhs sampling, seed 9999"
    )
    assert lines[1] == (
        "  properties.dry_mass_properties.mass: 17 kg, "
        "Normal(sigma=0.25, low=None, high=None)"
    )
    assert lines[4] == (
        "  config.truth.launch_rail.friction_coefficient: 0, Uniform(low=0.0, high=0.2)"
    )
    assert lines[-1] == "Running the campaign is not built yet."


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
