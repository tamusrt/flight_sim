"""Integration kernel tests."""

import pytest

from flight_sim.__main__ import main


def test_placeholder() -> None:
    """Placeholder test to keep the suite non-empty."""
    assert True


def test_main_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() reports a None test input when no arguments are given."""
    monkeypatch.setattr("sys.argv", ["flight_sim"])

    main()

    assert capsys.readouterr().out == "Initializing FS with test input None\n"


def test_main_echoes_test_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() echoes the value passed via --test_input."""
    monkeypatch.setattr("sys.argv", ["flight_sim", "--test_input", "hello"])

    main()

    assert capsys.readouterr().out == "Initializing FS with test input hello\n"


def test_main_rejects_unknown_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse exits with status 2 when an unknown argument is supplied."""
    monkeypatch.setattr("sys.argv", ["flight_sim", "--nope"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2
