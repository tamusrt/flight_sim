"""EDITH command line tests."""
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
