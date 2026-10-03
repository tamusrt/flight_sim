# ruff: noqa: E501
# pylint: disable=line-too-long
"""Tests for turning RASAero "Run Test" alpha files into the aero CSV."""

import csv
import math
from pathlib import Path

import pytest

from flight_sim.whatif.ras_csv import Options, convert, main, read_alpha_file

_MACH = (0.1, 0.5, 1.0, 2.0)
_IN = 0.0254


def _block(mach: float, alpha: int, xcp_in: float = 80.0) -> str:
    """One Mach point of an alpha file: the six numeric lines RASAero writes."""
    m = f"{mach:.2f}"
    cn, ca = 0.2 * alpha, 0.5 + 0.01 * alpha
    return (
        f" {m}   0.00    0.509    0.509   0.309   0.023   0.015  0.091   0.071   0.000   0.000    1039946\n"
        f" {m}  11.13  137.029\n"
        f" {m}   {alpha:.2f}    0.777    0.000\n"
        f" {m}   {alpha:.2f}    {cn:.3f}  {xcp_in:.3f}\n"
        f" {m}   {alpha:.2f}    0.739    0.564   {cn:.3f}   {ca:.3f}\n"
        f" {m}   {alpha:.2f}    0.739    0.564   {cn:.3f}   {ca:.3f}\n\n"
    )


def _write(folder: Path, alphas: tuple[int, ...] = (0, 1, 2)) -> Path:
    """Alpha files in the layout RASAero writes, banner line included."""
    for alpha in alphas:
        text = "   -------->  BODY TRANSITIONING TO TURBULENT FLOW  <--------\n\n"
        text += "".join(_block(mach, alpha) for mach in _MACH)
        (folder / f"alpha{alpha}.txt").write_text(text)
    return folder


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_alpha_file_is_read_per_mach_point(tmp_path: Path) -> None:
    """Each Mach point gives its CN, CA and centre of pressure."""
    blocks = read_alpha_file(_write(tmp_path) / "alpha2.txt")
    assert sorted(blocks) == [10, 50, 100, 200]
    assert blocks[50]["cn"] == pytest.approx(0.4)
    assert blocks[50]["ca"] == pytest.approx(0.52)
    assert blocks[50]["xcp_in"] == pytest.approx(80.0)


def test_csv_has_the_flight_sim_columns_and_conventions(tmp_path: Path) -> None:
    """Cx = -CA, Cz = -CN cos(phi), Cy = -CN sin(phi), moments about the nose tip."""
    out = tmp_path / "aero.csv"
    summary = convert(_write(tmp_path), out)
    assert summary.alphas == [0, 1, 2] and summary.rows == 4 * 3 * 25
    assert not summary.power_differs
    rows = _rows(out)
    assert list(rows[0]) == [
        "Mach",
        "Alpha",
        "Phi",
        "Cx",
        "Cy",
        "Cz",
        "CMx",
        "CMy",
        "CMz",
    ]
    row = next(
        r for r in rows if r["Mach"] == "0.5" and r["Alpha"] == "2" and r["Phi"] == "90"
    )
    assert float(row["Cx"]) == pytest.approx(-0.52)
    assert float(row["Cy"]) == pytest.approx(-0.4 * math.sin(math.pi / 2), abs=1e-5)
    assert float(row["Cz"]) == pytest.approx(0.0, abs=1e-5)
    flat = next(
        r for r in rows if r["Mach"] == "0.5" and r["Alpha"] == "2" and r["Phi"] == "0"
    )
    xcp = 80.0 * _IN
    assert float(flat["Cz"]) == pytest.approx(-0.4)
    assert float(flat["CMy"]) == pytest.approx(xcp * -0.4 / 0.1524, rel=1e-4)
    assert float(flat["CMx"]) == 0.0


def test_mach_points_are_thinned_and_capped(tmp_path: Path) -> None:
    """Every 0.01 to Mach 1.5, every 0.05 above, nothing past the maximum."""
    out = tmp_path / "aero.csv"
    summary = convert(_write(tmp_path), out, Options(max_mach=1.2))
    assert summary.mach == (0.1, 1.0)
    assert {r["Mach"] for r in _rows(out)} == {"0.1", "0.5", "1"}


def test_alpha_files_must_agree(tmp_path: Path) -> None:
    """Different Mach points, a wrong alpha inside a file, or no files all stop the run."""
    with pytest.raises(SystemExit, match="No alpha"):
        convert(tmp_path, tmp_path / "x.csv")
    _write(tmp_path)
    (tmp_path / "alpha1.txt").write_text(_block(0.1, 1))
    with pytest.raises(SystemExit, match="same Mach"):
        convert(tmp_path, tmp_path / "x.csv")
    _write(tmp_path)
    (tmp_path / "alpha2.txt").write_text("".join(_block(mach, 7) for mach in _MACH))
    with pytest.raises(SystemExit, match=r"alpha2\.txt contains alpha"):
        convert(tmp_path, tmp_path / "x.csv")


def test_truncated_file_is_an_error(tmp_path: Path) -> None:
    """A file cut off mid-block (RASAero closed early) is not silently used."""
    _write(tmp_path)
    text = (tmp_path / "alpha0.txt").read_text().rstrip("\n").rsplit("\n", 1)[0]
    (tmp_path / "alpha0.txt").write_text(text)
    with pytest.raises(ValueError, match="multiple of 6"):
        convert(tmp_path, tmp_path / "x.csv")


def test_command_line_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--in``/``--out`` and the default paths beside ``here`` both work."""
    _write(tmp_path)
    main(["--in", str(tmp_path), "--out", str(tmp_path / "chosen.csv")])
    assert (tmp_path / "chosen.csv").exists()
    assert "Read 3 alpha files" in capsys.readouterr().out
    main([], here=tmp_path)
    assert (tmp_path / "aero.csv").exists()
