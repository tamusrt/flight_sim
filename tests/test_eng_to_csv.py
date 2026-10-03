"""Tests for converting RASP .eng motor files."""

from pathlib import Path

import numpy as np
import pytest

from flight_sim.units import matrix, scalar, vector
from flight_sim.utilities.data_loader import eng_to_csv
from flight_sim.vehicle.engine import PropellantGrain, solid_engine_from_csv
from flight_sim.vehicle.mass_properties import MassProperties

_GRAIN = PropellantGrain(
    mass=scalar(1.2, "kg"),
    length=scalar(0.4, "m"),
    outer_diameter=scalar(0.07, "m"),
    core_diameter=scalar(0.02, "m"),
    cg_location=vector((-1.0, 0.0, 0.0), "m"),
)
_CASING = MassProperties(
    mass=scalar(1.3, "kg"),
    cg_location=vector((-1.0, 0.0, 0.0), "m"),
    inertia=matrix(np.diag((0.001, 0.02, 0.02)), "kg*m**2"),
)

_ENG = """; A test motor
; second comment
TEST 75 500 0 1.2 2.5 Team   ; description
0.1 100.0
0.5 120.0 ; inline comment

1.0 0.0
NEXT 75 500 0 1.2 2.5 Team
0.1 999.0
"""


def test_eng_file_becomes_a_csv_beside_it(tmp_path: Path) -> None:
    """Comments and the second motor are skipped and (0, 0) is added."""
    source = tmp_path / "motor.ENG"
    source.write_text(_ENG, encoding="utf-8")

    csv_path = Path(eng_to_csv(str(source)))

    assert csv_path == tmp_path / "motor.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert data["Time"] == pytest.approx([0.0, 0.1, 0.5, 1.0])
    assert data["Thrust"] == pytest.approx([0.0, 100.0, 120.0, 0.0])


def test_engine_reads_an_eng_file(tmp_path: Path) -> None:
    """The motor model accepts the .eng file directly."""
    source = tmp_path / "motor.eng"
    source.write_text(_ENG, encoding="utf-8")
    engine = solid_engine_from_csv(str(source), _GRAIN, _CASING)
    assert engine.get_thrust(0.05) == pytest.approx(50.0)
    assert engine.total_impulse == pytest.approx(5.0 + 44.0 + 30.0)


def test_csv_paths_are_left_alone() -> None:
    """A CSV is returned unchanged and not rewritten."""
    path = "tests/test_data/standard_motor.csv"
    assert eng_to_csv(path) == path


@pytest.mark.parametrize(
    "body",
    [
        "; only comments\nNAME 1 2 0 1 2 X\n",
        "NAME 1 2 0 1 2 X\n0.5 10\n0.5 12\n",
        "NAME 1 2 0 1 2 X\n0.1 10\n0.5\n0.9 20\n",
        "NAME 1 2 0 1 2 X\n0.1 10\n0.5 abc\n",
    ],
)
def test_bad_eng_files_are_rejected(tmp_path: Path, body: str) -> None:
    """No samples, a broken sample, or non-increasing times raise ValueError."""
    source = tmp_path / "bad.eng"
    source.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.eng"):
        eng_to_csv(str(source))
