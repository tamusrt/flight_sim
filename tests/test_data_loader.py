"""Aerodynamic CSV loading tests."""

import pytest

from flight_sim.utilities.data_loader import AeroCoefficients, aero_table_from_csv

_AERO_CSV = "tests/test_data/standard_aero.csv"


def test_aero_table_returns_coefficients_by_column_name() -> None:
    """Each field is filled from its own CSV column, whatever the column order."""
    table = aero_table_from_csv(_AERO_CSV)

    assert table(0.1, 5.0, 90.0) == AeroCoefficients(
        cx=-0.6, cy=0.1, cz=0.0, cmx=0.0, cmy=0.0, cmz=-0.1
    )


def test_aero_table_interpolates_between_grid_points() -> None:
    """A point inside a grid cell is interpolated trilinearly."""
    coefficients = aero_table_from_csv(_AERO_CSV)(0.15, 2.5, 45.0)

    assert isinstance(coefficients, AeroCoefficients)
    assert coefficients.cx == pytest.approx(-0.575)
    assert coefficients.cy == pytest.approx(0.0275)
    assert coefficients.cz == pytest.approx(-0.0275)
    assert coefficients.cmx == pytest.approx(0.0)
    assert coefficients.cmy == pytest.approx(-0.025)
    assert coefficients.cmz == pytest.approx(-0.025)


@pytest.mark.parametrize("phi", [-90.0, 630.0])
def test_aero_table_wraps_phi_into_one_turn(phi: float) -> None:
    """A phi outside [0, 360) looks up the same point as its wrapped angle."""
    table = aero_table_from_csv(_AERO_CSV)

    assert table(0.1, 5.0, phi) == table(0.1, 5.0, 270.0)
