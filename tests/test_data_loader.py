"""Aerodynamic CSV loading tests."""

import pytest

from flight_sim.utilities.data_loader import AeroCoefficients, aero_table_from_csv


def test_aero_table_returns_coefficients_by_column_name() -> None:
    """Each field is filled from its own CSV column, whatever the column order."""
    table = aero_table_from_csv("tests/test_data/standard_aero.csv")

    assert table(0.1, 0.0) == AeroCoefficients(
        cd=0.5, cl=0.0, cy=0.0, c_roll=0.0, cm=-0.1, cn=0.0
    )


def test_aero_table_interpolates_between_grid_points() -> None:
    """A point inside a grid cell is interpolated bilinearly."""
    coefficients = aero_table_from_csv("tests/test_data/standard_aero.csv")(0.15, 2.5)

    assert isinstance(coefficients, AeroCoefficients)
    assert coefficients.cd == pytest.approx(0.575)
    assert coefficients.cl == pytest.approx(0.055)
    assert coefficients.cm == pytest.approx(-0.1)
