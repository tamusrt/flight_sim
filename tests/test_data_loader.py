"""Aerodynamic CSV loading tests."""

import math
from dataclasses import astuple

import numpy as np
import pytest

from flight_sim.units import scalar, vector
from flight_sim.utilities.data_loader import (
    AeroCoefficients,
    AeroTable,
    aero_table_from_csv,
)

_AERO_CSV = "tests/test_data/standard_aero.csv"


def _csv_table() -> AeroTable:
    """Return the body-frame test table."""
    return aero_table_from_csv(
        _AERO_CSV,
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
    )


def test_aero_table_returns_coefficients_by_column_name() -> None:
    """Each field is filled from its own CSV column, whatever the column order."""
    assert _csv_table()(0.1, 5.0, 90.0) == AeroCoefficients(
        cx=-0.6, cy=-0.1, cz=0.0, cmx=0.0, cmy=0.0, cmz=2.0
    )


def test_aero_table_interpolates_between_grid_points() -> None:
    """A point inside a grid cell is interpolated trilinearly."""
    coefficients = _csv_table()(0.15, 2.5, 45.0)

    assert isinstance(coefficients, AeroCoefficients)
    assert coefficients.cx == pytest.approx(-0.575)
    assert coefficients.cy == pytest.approx(-0.0275)
    assert coefficients.cz == pytest.approx(-0.0275)
    assert coefficients.cmx == pytest.approx(0.0)
    assert coefficients.cmy == pytest.approx(-0.55)
    assert coefficients.cmz == pytest.approx(0.55)


@pytest.mark.parametrize("phi", [-90.0, 630.0])
def test_aero_table_wraps_phi_into_one_turn(phi: float) -> None:
    """A phi outside [0, 360) looks up the same point as its wrapped angle."""
    table = _csv_table()

    assert table(0.1, 5.0, phi) == table(0.1, 5.0, 270.0)


def _missile_table() -> AeroTable:
    """Return the test table's axisymmetric loads tabulated in the missile frame."""
    return AeroTable(
        mach_axis=np.array([0.1, 0.2]),
        alpha_axis=np.array([0.0, 5.0]),
        phi_axis=np.array([0.0, 360.0]),
        values=np.tile([-0.6, 0.0, -0.1, 0.0, -2.0, 0.0], (2, 2, 2, 1)),
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
        frame="missile",
    )


@pytest.mark.parametrize("phi", [0.0, 45.0, 200.0, 300.0])
def test_missile_frame_loads_turn_with_the_crossflow(phi: float) -> None:
    """Missile-frame loads keep their size at any phi, opposing the crossflow."""
    sin_phi, cos_phi = math.sin(math.radians(phi)), math.cos(math.radians(phi))

    coefficients = _missile_table()(0.1, 5.0, phi)

    assert astuple(coefficients) == pytest.approx(
        (-0.6, -0.1 * sin_phi, -0.1 * cos_phi, 0.0, -2.0 * cos_phi, 2.0 * sin_phi)
    )


@pytest.mark.parametrize("phi", [0.0, 90.0, 180.0, 270.0])
def test_missile_frame_matches_body_frame_on_the_grid(phi: float) -> None:
    """At the body table's grid angles both frames give the same body loads."""
    missile = _missile_table()(0.1, 5.0, phi)
    body = _csv_table()(0.1, 5.0, phi)

    assert astuple(missile) == pytest.approx(astuple(body), abs=1e-12)
