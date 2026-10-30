"""Aerodynamic CSV loading tests."""

import math
from dataclasses import astuple, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flight_sim.units import scalar, vector
from flight_sim.utilities.data_loader import (
    AeroCoefficients,
    AeroFrame,
    AeroTable,
    Extrapolation,
    aero_table_from_csv,
)

_AERO_CSV = "tests/test_data/standard_aero.csv"


def _csv_table(extrapolation: Extrapolation = "linear") -> AeroTable:
    """Return the body-frame test table, over Mach 0.1 to 0.2 and alpha 0 to 5."""
    return aero_table_from_csv(
        _AERO_CSV,
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
        frame="body",
        extrapolation=extrapolation,
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


@pytest.mark.parametrize(
    ("mach", "alpha", "linear", "clamped"),
    [
        (0.4, 0.0, -0.65, -0.55),  # Above the top Mach
        (0.0, 0.0, -0.45, -0.5),  # Below the bottom Mach
        (0.1, 10.0, -0.7, -0.6),  # Beyond the top alpha
    ],
)
def test_extrapolation_sets_coefficients_off_the_grid(
    mach: float, alpha: float, linear: float, clamped: float
) -> None:
    """Off the grid, "linear" extends the edge cell and "clamp" holds the edge."""
    assert _csv_table("linear")(mach, alpha, 0.0).cx == pytest.approx(linear)
    assert _csv_table("clamp")(mach, alpha, 0.0).cx == pytest.approx(clamped)


def test_clamp_leaves_coefficients_on_the_grid_alone() -> None:
    """Within the grid both extrapolations interpolate the same way."""
    assert _csv_table("clamp")(0.15, 2.5, 45.0) == _csv_table("linear")(0.15, 2.5, 45.0)


@pytest.mark.parametrize("option", [{"frame": "Missile"}, {"extrapolation": "nearest"}])
def test_aero_table_rejects_an_unknown_option(option: dict[str, Any]) -> None:
    """A frame or extrapolation that is not one of its options is rejected."""
    with pytest.raises(ValueError, match="Unknown"):
        replace(_csv_table(), **option)


def _table(
    values: np.ndarray, alphas: list[float], frame: AeroFrame = "body"
) -> AeroTable:
    """Return a unit-reference table over two Mach numbers and one turn of phi."""
    return AeroTable(
        mach_axis=np.array([0.1, 0.2]),
        alpha_axis=np.array(alphas),
        phi_axis=np.array([0.0, 360.0]),
        values=values,
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
        frame=frame,
        extrapolation="linear",
    )


def _missile_table() -> AeroTable:
    """Return the test table's axisymmetric loads tabulated in the missile frame."""
    return _table(
        np.tile([-0.6, 0.0, -0.1, 0.0, -20.0], (2, 2, 2, 1)), [0.0, 5.0], "missile"
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


def _cp_values(alphas: list[float], xcp: list[float]) -> np.ndarray:
    """Return values whose normal force grows linearly with alpha at a given CP."""
    values = np.zeros((2, len(alphas), 2, 5))
    values[..., 0] = -0.5
    values[..., 2] = -0.3 * np.array(alphas)[None, :, None]
    values[..., 4] = np.array(xcp)[None, :, None]
    return values


@pytest.mark.parametrize("alpha", [0.01, 0.5, 2.5])
def test_cp_holds_between_alpha_rows(alpha: float) -> None:
    """The force acts at the interpolated CP, so the CP holds between rows."""
    table = _table(_cp_values([0.0, 1.0, 4.0], [-10.0] * 3), [0.0, 1.0, 4.0])

    coefficients = table(0.15, alpha, 0.0)

    assert coefficients.cz == pytest.approx(-0.3 * alpha)
    assert coefficients.cmy == pytest.approx(-10.0 * 0.3 * alpha)
    assert -coefficients.cmy / coefficients.cz == pytest.approx(-10.0)


def test_cp_approaches_its_zero_alpha_limit() -> None:
    """Below the first nonzero alpha row the CP moves toward the zero alpha xCP."""
    table = _table(_cp_values([0.0, 1.0], [-12.0, -11.0]), [0.0, 1.0])

    coefficients = table(0.15, 0.25, 0.0)

    assert -coefficients.cmy / coefficients.cz == pytest.approx(-11.75)


@pytest.mark.parametrize("phi", [0.0, 45.0, 200.0])
def test_cp_moment_turns_with_missile_frame_force(phi: float) -> None:
    """In the missile frame the moment stays that of the force at the CP."""
    values = _cp_values([0.0, 1.0, 4.0], [-10.0] * 3)
    values[..., 3] = 0.05
    table = _table(values, [0.0, 1.0, 4.0], "missile")

    coefficients = table(0.15, 2.0, phi)
    force = np.array([coefficients.cx, coefficients.cy, coefficients.cz])
    moment = np.array([coefficients.cmx, coefficients.cmy, coefficients.cmz])

    assert moment == pytest.approx(
        np.cross([-10.0, 0.0, 0.0], force) + np.array([0.05, 0.0, 0.0]), abs=1e-12
    )


def test_values_must_match_the_axes() -> None:
    """A table without one column per CSV coefficient is refused."""
    with pytest.raises(ValueError, match="shape"):
        _table(np.zeros((2, 2, 2, 7)), [0.0, 5.0])


def test_values_must_be_finite() -> None:
    """A missing or undefined xCP is refused."""
    values = _cp_values([0.0, 1.0], [-10.0, -10.0])
    values[0, 0, 0, 4] = math.nan

    with pytest.raises(ValueError, match="finite"):
        _table(values, [0.0, 1.0])


def test_zero_alpha_xcp_far_from_the_next_row_is_flagged() -> None:
    """A zero alpha xCP left at a placeholder is flagged."""
    with pytest.warns(UserWarning, match="Zero alpha xCP"):
        _table(_cp_values([0.0, 1.0], [0.0, -12.0]), [0.0, 1.0])


def test_power_on_cx_applies_only_while_powered() -> None:
    """A sixth column replaces Cx while the motor burns."""
    values = np.zeros((2, 2, 2, 6))
    values[..., 0], values[..., 5] = -0.5, -0.4
    table = AeroTable(
        mach_axis=np.array([0.1, 0.8]),
        alpha_axis=np.array([0.0, 5.0]),
        phi_axis=np.array([0.0, 360.0]),
        values=values,
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
        frame="body",
        extrapolation="linear",
    )

    assert table(0.3, 2.0, 0.0).cx == pytest.approx(-0.5)
    assert table(0.3, 2.0, 0.0, powered=True).cx == pytest.approx(-0.4)
    # Without the column, powered flight keeps the power-off Cx
    assert _csv_table()(0.3, 2.0, 0.0, powered=True) == _csv_table()(0.3, 2.0, 0.0)


def test_csv_reads_power_on_cx_when_present(tmp_path: Path) -> None:
    """A CxPowerOn column adds a sixth value to the table."""
    path = tmp_path / "aero.csv"
    rows = ["Mach,Alpha,Phi,Cx,Cy,Cz,CMx,xCP,CxPowerOn"]
    for mach in (0.5, 1.5):
        for alpha in (0, 5):
            for phi in (0, 360):
                rows.append(f"{mach},{alpha},{phi},-0.5,0,0,0,-1,-0.4")
    path.write_text("\n".join(rows))

    table = aero_table_from_csv(
        path,
        reference_area=scalar(1.0, "m**2"),
        reference_length=scalar(1.0, "m"),
        reference_point=vector((0.0, 0.0, 0.0), "m"),
        frame="body",
        extrapolation="linear",
    )

    assert table.values.shape[-1] == 6
    assert _csv_table().values.shape[-1] == 5
