"""Reynolds number drag correction tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from support import truth

from flight_sim.environment.atmosphere import StandardAtmosphere1976
from flight_sim.integration import IntegrationConfiguration, flight_point_si
from flight_sim.units import vector
from flight_sim.vehicle.reynolds_correction import ReynoldsCorrection
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_CORRECTION = ReynoldsCorrection(
    np.array([0.1, 0.2, 0.4, 0.8]), np.array([0.6, 0.5, 0.5, 0.4])
)


def test_correction_follows_the_low_mach_drag_curve() -> None:
    """Thinner air lowers the Reynolds number and adds the table's own drag,
    until the drag stops falling at the roughness limit."""
    sea_level = StandardAtmosphere1976().conditions(0.0)
    # Half the density halves the Reynolds number: Mach 0.2 flies like 0.1
    half = sea_level._replace(air_density=sea_level.air_density / 2)

    assert _CORRECTION(0.2, sea_level) == pytest.approx(0.0)
    assert _CORRECTION(0.2, half) == pytest.approx(-0.1)
    # Mach 0.8 flies like 0.4, both past the limit: the later fall is not Reynolds
    assert _CORRECTION(0.8, half) == pytest.approx(0.0)
    assert _CORRECTION(0.2, half._replace(air_density=0.0)) == 0.0


def test_csv_reads_sorted_rows_with_blanks_unknown(tmp_path: Path) -> None:
    """Rows load in Mach order, and a blank CxViscous is unknown."""
    path = tmp_path / "viscous.csv"
    path.write_text("Mach,CxViscous\n0.4,0.5\n0.1,0.6\n0.8,\n")

    correction = ReynoldsCorrection.from_csv(path)

    assert np.array_equal(correction.mach_axis, [0.1, 0.4, 0.8])
    assert np.array_equal(correction.cx_viscous[:2], [0.6, 0.5])
    assert np.isnan(correction.cx_viscous[2])


def test_csv_reads_the_alpha_0_rows_of_an_aero_table(tmp_path: Path) -> None:
    """An aero table gives its zero-alpha rows once, whatever other angles it has."""
    path = tmp_path / "aero.csv"
    path.write_text(
        "Mach,Alpha,Phi,Cx,CxViscous\n"
        "0.1,0,0,-0.9,0.6\n0.1,0,360,-0.9,0.6\n0.1,5,0,-1.0,0.7\n0.4,0,0,-0.6,0.5\n"
    )

    correction = ReynoldsCorrection.from_csv(path)

    assert np.array_equal(correction.mach_axis, [0.1, 0.4])
    assert np.array_equal(correction.cx_viscous, [0.6, 0.5])


def test_mismatched_rows_are_refused() -> None:
    """There must be one CxViscous per Mach."""
    with pytest.raises(ValueError, match="one CxViscous per Mach"):
        ReynoldsCorrection(np.array([0.1, 0.2]), np.array([0.6]))


def test_rocket_correction_adds_to_the_flight_cx(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """A rocket with a correction flies its table's Cx plus the correction."""
    config = IntegrationConfiguration(truth=truth())
    # High up, where the air is thin enough to correct
    state = RocketState(
        position=vector((5000.0, 0.0, 0.0), "m"),
        velocity=vector((100.0, 5.0, 0.0), "m/s"),
    )
    corrected = replace(unpowered_rocket_properties, reynolds_correction=_CORRECTION)

    plain = flight_point_si(0.0, state, unpowered_rocket_properties, config)
    point = flight_point_si(0.0, state, corrected, config)
    change = _CORRECTION(point.aero.mach, point.environment.atmosphere)

    assert change != 0.0
    assert point.aero.coefficients.cx == pytest.approx(
        plain.aero.coefficients.cx + change
    )
