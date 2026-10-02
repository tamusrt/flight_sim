"""Fixtures shared across the test modules."""

import pytest

from flight_sim.units import scalar, zero_vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.vehicle.rocket_properties import RocketProperties


@pytest.fixture(name="baseline_rocket_properties")
def _baseline_rocket_properties() -> RocketProperties:
    """Provides the standard rocket with the hand-made test aero table.

    The table's center of pressure is 3.048 m aft of the nose tip.
    """
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "tests/test_data/standard_aero.csv",
            reference_area=scalar(0.0182414692, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=zero_vector("m"),
        ),
        motor_file_path="tests/test_data/standard_motor.csv",
        propellant_mass=5.0,
    )
