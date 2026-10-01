"""Fixtures shared across the test modules."""

import pytest

from flight_sim.units import scalar
from flight_sim.vehicle.rocket_properties import RocketProperties


@pytest.fixture(name="baseline_rocket_properties")
def _baseline_rocket_properties() -> RocketProperties:
    """Provides the standard test rocket."""
    return RocketProperties(
        aero_file_path="tests/test_data/standard_aero.csv",
        motor_file_path="tests/test_data/standard_motor.csv",
        propellant_mass=2.5,
        reference_area=scalar(0.0182414692, "m**2"),
        reference_length=scalar(0.1524, "m"),
    )
