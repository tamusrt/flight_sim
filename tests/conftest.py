"""Fixtures shared across the test modules."""

import pytest

from flight_sim.__main__ import get_default_properties
from flight_sim.vehicle.rocket_properties import RocketProperties


@pytest.fixture(name="baseline_rocket_properties")
def _baseline_rocket_properties() -> RocketProperties:
    """Provides the standard test rocket, whose center of pressure is 3.048 m aft."""
    return get_default_properties()
