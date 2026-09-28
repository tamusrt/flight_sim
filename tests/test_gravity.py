"""Gravity model tests."""

import numpy as np
import pytest

from flight_sim.environment.gravity import ConstantGravity, WGS84Gravity


def test_wgs84_gravity_at_the_equator_and_pole() -> None:
    """Surface gravity matches the WGS84 equatorial and polar values."""
    model = WGS84Gravity()

    assert model.magnitude(0.0, 0.0) == pytest.approx(9.7803253359, rel=1e-9)
    assert model.magnitude(np.pi / 2, 0.0) == pytest.approx(9.8321849378, rel=1e-9)


def test_wgs84_gravity_decreases_with_altitude() -> None:
    """Gravity weakens with height above the ellipsoid."""
    model = WGS84Gravity()
    latitude = np.radians(45.0)

    assert model.magnitude(latitude, 10000.0) < model.magnitude(latitude, 0.0)
    assert model.magnitude(latitude, 10000.0) == pytest.approx(
        model.magnitude(latitude, 0.0), rel=1e-2
    )


def test_constant_gravity_ignores_location() -> None:
    """A constant model reports the same magnitude everywhere."""
    model = ConstantGravity(9.81)

    assert model.magnitude(0.0, 0.0) == 9.81
    assert model.magnitude(np.pi / 2, 50000.0) == 9.81
