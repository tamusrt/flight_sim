"""Wind model tests."""

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.environment.wind import UniformWind
from flight_sim.units import scalar


@pytest.mark.parametrize(
    ("from_azimuth_deg", "velocity"),
    [(0.0, (0.0, 0.0, -5.0)), (90.0, (0.0, -5.0, 0.0)), (225.0, (0.0, 3.5355, 3.5355))],
)
def test_uniform_wind_blows_away_from_its_azimuth(
    from_azimuth_deg: float, velocity: tuple[float, float, float]
) -> None:
    """A wind from an azimuth blows horizontally the opposite way, at every altitude."""
    wind = UniformWind(
        speed=scalar(5.0, "m/s"), from_azimuth=scalar(from_azimuth_deg, "deg")
    )

    assert wind.velocity(0.0) == pytest.approx(velocity, abs=1e-4)
    assert wind.velocity(30000.0) == pytest.approx(velocity, abs=1e-4)


def test_default_wind_is_calm() -> None:
    """With no speed given the air is still."""
    assert np.array_equal(UniformWind().velocity(0.0), np.zeros(3))


def test_uniform_wind_rejects_a_speed_that_is_not_a_speed() -> None:
    """A speed in the wrong dimension is rejected on construction."""
    with pytest.raises(DimensionalityError, match=r"UniformWind\.speed"):
        UniformWind(speed=scalar(5.0, "m"))
