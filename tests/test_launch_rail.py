"""Launch rail tests."""

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.units import scalar
from flight_sim.utilities.dcm import body_to_world


def _rail(elevation_deg: float, azimuth_deg: float) -> LaunchRail:
    """Return a 5 m rail at the given elevation and azimuth."""
    return LaunchRail(
        length=scalar(5.0, "m"),
        elevation=scalar(elevation_deg, "deg"),
        azimuth=scalar(azimuth_deg, "deg"),
    )


@pytest.mark.parametrize(
    ("elevation_deg", "azimuth_deg", "direction"),
    [
        (90.0, 0.0, (1.0, 0.0, 0.0)),
        (0.0, 0.0, (0.0, 0.0, 1.0)),
        (0.0, 90.0, (0.0, 1.0, 0.0)),
        (30.0, 180.0, (0.5, 0.0, -np.sqrt(3) / 2)),
    ],
)
def test_rail_points_along_its_elevation_and_azimuth(
    elevation_deg: float, azimuth_deg: float, direction: tuple[float, float, float]
) -> None:
    """Elevation is above the horizon; azimuth runs clockwise from north (+Z)."""
    assert _rail(elevation_deg, azimuth_deg).direction() == pytest.approx(
        direction, abs=1e-12
    )


@pytest.mark.parametrize(
    ("elevation_deg", "azimuth_deg"), [(85.0, 0.0), (80.0, 90.0), (60.0, 225.0)]
)
def test_rocket_on_the_rail_points_its_nose_up_it(
    elevation_deg: float, azimuth_deg: float
) -> None:
    """The rail's attitude turns the body +X axis onto the rail."""
    rail = _rail(elevation_deg, azimuth_deg)

    nose = body_to_world(rail.orientation())[:, 0]

    assert nose == pytest.approx(rail.direction())


def test_vertical_rail_gives_the_pad_attitude() -> None:
    """A vertical rail leaves the body axes on the world axes."""
    assert body_to_world(_rail(90.0, 135.0).orientation()) == pytest.approx(np.eye(3))


def test_rail_rejects_an_elevation_that_is_not_an_angle() -> None:
    """An elevation in the wrong dimension is rejected on construction."""
    with pytest.raises(DimensionalityError, match=r"LaunchRail\.elevation"):
        LaunchRail(length=scalar(5.0, "m"), elevation=scalar(85.0, "m"))
