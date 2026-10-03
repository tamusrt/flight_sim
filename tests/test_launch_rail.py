"""Launch rail tests."""

from dataclasses import replace

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.units import scalar, vector
from flight_sim.utilities.dcm import body_to_world
from flight_sim.vehicle.rocket_state import RocketState


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


def test_distance_along_is_measured_up_the_rail_from_its_foot() -> None:
    """Only the component of the offset from the foot along the rail counts."""
    rail = _rail(30.0, 90.0)
    start = np.array([100.0, 20.0, 0.0])
    up_the_rail = start + 3.0 * rail.direction()
    sideways = np.array([0.0, 0.0, 7.0])  # North, across an east-facing rail

    assert rail.distance_along(up_the_rail, start) == pytest.approx(3.0)
    assert rail.distance_along(up_the_rail + sideways, start) == pytest.approx(3.0)
    assert rail.distance_along(start - rail.direction(), start) == pytest.approx(-1.0)


def test_mount_puts_the_rail_foot_at_the_rocket() -> None:
    """Mounting points the nose up the rail and starts the rail at the rocket."""
    rail = _rail(80.0, 45.0)
    state = RocketState(
        position=vector((500.0, 30.0, -20.0), "m"),
        velocity=vector((1.0, 2.0, 3.0), "m/s"),
    )

    mounted = rail.mount(state)

    assert mounted.orientation == rail.orientation()
    assert mounted.rail_start is not None
    assert mounted.rail_start.m_as("m") == pytest.approx([500.0, 30.0, -20.0])
    assert mounted.velocity.m_as("m/s") == pytest.approx([1.0, 2.0, 3.0])
    assert state.rail_start is None


def test_exit_event_counts_the_distance_left_on_the_rail() -> None:
    """The event value falls from the length at the foot to zero at the end."""
    rail = _rail(60.0, 0.0)
    mounted = rail.mount(RocketState(position=vector((10.0, 0.0, 5.0), "m")))
    moved = replace(
        mounted, position=vector(np.array([10.0, 0.0, 5.0]) + rail.direction(), "m")
    )

    assert rail.exit_event.value(0.0, mounted) == pytest.approx(5.0)
    assert rail.exit_event.value(0.0, moved) == pytest.approx(4.0)
    assert rail.exit_event.value(0.0, RocketState()) == pytest.approx(5.0)


def test_rail_rejects_a_negative_friction_coefficient() -> None:
    """Friction cannot push the rocket along its motion."""
    with pytest.raises(ValueError, match="Friction coefficient"):
        LaunchRail(length=scalar(5.0, "m"), friction_coefficient=-0.1)


def test_rail_is_immutable() -> None:
    """The cached geometry cannot go stale through reassignment."""
    rail = _rail(85.0, 0.0)

    with pytest.raises(AttributeError):
        rail.length = scalar(1.0, "m")  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        rail.direction()[0] = 0.0
