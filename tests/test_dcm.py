"""Direction cosine matrix tests."""

import math

import numpy as np
import pytest

from flight_sim.utilities.dcm import (
    aero_angles,
    body_to_world,
    missile_to_body,
    world_to_body,
)
from flight_sim.utilities.quaternion import Quaternion


def test_body_to_world_of_identity_quaternion_is_identity() -> None:
    """The identity quaternion leaves every vector where it is."""
    assert body_to_world(Quaternion()) == pytest.approx(np.eye(3))
    assert world_to_body(Quaternion()) == pytest.approx(np.eye(3))


def test_body_to_world_quarter_turn_about_z_maps_x_to_y() -> None:
    """A 90 degree rotation about Z turns the body X axis onto world Y."""
    half_angle = np.pi / 4
    rotation = body_to_world(Quaternion(q_w=np.cos(half_angle), q_z=np.sin(half_angle)))

    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_world_to_body_is_the_transpose_of_body_to_world() -> None:
    """The two matrices of one orientation undo each other."""
    orientation = Quaternion(q_w=0.5, q_x=-0.5, q_y=0.5, q_z=0.5)

    assert world_to_body(orientation) == pytest.approx(body_to_world(orientation).T)
    assert world_to_body(orientation) @ body_to_world(orientation) == pytest.approx(
        np.eye(3)
    )


def test_body_to_world_ignores_quaternion_length() -> None:
    """A scaled quaternion gives the same rotation as its unit counterpart."""
    scaled = body_to_world(Quaternion(q_w=2.0, q_z=2.0))

    assert scaled == pytest.approx(body_to_world(Quaternion(q_w=1.0, q_z=1.0)))
    assert np.linalg.det(scaled) == pytest.approx(1.0)


def test_missile_to_body_columns_are_the_missile_axes_rolled_about_x() -> None:
    """The columns are X, then Y and Z rolled about the nose by phi_a."""
    phi_a = math.radians(30.0)
    matrix = missile_to_body(phi_a)
    x_missile, y_missile, z_missile = matrix.T

    assert x_missile == pytest.approx([1.0, 0.0, 0.0])
    assert y_missile == pytest.approx([0.0, math.cos(phi_a), math.sin(phi_a)])
    assert z_missile == pytest.approx([0.0, -math.sin(phi_a), math.cos(phi_a)])
    assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "airspeed",
    [
        (100.0, 0.0, 0.0),
        (100.0, 50.0, 0.0),
        (100.0, -50.0, 0.0),
        (100.0, 0.0, 50.0),
        (100.0, 0.0, -50.0),
        (3.0, -4.0, 7.0),
        (-1.0, 2.0, -5.0),
    ],
)
def test_aero_angles_lay_the_airspeed_in_the_missile_frame_x_z_plane(
    airspeed: tuple[float, float, float],
) -> None:
    """In the missile frame the airspeed has no Y component and +Z crossflow."""
    airspeed_body = np.array(airspeed)
    alpha_tot, phi_a = aero_angles(airspeed_body)
    speed = np.linalg.norm(airspeed_body)

    airspeed_missile = missile_to_body(phi_a).T @ airspeed_body

    assert airspeed_missile[1] == pytest.approx(0.0, abs=1e-12)
    assert airspeed_missile[2] >= 0.0
    assert airspeed_missile[0] == pytest.approx(speed * math.cos(alpha_tot))
    assert airspeed_missile[2] == pytest.approx(speed * math.sin(alpha_tot))


@pytest.mark.parametrize(
    ("airspeed", "expected_alpha_deg", "expected_phi_deg"),
    [
        ((100.0, 0.0, 0.0), 0.0, 0.0),
        ((-100.0, 0.0, 0.0), 180.0, 0.0),
        ((100.0, 0.0, 100.0), 45.0, 0.0),
        ((0.0, -3.0, 0.0), 90.0, 90.0),
        ((1.0, 0.0, -1.0), 45.0, 180.0),
        ((100.0, 100.0, 0.0), 45.0, 270.0),
        ((0.0, 0.0, 0.0), 0.0, 0.0),
    ],
)
def test_aero_angles_follow_the_crossflow_direction(
    airspeed: tuple[float, float, float],
    expected_alpha_deg: float,
    expected_phi_deg: float,
) -> None:
    """Alpha is the angle from the nose; phi_a rolls missile +Z onto the crossflow."""
    alpha_tot, phi_a = aero_angles(np.array(airspeed))

    assert math.degrees(alpha_tot) == pytest.approx(expected_alpha_deg)
    assert math.degrees(phi_a) == pytest.approx(expected_phi_deg)


def test_aero_angles_phi_stays_below_a_full_turn() -> None:
    """A crossflow a rounding error off +Z reports phi_a as 0, not 2 pi."""
    _alpha_tot, phi_a = aero_angles(np.array([1.0, 1e-300, 1.0]))

    assert 0.0 <= phi_a < math.tau
