"""Quaternion module tests."""

import numpy as np
import pytest

from flight_sim.units import vector
from flight_sim.utilities.quaternion import (
    Quaternion,
    cross,
    quaternion_kinematics,
    rotation_matrix,
)


def test_quaternion_normalized_returns_unit_length() -> None:
    """Normalizing rescales every component by the quaternion's length."""
    unit = Quaternion(q_w=2.0, q_x=0.0, q_y=-2.0, q_z=1.0).normalized()

    assert unit.q_w == pytest.approx(2 / 3)
    assert unit.q_x == pytest.approx(0.0)
    assert unit.q_y == pytest.approx(-2 / 3)
    assert unit.q_z == pytest.approx(1 / 3)


def test_rotation_matrix_of_identity_quaternion_is_identity() -> None:
    """The identity quaternion leaves every vector where it is."""
    assert rotation_matrix(1.0, 0.0, 0.0, 0.0) == pytest.approx(np.eye(3))


def test_rotation_matrix_quarter_turn_about_z_maps_x_to_y() -> None:
    """A 90 degree rotation about Z turns the body X axis onto world Y."""
    half_angle = np.pi / 4
    rotation = rotation_matrix(np.cos(half_angle), 0.0, 0.0, np.sin(half_angle))

    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0])


def test_rotation_matrix_ignores_quaternion_length() -> None:
    """A scaled quaternion gives the same rotation as its unit counterpart."""
    scaled = rotation_matrix(2.0, 0.0, 0.0, 2.0)

    assert scaled == pytest.approx(rotation_matrix(1.0, 0.0, 0.0, 1.0))


def test_cross_matches_numpy() -> None:
    """The cross product agrees with numpy for a general pair of vectors."""
    a = np.array([1.0, -2.0, 3.5])
    b = np.array([-0.5, 4.0, 2.0])

    assert cross(a, b) == pytest.approx(np.cross(a, b))


def test_quaternion_kinematics_matches_xi_matrix() -> None:
    """The quaternion rate equals 0.5 * Xi(q) * omega for a general orientation."""
    orientation = Quaternion(q_w=0.5, q_x=-0.5, q_y=0.5, q_z=0.5)
    angular_velocity = vector((1.0, 2.0, 3.0), "rad/s")

    q_dot = quaternion_kinematics(orientation, angular_velocity)

    assert q_dot.q_w == pytest.approx(-1.0)
    assert q_dot.q_x == pytest.approx(0.5)
    assert q_dot.q_y == pytest.approx(1.5)
    assert q_dot.q_z == pytest.approx(0.0)


def test_quaternion_kinematics_accepts_any_angular_rate_unit() -> None:
    """An angular velocity in deg/s gives the same rate as the equivalent rad/s."""
    from_deg = quaternion_kinematics(Quaternion(), vector((0.0, 0.0, 180.0), "deg/s"))

    assert from_deg.q_z == pytest.approx(np.pi / 2)
