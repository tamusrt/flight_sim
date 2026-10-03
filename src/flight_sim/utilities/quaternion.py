"""Quaternion orientation and the vector math built on it.

Quaternion components are dimensionless, so the functions here work on
floats and SI arrays. ``quaternion_kinematics`` is the quantity-taking entry
point.
"""

import math
from dataclasses import dataclass

import numpy as np

from flight_sim.units import Vector


@dataclass
class Quaternion:
    """Quaternion representation tracking vehicle orientation."""

    # Components are dimensionless by definition, so they stay plain floats.
    q_w: float = 1.0
    q_x: float = 0.0
    q_y: float = 0.0
    q_z: float = 0.0

    def normalized(self) -> "Quaternion":
        """Rescale to unit length so the quaternion stays a valid rotation.

        Returns:
            Quaternion: A unit quaternion pointing the same way as this one.
        """
        norm = math.sqrt(self.q_w**2 + self.q_x**2 + self.q_y**2 + self.q_z**2)
        return Quaternion(
            q_w=self.q_w / norm,
            q_x=self.q_x / norm,
            q_y=self.q_y / norm,
            q_z=self.q_z / norm,
        )


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two 3-vectors.

    Args:
        a (np.ndarray): Left operand.
        b (np.ndarray): Right operand.

    Returns:
        np.ndarray: ``a x b``.
    """
    a_x, a_y, a_z = a.tolist()
    b_x, b_y, b_z = b.tolist()
    return np.array(
        [a_y * b_z - a_z * b_y, a_z * b_x - a_x * b_z, a_x * b_y - a_y * b_x]
    )


def quaternion_rates(
    q_w: float, q_x: float, q_y: float, q_z: float, angular_velocity: np.ndarray
) -> np.ndarray:
    """Evaluate q_dot = 0.5 * Xi(q) * omega.

    Args:
        q_w (float): Scalar component.
        q_x (float): Vector component along the body X axis.
        q_y (float): Vector component along the body Y axis.
        q_z (float): Vector component along the body Z axis.
        angular_velocity (np.ndarray): Body-frame angular velocity in rad/s.

    Returns:
        np.ndarray: Rate of change of (q_w, q_x, q_y, q_z), in 1/s.
    """
    p, q, r = angular_velocity.tolist()
    return 0.5 * np.array(
        [
            -q_x * p - q_y * q - q_z * r,
            q_w * p - q_z * q + q_y * r,
            q_z * p + q_w * q - q_x * r,
            -q_y * p + q_x * q + q_w * r,
        ]
    )


def quaternion_kinematics(
    orientation: Quaternion, angular_velocity: Vector
) -> Quaternion:
    """Compute the quaternion rate from the body-frame angular velocity.

    Evaluates the kinematic differential equation q_dot = 0.5 * Xi(q) * omega.

    Args:
        orientation (Quaternion): Current body-to-world orientation.
        angular_velocity (Vector): Angular velocity expressed in the body frame.

    Returns:
        Quaternion: Rate of change of each orientation component, in 1/s.
    """
    q_dot = quaternion_rates(
        orientation.q_w,
        orientation.q_x,
        orientation.q_y,
        orientation.q_z,
        angular_velocity.m_as("rad/s"),
    ).tolist()
    return Quaternion(q_w=q_dot[0], q_x=q_dot[1], q_y=q_dot[2], q_z=q_dot[3])
