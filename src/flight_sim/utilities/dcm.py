"""Direction cosine matrices between the simulation's frames, over r3f.

World frame: fixed to the pad, +X up (altitude is the X coordinate and
gravity acts along -X), Y and Z horizontal, right-handed. Body frame: +X out
the nose, Y and Z transverse, right-handed; it coincides with the world frame
on the pad, so the identity quaternion is the launch attitude. Missile frame:
the body frame rolled about X by the aerodynamic roll angle ``phi_a`` so that
the airspeed lies in its X-Z plane with the crossflow along its +Z axis; the
total angle of attack ``alpha_tot`` is the angle from its X axis to the
airspeed. Aerodynamic coefficients are indexed by ``alpha_tot`` and ``phi_a``
and give body-frame force and moment components. Reference:
https://ntrs.nasa.gov/api/citations/20130003336/downloads/20130003336.pdf

r3f matrices are passive, taking the components of a vector from one frame
to another; each function's name gives the direction.
"""

import math

import numpy as np
import r3f

from flight_sim.utilities.quaternion import Quaternion


def world_to_body(orientation: Quaternion) -> np.ndarray:
    """Return the matrix taking world components to body components.

    Args:
        orientation (Quaternion): Body-to-world rotation, of any length.

    Returns:
        np.ndarray: 3x3 ``C`` with ``v_body = C @ v_world``.
    """
    components = np.array(
        [orientation.q_w, orientation.q_x, orientation.q_y, orientation.q_z]
    )
    matrix: np.ndarray = r3f.quat_to_dcm(
        components / math.sqrt(float(components @ components))
    )
    return matrix


def body_to_world(orientation: Quaternion) -> np.ndarray:
    """Return the matrix taking body components to world components.

    Args:
        orientation (Quaternion): Body-to-world rotation, of any length.

    Returns:
        np.ndarray: 3x3 ``C`` with ``v_world = C @ v_body``.
    """
    return world_to_body(orientation).T


def missile_to_body(phi_a_rad: float) -> np.ndarray:
    """Return the matrix taking missile frame components to body components.

    Args:
        phi_a_rad (float): Aerodynamic roll angle in radians.

    Returns:
        np.ndarray: 3x3 ``C`` with ``v_body = C @ v_missile``.
    """
    matrix: np.ndarray = r3f.dcm(phi_a_rad, "x").T
    return matrix


def aero_angles(airspeed_body: np.ndarray) -> tuple[float, float]:
    """Compute the total angle of attack and the aerodynamic roll angle.

    Args:
        airspeed_body (np.ndarray): Velocity relative to the air, body components.

    Returns:
        tuple[float, float]: ``(alpha_tot, phi_a)`` in radians, in [0, pi]
            and [0, 2 pi).
    """
    v_x, v_y, v_z = airspeed_body.tolist()
    alpha_tot = math.atan2(math.hypot(v_y, v_z), v_x)
    phi_a = math.atan2(-v_y, v_z) % math.tau
    # A crossflow a rounding error off +Z wraps to exactly 2 pi
    if phi_a >= math.tau:
        phi_a = 0.0
    return alpha_tot, phi_a
