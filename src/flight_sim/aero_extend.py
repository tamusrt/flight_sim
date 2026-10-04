"""Aerodynamics past the RASAero table's last angle of attack, up to 180 degrees.

RASAero II gives the normal force, axial force and centre of pressure up to
about 30 degrees, where its slender-body methods stop being valid. A rocket
that tumbles after apogee flies at any angle, side-on (90 degrees) or tail
first (180 degrees), so the table is carried on to 180 degrees in 10 degree
steps with a crossflow model (Jorgensen's, as in NASA TR R-474 and the
missile Datcom), fitted to the table's own last rows so the two join:

    CN(a) = CNa |sin a cos a| + K Cdc(M sin a) sin^2 a

The first term is the lift that grows linearly at small angles, with the
table's own slope ``CNa``; the second is the crossflow drag of the body and
fins seen side-on, ``Cdc`` the drag of a cylinder across the flow at the
crossflow Mach number ``M sin a`` and ``K`` (planform area over reference
area, times an efficiency) chosen so the model gives the table's normal
force at its last angle (for a table that stops below 20 degrees, never less
than the body's own side area gives, its length estimated from the centre of
pressure). Nothing else about the
shape is needed.

The axial force falls with ``cos a`` to zero side-on; beyond 90 degrees the
base meets the air first and the axial force turns round (a blunt base drags
like a flat plate, ``CA = CA_base cos a`` with ``CA_base`` about 0.9).

The centre of pressure is a blend of the two terms' own: the linear lift
acts where the table's small-angle centre of pressure is (near the fins,
also when flying tail first, where it makes the rocket flip back nose
first), and the crossflow acts at a point fitted from the table's last row
(the middle of the side area). A small blend over 30 degrees removes any
step at the join.

These rows are rough, perhaps 25 percent, which is enough to make a tumble
look and decay like a tumble; they are not for the ascent.
"""

from __future__ import annotations

import math

import numpy as np

STEP_DEG = 10.0
CA_BASE = 0.9  # Axial force coefficient with the base meeting the flow
_BLEND_DEG = 30.0  # Angle over which the fit to the table's last row fades
_ETA = 0.7  # Crossflow efficiency of a long body (Jorgensen)
_FIT_FROM_DEG = 20.0  # Tables reaching this far fix the crossflow on their own


def crossflow_drag(crossflow_mach: float) -> float:
    """Drag coefficient of a long cylinder across the flow (Jorgensen's curve)."""
    m = abs(crossflow_mach)
    if m <= 0.4:
        return 1.2
    if m <= 1.0:
        return 1.2 + (m - 0.4) / 0.6 * 0.6
    if m <= 2.0:
        return 1.8 - (m - 1.0) * 0.3
    return 1.5


def new_alphas(last_deg: float) -> list[float]:
    """The angles to add after a table's last one, every 10 degrees to 180."""
    start = math.floor(last_deg / STEP_DEG) * STEP_DEG + STEP_DEG
    return [float(a) for a in np.arange(start, 180.0 + 1e-9, STEP_DEG)]


def extend(
    alpha_deg: np.ndarray,
    mach: float,
    cn: np.ndarray,
    ca: np.ndarray,
    xcp: np.ndarray,
    ref_length: float = 1.0,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Carry one Mach number's normal force, axial force and CP on to 180 degrees.

    Args:
        alpha_deg (np.ndarray): The table's angles, increasing.
        mach (float): The Mach number of these rows.
        cn (np.ndarray): Normal force coefficient at each angle.
        ca (np.ndarray): Axial force coefficient (positive is drag) at each.
        xcp (np.ndarray): Centre of pressure from the nose, any length unit.
        ref_length (float): The body diameter (the reference length) in the
            same unit, for the least crossflow a body of that length gives.

    Returns:
        The added angles and the three coefficients at each (empty when the
        table already reaches 180 degrees).
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    added = new_alphas(float(alpha_deg[-1]))
    if not added or alpha_deg[-1] >= 180.0:
        return [], np.zeros(0), np.zeros(0), np.zeros(0)
    end = math.radians(float(alpha_deg[-1]))
    cn_end, ca_end, xcp_end = float(cn[-1]), float(ca[-1]), float(xcp[-1])
    # the small-angle slope and centre of pressure: first row above 0, up to 4 degrees
    small = [i for i, a in enumerate(alpha_deg) if 0.0 < a <= 4.0] or [1]
    i0 = small[0]
    a0 = math.radians(float(alpha_deg[i0]))
    slope = float(cn[i0]) / (math.sin(a0) * math.cos(a0)) if cn[i0] > 0 else 0.0
    xcp0 = float(xcp[i0]) if cn[i0] > 0 else xcp_end
    lin_end = slope * abs(math.sin(end) * math.cos(end))
    cross_end = max(cn_end - lin_end, 0.0)
    k = cross_end / (math.sin(end) ** 2 * crossflow_drag(mach * math.sin(end)))
    # a table that stops below 20 degrees says too little about the crossflow:
    # then never less than the body alone side-on, its side area over the
    # reference area, 4 L / (pi D), with L taken from the CP (about 85 percent
    # of the length back) and Jorgensen's efficiency of about 0.7
    if alpha_deg[-1] < _FIT_FROM_DEG and ref_length > 0 and xcp0 > 0:
        k = max(k, _ETA * 4.0 / math.pi * (xcp0 / 0.85) / ref_length)
    x_cross = 0.65 * xcp0
    if cross_end > 1e-6:
        x_cross = (cn_end * xcp_end - lin_end * xcp0) / cross_end
        x_cross = min(max(x_cross, 0.4 * xcp0), 1.1 * xcp0)
    model_end = lin_end + cross_end
    fit = cn_end / model_end if model_end > 1e-9 else 1.0
    ca_base = max(CA_BASE, float(ca[0]))
    cn_out, ca_out, xcp_out = [], [], []
    for deg in added:
        a = math.radians(deg)
        lin = slope * abs(math.sin(a) * math.cos(a))
        cross = k * crossflow_drag(mach * math.sin(a)) * math.sin(a) ** 2
        share = min((deg - alpha_deg[-1]) / _BLEND_DEG, 1.0)
        scale = fit + (1.0 - fit) * share
        total = (lin + cross) * scale
        cn_out.append(total)
        xcp_out.append(
            (lin * xcp0 + cross * x_cross) / (lin + cross)
            if lin + cross > 1e-9
            else x_cross
        )
        if deg <= 90.0:
            ca_out.append(ca_end * math.cos(a) / math.cos(end))
        else:
            ca_out.append(ca_base * math.cos(a))
    return added, np.array(cn_out), np.array(ca_out), np.array(xcp_out)


def extend_values(
    mach_axis: np.ndarray, alpha_axis: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Carry an aero table's grid (Mach, alpha, phi, 6 coefficients) on to 180 degrees.

    The coefficients are [Cx, Cy, Cz, CMx, CMy, CMz] with the moments about the
    nose tip. Each new row is the table's last row scaled: the side force and
    moment components by the change in normal force and in normal force times
    the centre of pressure, so the direction conventions of the table (body or
    missile axes, the sign of phi) carry over as they are.

    Returns:
        The new alpha axis and values; the same ones when nothing is added.
    """
    added = new_alphas(float(alpha_axis[-1]))
    if not added or alpha_axis[-1] >= 180.0:
        return alpha_axis, values
    nm, _, nphi, _ = values.shape
    extra = np.zeros((nm, len(added), nphi, values.shape[3]))
    end_rad = math.radians(float(alpha_axis[-1]))
    for i in range(nm):
        side = np.hypot(values[i, :, :, 1], values[i, :, :, 2]).max(axis=1)
        moment = np.hypot(values[i, :, :, 4], values[i, :, :, 5]).max(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xcp = np.where(side > 1e-9, moment / side, 0.0)
        ca = -values[i, :, 0, 0]
        _, cn_new, ca_new, xcp_new = extend(
            alpha_axis, float(mach_axis[i]), side, ca, xcp
        )
        last = values[i, -1]
        cn_end, m_end = side[-1], moment[-1]
        for j, deg in enumerate(added):
            row = np.empty_like(last)
            row[:, 0] = -ca_new[j]
            ratio = cn_new[j] / cn_end if cn_end > 1e-9 else 0.0
            m_ratio = cn_new[j] * xcp_new[j] / m_end if m_end > 1e-9 else 0.0
            row[:, 1:3] = last[:, 1:3] * ratio
            row[:, 4:6] = last[:, 4:6] * m_ratio
            row[:, 3] = (
                last[:, 3] * max(math.cos(math.radians(deg)), 0.0) / math.cos(end_rad)
            )
            extra[i, j] = row
    return (
        np.concatenate((alpha_axis, np.array(added))),
        np.concatenate((values, extra), axis=1),
    )
