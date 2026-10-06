"""Prototype: _state_rates flattened to scalar math on plain arrays.

Written so the same source runs as pure Python or under numba.njit. Only
supports the default models (StandardAtmosphere1976, UniformWind,
WGS84Gravity, SolidEngine, missile-frame AeroTable).
"""

import math

import numba
import numpy as np

from flight_sim.environment import atmosphere as A
from flight_sim.environment import gravity as G

LAYER_BASE = np.array(A._LAYER_BASE_M)
LAYER_LAPSE = np.array(A._LAYER_LAPSE_K_PER_M)
LAYER_T = np.array(A._LAYER_BASE_TEMPERATURE_K)
LAYER_P = np.array(A._LAYER_BASE_PRESSURE_PA)


@numba.njit(cache=True)
def cell(axis, x):
    i = np.searchsorted(axis, x, side="right") - 1
    return min(max(i, 0), axis.shape[0] - 2)


@numba.njit(cache=True)
def atmosphere(alt):
    alt = min(max(alt, -5000.0), 86000.0)
    h = 6356766.0 * alt / (6356766.0 + alt)
    layer = max(np.searchsorted(LAYER_BASE, h, side="right") - 1, 0)
    t0 = LAYER_T[layer]
    dh = h - LAYER_BASE[layer]
    lapse = LAYER_LAPSE[layer]
    temp = t0 + lapse * dh
    k = A._G0 * A._MOLAR_MASS / A._GAS_CONSTANT
    if lapse == 0.0:
        p = LAYER_P[layer] * math.exp(-k * dh / t0)
    else:
        p = LAYER_P[layer] * (temp / t0) ** (-k / lapse)
    rho = p / (A._AIR_GAS_CONSTANT * temp)
    a = math.sqrt(A._HEAT_CAPACITY_RATIO * A._AIR_GAS_CONSTANT * temp)
    return rho, a


@numba.njit(cache=True)
def gravity(lat, alt):
    s2 = math.sin(lat) ** 2
    g0 = (G._WGS84_EQUATORIAL_GRAVITY * (1.0 + G._WGS84_SOMIGLIANA_CONSTANT * s2)
          / math.sqrt(1.0 - G._WGS84_ECCENTRICITY_SQUARED * s2))
    return g0 * (1.0 - 2.0 / G._WGS84_SEMI_MAJOR_AXIS_M
                 * (1.0 + G._WGS84_FLATTENING + G._ROTATION_PARAMETER
                    - 2.0 * G._WGS84_FLATTENING * s2) * alt
                 + 3.0 * alt * alt / G._WGS84_SEMI_MAJOR_AXIS_M ** 2)


@numba.njit(cache=True)
def rates(t, y, out,
          # engine
          times, thrusts, impulses, ign,
          # grain: m0, outer_sq, core_sq, len_sq, cg(3)
          grain, grain_cg,
          # casing / dry: mass, cg(3), inertia(3x3)
          cas_m, cas_cg, cas_I, dry_m, dry_cg, dry_I,
          # aero
          mach_ax, alpha_ax, phi_ax, table, s_ref, l_ref, ref_pt,
          # env
          wind, lat, elev,
          # rail: on_rail flag, direction(3), friction, start(3)
          on_rail, rdir, mu, rstart):
    px, py, pz, vx, vy, vz, p, q, r, qw, qx, qy, qz = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12])
    alt = elev + px

    # DCM world->body from normalised quaternion
    n = 1.0 / math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    w, a, b, c = qw * n, qx * n, qy * n, qz * n
    c00 = w*w + a*a - b*b - c*c; c01 = 2*(a*b + w*c); c02 = 2*(a*c - w*b)
    c10 = 2*(a*b - w*c); c11 = w*w - a*a + b*b - c*c; c12 = 2*(b*c + w*a)
    c20 = 2*(a*c + w*b); c21 = 2*(b*c - w*a); c22 = w*w - a*a - b*b + c*c

    # Engine: thrust and burned fraction
    bt = t - ign
    thrust = 0.0
    bf = 0.0
    if bt >= times[-1]:
        bf = 1.0
    elif bt > times[0]:
        i = np.searchsorted(times, bt, side="right") - 1
        into = bt - times[i]
        slope = (thrusts[i + 1] - thrusts[i]) / (times[i + 1] - times[i])
        thrust = thrusts[i] + slope * into
        bf = (impulses[i] + into * (thrusts[i] + 0.5 * slope * into)) / impulses[-1]
    if bt == times[0]:
        thrust = thrusts[0]
    elif bt == times[-1]:
        thrust = thrusts[-1]

    # Grain (diagonal inertia about its own CG)
    rem = 1.0 - bf
    osq = grain[1]
    isq = osq - rem * (osq - grain[2])
    gm = grain[0] * rem
    g_ax = 0.5 * gm * (osq + isq)
    g_tr = gm * (3.0 * (osq + isq) + grain[3]) / 12.0

    # combine(casing, grain) then combine(dry, engine)
    em = cas_m + gm
    dx = cas_cg[0] - grain_cg[0]; dy = cas_cg[1] - grain_cg[1]; dz = cas_cg[2] - grain_cg[2]
    k = cas_m * gm / em if em > 0 else 0.0
    d2 = dx*dx + dy*dy + dz*dz
    e00 = cas_I[0, 0] + g_ax + k * (d2 - dx*dx)
    e11 = cas_I[1, 1] + g_tr + k * (d2 - dy*dy)
    e22 = cas_I[2, 2] + g_tr + k * (d2 - dz*dz)
    e01 = cas_I[0, 1] - k * dx*dy; e02 = cas_I[0, 2] - k * dx*dz; e12 = cas_I[1, 2] - k * dy*dz
    ecx = grain_cg[0] + cas_m / em * dx; ecy = grain_cg[1] + cas_m / em * dy; ecz = grain_cg[2] + cas_m / em * dz

    m = dry_m + em
    dx = dry_cg[0] - ecx; dy = dry_cg[1] - ecy; dz = dry_cg[2] - ecz
    k = dry_m * em / m
    d2 = dx*dx + dy*dy + dz*dz
    i00 = dry_I[0, 0] + e00 + k * (d2 - dx*dx)
    i11 = dry_I[1, 1] + e11 + k * (d2 - dy*dy)
    i22 = dry_I[2, 2] + e22 + k * (d2 - dz*dz)
    i01 = dry_I[0, 1] + e01 - k * dx*dy
    i02 = dry_I[0, 2] + e02 - k * dx*dz
    i12 = dry_I[1, 2] + e12 - k * dy*dz
    cgx = ecx + dry_m / m * dx; cgy = ecy + dry_m / m * dy; cgz = ecz + dry_m / m * dz

    # Aero
    rho, sos = atmosphere(alt)
    wx, wy, wz = vx - wind[0], vy - wind[1], vz - wind[2]
    ubx = c00*wx + c01*wy + c02*wz
    uby = c10*wx + c11*wy + c12*wz
    ubz = c20*wx + c21*wy + c22*wz
    speed = math.sqrt(ubx*ubx + uby*uby + ubz*ubz)
    alpha = math.degrees(math.atan2(math.hypot(uby, ubz), ubx))
    phi_r = math.atan2(uby, ubz) % (2 * math.pi)
    if phi_r >= 2 * math.pi:
        phi_r = 0.0
    phi = math.degrees(phi_r) % 360.0
    mach = speed / sos
    i = cell(mach_ax, mach); j = cell(alpha_ax, alpha); kk = cell(phi_ax, phi)
    tm = (mach - mach_ax[i]) / (mach_ax[i + 1] - mach_ax[i])
    ta = (alpha - alpha_ax[j]) / (alpha_ax[j + 1] - alpha_ax[j])
    tp = (phi - phi_ax[kk]) / (phi_ax[kk + 1] - phi_ax[kk])
    co = np.empty(6)
    for col in range(6):
        v000 = table[i, j, kk, col]; v001 = table[i, j, kk + 1, col]
        v010 = table[i, j + 1, kk, col]; v011 = table[i, j + 1, kk + 1, col]
        v100 = table[i + 1, j, kk, col]; v101 = table[i + 1, j, kk + 1, col]
        v110 = table[i + 1, j + 1, kk, col]; v111 = table[i + 1, j + 1, kk + 1, col]
        a00 = v000 + tp * (v001 - v000); a01 = v010 + tp * (v011 - v010)
        a10 = v100 + tp * (v101 - v100); a11 = v110 + tp * (v111 - v110)
        b0 = a00 + ta * (a01 - a00); b1 = a10 + ta * (a11 - a10)
        co[col] = b0 + tm * (b1 - b0)
    # missile -> body: passive rotation about x by phi
    cp, sp = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    cx = co[0]; cy = cp * co[1] + sp * co[2]; cz = -sp * co[1] + cp * co[2]
    cmx = co[3]; cmy = cp * co[4] + sp * co[5]; cmz = -sp * co[4] + cp * co[5]

    fs = 0.5 * rho * speed * speed * s_ref
    fx, fy, fz = fs * cx, fs * cy, fs * cz
    lx, ly, lz = ref_pt[0] - cgx, ref_pt[1] - cgy, ref_pt[2] - cgz
    ml = fs * l_ref
    tx = ml * cmx + (ly * fz - lz * fy)
    ty = ml * cmy + (lz * fx - lx * fz)
    tz = ml * cmz + (lx * fy - ly * fx)

    fx += thrust
    # body -> world is the transpose
    ax = (c00*fx + c10*fy + c20*fz) / m - gravity(lat, alt)
    ay = (c01*fx + c11*fy + c21*fz) / m
    az = (c02*fx + c12*fy + c22*fz) / m

    if on_rail:
        along = ax*rdir[0] + ay*rdir[1] + az*rdir[2]
        cx_, cy_, cz_ = ax - along*rdir[0], ay - along*rdir[1], az - along*rdir[2]
        fr = mu * math.sqrt(cx_*cx_ + cy_*cy_ + cz_*cz_)
        sp_ = vx*rdir[0] + vy*rdir[1] + vz*rdir[2]
        if sp_ > 0.0:
            along -= fr
        elif sp_ < 0.0:
            along += fr
        else:
            along = math.copysign(max(abs(along) - fr, 0.0), along)
        if (px-rstart[0])*rdir[0] + (py-rstart[1])*rdir[1] + (pz-rstart[2])*rdir[2] <= 0.0:
            along = max(along, 0.0)
        ax, ay, az = along*rdir[0], along*rdir[1], along*rdir[2]
        dp = dq = dr = 0.0
    else:
        # Euler: I w_dot = tau - w x (I w), solved by the symmetric adjugate
        hx = i00*p + i01*q + i02*r; hy = i01*p + i11*q + i12*r; hz = i02*p + i12*q + i22*r
        bx = tx - (q*hz - r*hy); by = ty - (r*hx - p*hz); bz = tz - (p*hy - q*hx)
        a00 = i11*i22 - i12*i12; a01 = i02*i12 - i01*i22; a02 = i01*i12 - i02*i11
        a11 = i00*i22 - i02*i02; a12 = i01*i02 - i00*i12; a22 = i00*i11 - i01*i01
        det = i00*a00 + i01*a01 + i02*a02
        dp = (a00*bx + a01*by + a02*bz) / det
        dq = (a01*bx + a11*by + a12*bz) / det
        dr = (a02*bx + a12*by + a22*bz) / det

    out[0] = vx; out[1] = vy; out[2] = vz
    out[3] = ax; out[4] = ay; out[5] = az
    out[6] = dp; out[7] = dq; out[8] = dr
    out[9] = 0.5 * (-qx*p - qy*q - qz*r)
    out[10] = 0.5 * (qw*p - qz*q + qy*r)
    out[11] = 0.5 * (qz*p + qw*q - qx*r)
    out[12] = 0.5 * (-qy*p + qx*q + qw*r)
    return out


def params(inputs):
    """Flatten a _StepInputs into the positional args of rates()."""
    pr = inputs.properties
    eng = pr.engine
    gr = eng.grain
    tab = pr.aero_table
    rail = inputs.rail
    return (
        eng.times, eng.thrusts, eng._impulses, eng._ignition_s,
        np.array([gr._mass_kg, gr._outer_radius_sq, gr._core_radius_sq, gr._length_sq]), gr._cg_m,
        eng.casing.si.mass, eng.casing.si.cg_location, eng.casing.si.inertia,
        pr.dry_mass_properties.si.mass, pr.dry_mass_properties.si.cg_location, pr.dry_mass_properties.si.inertia,
        tab.mach_axis, tab.alpha_axis, tab.phi_axis, np.ascontiguousarray(tab.values),
        tab.reference_area_m2, tab.reference_length_m, tab.reference_point_m,
        inputs.wind.velocity(0.0), inputs.latitude_rad, inputs.elevation_m,
        rail is not None,
        rail[0].direction() if rail else np.zeros(3), rail[0].friction_coefficient if rail else 0.0,
        np.asarray(rail[1], dtype=float) if rail else np.zeros(3),
    )
