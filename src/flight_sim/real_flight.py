"""Turn a Blue Raven flight log into viewer telemetry, to compare with the sim.

The log holds the vertical, downrange and crossrange inertial solution in
feet, plus the tilt from vertical and the roll angle. It has no heading, so
downrange is laid along the launch rail's azimuth with crossrange to its
right, and the nose tilts toward the horizontal velocity. Inertial positions
are 16 bit counts of feet and wrap, so they are unwrapped first.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import r3f

from flight_sim.utilities.dcm import world_direction

_FT_TO_M = 0.3048
_WRAP_FT = 65536.0
# Horizontal speed below which the nose heading falls back to the rail's, m/s
_MIN_HEADING_SPEED_M_S = 5.0


def _quaternion_from_axes(nose: np.ndarray, side: np.ndarray) -> list[float]:
    """Body-to-world quaternion [w, x, y, z] of a body with these axes."""
    body_to_world = np.column_stack((nose, side, np.cross(nose, side)))
    quaternion = r3f.dcm_to_quat(body_to_world.T)
    return [float(x) for x in quaternion]


def _attitudes(
    log: pd.DataFrame, velocity: np.ndarray, along: np.ndarray, end: int
) -> list[list[float]]:
    """Quaternions from tilt, roll and a heading toward the horizontal velocity."""
    tilt = np.radians(log["Tilt_Angle_(deg)"].to_numpy(dtype=float))
    roll = np.radians(log["Roll_Angle_(deg)"].to_numpy(dtype=float))
    quaternions = []
    for k in range(end):
        horizontal = velocity[k] * np.array([0.0, 1.0, 1.0])
        speed = float(np.linalg.norm(horizontal))
        heading = horizontal / speed if speed > _MIN_HEADING_SPEED_M_S else along
        nose = np.cos(tilt[k]) * np.array([1.0, 0.0, 0.0]) + np.sin(tilt[k]) * heading
        side0 = np.cross(np.array([1.0, 0.0, 0.0]), heading)
        side0 /= float(np.linalg.norm(side0))
        third = np.cross(nose, side0)
        side = math.cos(roll[k]) * side0 + math.sin(roll[k]) * third
        quaternions.append(_quaternion_from_axes(nose, side))
    return quaternions


def real_flight_telemetry(  # pylint: disable=too-many-locals
    log_path: str | Path, azimuth_deg: float
) -> dict[str, object]:
    """Read a Blue Raven log in the format the flight viewer expects.

    Args:
        log_path (str | Path): The ``LR`` flight log CSV.
        azimuth_deg (float): Compass azimuth that downrange points along.

    Returns:
        dict[str, object]: Telemetry with time from liftoff, cut at the
            first touchdown after apogee.
    """
    log = pd.read_csv(log_path)
    log = log[log["Flight_Time_(s)"] >= 0.0].reset_index(drop=True)
    time = log["Flight_Time_(s)"].to_numpy(dtype=float)

    def feet(column: str) -> np.ndarray:
        return np.unwrap(log[column].to_numpy(dtype=float), period=_WRAP_FT)

    up = feet("Inertial_Altitude") * _FT_TO_M
    down_range = feet("Inertial_DR_Position") * _FT_TO_M
    cross_range = feet("Inertial_CR_position") * _FT_TO_M
    vertical_speed = log["Velocity_Up"].to_numpy(dtype=float) * _FT_TO_M
    range_speed = log["Velocity_DR"].to_numpy(dtype=float) * _FT_TO_M
    cross_speed = log["Velocity_CR"].to_numpy(dtype=float) * _FT_TO_M

    azimuth = math.radians(azimuth_deg)
    along = world_direction(0.0, azimuth)
    right = world_direction(0.0, azimuth + math.pi / 2)
    position = np.outer(up, [1.0, 0.0, 0.0]) + np.outer(down_range, along)
    position += np.outer(cross_range, right)
    velocity = np.outer(vertical_speed, [1.0, 0.0, 0.0])
    velocity += np.outer(range_speed, along) + np.outer(cross_speed, right)

    apogee = int(np.argmax(vertical_speed <= 0.0)) if np.any(vertical_speed <= 0) else 0
    below = np.flatnonzero((up <= 0.0) & (np.arange(up.size) > apogee))
    end = int(below[0]) + 1 if below.size else time.size
    quaternions = _attitudes(log, velocity, along, end)

    def first(column: str) -> float | None:
        hits = np.flatnonzero(log[column].to_numpy() > 0)
        return float(time[hits[0]]) if hits.size and hits[0] < end else None

    events: list[dict[str, object]] = []
    for kind, name, when in (
        ("burnout", "Burnout (flight computer)", first("Burnout_Coast")),
        ("apogee", "Apogee (inertial velocity zero)", float(time[apogee])),
        ("reco", "Apogee channel fired", first("Apo_fired")),
        ("reco", "Main channel fired", first("Main_fired")),
    ):
        if when is not None and when <= float(time[end - 1]):
            events.append({"kind": kind, "name": name, "t": when})

    return {
        "t": time[:end].tolist(),
        "pos": position[:end].tolist(),
        "vel": velocity[:end].tolist(),
        "quat": quaternions,
        "wind": [0.0, 0.0, 0.0],
        "events": events,
    }
