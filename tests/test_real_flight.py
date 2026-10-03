"""Tests for turning a Blue Raven log into viewer telemetry."""

import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.real_flight import real_flight_telemetry


def _write_log(path: Path, altitude_ft: np.ndarray, range_ft: np.ndarray) -> None:
    n = len(altitude_ft)
    time = np.arange(n) * 0.5 - 0.5
    frame = pd.DataFrame(
        {
            "Flight_Time_(s)": time,
            "Inertial_Altitude": altitude_ft,
            "Inertial_DR_Position": range_ft,
            "Inertial_CR_position": np.zeros(n),
            "Velocity_Up": np.gradient(altitude_ft, time),
            "Velocity_DR": np.gradient(range_ft, time),
            "Velocity_CR": np.zeros(n),
            "Tilt_Angle_(deg)": np.zeros(n),
            "Roll_Angle_(deg)": np.zeros(n),
            "Burnout_Coast": [0] * 3 + [1] * (n - 3),
            "Apo_fired": np.zeros(n),
            "Main_fired": np.zeros(n),
        }
    )
    frame.to_csv(path, index=False)


def test_unwraps_positions_and_cuts_at_touchdown(tmp_path: Path) -> None:
    """A wrapped altitude is unwrapped and the log ends at touchdown."""
    height = np.array([0, 20000, 50000, 70000, 50000, 20000, -10, -500.0])
    wrapped = (height + 32768) % 65536 - 32768
    path = tmp_path / "log.csv"
    _write_log(path, wrapped, height * 0.1)
    telemetry = real_flight_telemetry(path, 270.0)
    up = [p[0] for p in telemetry["pos"]]  # type: ignore[attr-defined]
    assert max(up) == max(height) * 0.3048
    assert len(telemetry["t"]) == 6  # type: ignore[arg-type]
    assert telemetry["t"][0] == 0.0  # type: ignore[index]


def test_quaternions_are_unit_and_nose_up_at_zero_tilt(tmp_path: Path) -> None:
    """With no tilt the nose is vertical."""
    height = np.array([0, 100, 300, 500, 300, 100, -10.0])
    path = tmp_path / "log.csv"
    _write_log(path, height, height * 0.2)
    telemetry = real_flight_telemetry(path, 90.0)
    for w, x, y, z in telemetry["quat"]:  # type: ignore[attr-defined]
        assert math.isclose(w * w + x * x + y * y + z * z, 1.0)
        assert math.isclose(1 - 2 * (y * y + z * z), 1.0, abs_tol=1e-9)
