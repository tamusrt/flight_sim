"""A cheap stand-in for the 6-DOF climb, trained on a few full climbs.

The climb is the costly part of a run once the descent is fast, and its end
(where and when apogee comes, and the rail exit numbers) changes smoothly with
the inputs. A radial basis function surrogate (thin-plate spline with a linear
trend) is fitted to a pilot set of full climbs and then predicts the apogee
state, the apogee time and the rail exit speed and stability for any other
draw. The descent is then flown from the predicted apogee by FastRECO, as for a
full climb.

Nine numbers describe a draw for the climb: thrust, dry mass, drag, the rail's
tilt and heading, the pad temperature and pressure, and the mean wind (east
and north) over the climb, which stands in for the base wind, its direction
and all the layers at once.

How good the guess is is measured, not assumed: ``cross_validated_error`` holds
out a part of the pilot set at a time, and ``EDITH`` re-flies a random share of
the other runs with the full climb to audit it (``batch``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator

from flight_sim.edith.inputs import Nominal, Variation

FEATURES = 9
TARGETS = (
    "apogee_m",
    "apogee_east",
    "apogee_north",
    "apogee_t",
    "vx",
    "vy",
    "vz",
    "rail_v",
    "rail_margin",
    "margin_lo",
    "margin_hi",
)
_SMOOTHING = (1e-4, 1e-2, 1.0)
_FOLDS = 8


def features(v: Variation, nominal: Nominal, top_m: float) -> np.ndarray:
    """The nine numbers of a draw that the climb depends on."""
    wind = v.mean_wind(nominal.pad_elevation_m, top_m)
    return np.array(
        [
            v.thrust,
            v.mass,
            v.drag,
            v.rail_elevation_deg,
            v.rail_azimuth_deg,
            v.temperature_k,
            v.pressure_pa,
            wind[0],
            wind[1],
        ]
    )


def _targets(payload: dict[str, Any]) -> list[float] | None:
    """A pilot climb's numbers, in the order of ``TARGETS`` (None if one is missing)."""
    values = [
        payload["position"][0],
        payload["position"][1],
        payload["position"][2],
        payload["time"],
        payload["velocity"][0],
        payload["velocity"][1],
        payload["velocity"][2],
        payload["rail_v"],
        payload["rail_margin"],
        payload["margin_lo"],
        payload["margin_hi"],
    ]
    if any(v is None or not np.isfinite(v) for v in values):
        return None
    return [float(v) for v in values]


class Surrogate:
    """The fitted climb model."""

    def __init__(self, x: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        rows, y = [], []
        self.payloads: list[dict[str, Any]] = []
        for row, payload in zip(x, payloads, strict=True):
            targets = _targets(payload)
            if targets is not None:
                rows.append(row)
                y.append(targets)
                self.payloads.append(payload)
        if len(rows) < FEATURES + 2:
            raise ValueError("Too few climbs to fit the surrogate")
        self.x = np.array(rows)
        self.y = np.array(y)
        self.x_mean, self.x_std = self.x.mean(axis=0), self.x.std(axis=0) + 1e-12
        self.y_mean, self.y_std = self.y.mean(axis=0), self.y.std(axis=0) + 1e-12
        self.smoothing, self.error = self._select()
        self.model = self._fit(self.x, self.y, self.smoothing)

    def _fit(self, x: np.ndarray, y: np.ndarray, smoothing: float) -> RBFInterpolator:
        return RBFInterpolator(
            (x - self.x_mean) / self.x_std,
            (y - self.y_mean) / self.y_std,
            kernel="thin_plate_spline",
            degree=1,
            smoothing=smoothing,
        )

    def _select(self) -> tuple[float, dict[str, float]]:
        """The smoothing that predicts held-out climbs best, and its errors."""
        order = np.random.default_rng(0).permutation(len(self.x))
        best_score, best_smoothing = math.inf, _SMOOTHING[0]
        best_errors = np.zeros_like(self.y)
        for smoothing in _SMOOTHING:
            errors = np.zeros_like(self.y)
            for fold in range(_FOLDS):
                held = order[fold::_FOLDS]
                keep = np.setdiff1d(order, held)
                model = self._fit(self.x[keep], self.y[keep], smoothing)
                guess = model((self.x[held] - self.x_mean) / self.x_std)
                errors[held] = guess * self.y_std + self.y_mean - self.y[held]
            score = float(np.mean((errors / self.y_std) ** 2))
            if score < best_score:
                best_score, best_smoothing, best_errors = score, smoothing, errors
        rmse = np.sqrt(np.mean(best_errors**2, axis=0))
        return best_smoothing, {
            name: float(e) for name, e in zip(TARGETS, rmse, strict=True)
        }

    def predict(self, row: np.ndarray) -> dict[str, Any]:
        """The climb's end for one draw, as ``run.Rocket.fly`` takes it."""
        z = (row - self.x_mean) / self.x_std
        out = self.model(z[None, :])[0] * self.y_std + self.y_mean
        values = dict(zip(TARGETS, (float(v) for v in out), strict=True))
        nearest = self.payloads[
            int(
                np.argmin(
                    np.linalg.norm((self.x - self.x_mean) / self.x_std - z, axis=1)
                )
            )
        ]
        return {
            "time": values["apogee_t"],
            "position": [
                values["apogee_m"],
                values["apogee_east"],
                values["apogee_north"],
            ],
            "velocity": [0.0, values["vy"], values["vz"]],
            "quaternion": nearest["quaternion"],
            "rates": nearest["rates"],
            "rail_v": values["rail_v"],
            "rail_margin": values["rail_margin"],
            "margin_lo": values["margin_lo"],
            "margin_hi": values["margin_hi"],
        }

    def check_tolerances(self) -> dict[str, float]:
        """How far the surrogate can be off: twice its held-out error."""
        return {
            "rail_v": 2.0 * self.error["rail_v"],
            "rail_margin": 2.0 * self.error["rail_margin"],
            "margin_lo": 2.0 * self.error["margin_lo"],
            "margin_hi": 2.0 * self.error["margin_hi"],
        }
