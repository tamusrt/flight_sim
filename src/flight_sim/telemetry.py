"""Channels a run records, named where the physics computes them.

``record`` adds a value to the step a ``Recorder`` is recording, and does
nothing outside one, so trial states of the integrator are never recorded.
Values are recorded in SI units, or in the units they are recorded with.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np

from flight_sim.units import UnitSystem, parse_units, system_units, ureg
from flight_sim.utilities.quaternion import Quaternion, slerp

# How a channel is read between rows: blended linearly, taken from the nearer
# row, slerped as a (q_w, q_x, q_y, q_z) quaternion, or blended as degrees in
# [0, 360) the short way round
Interp: TypeAlias = Literal["linear", "nearest", "slerp", "wrapped"]


class Channel(NamedTuple):
    """One recorded quantity at every row."""

    values: np.ndarray  # Shape (rows, *value shape)
    units: str  # Pint units of the values; "" for dimensionless
    interp: Interp = "linear"

    def quantity(self, system: UnitSystem) -> Any:
        """Return the values as a quantity array in a unit system."""
        quantity = ureg.Quantity(self.values, parse_units(self.units))
        return quantity.to(parse_units(system_units(self.units, system)))


_RECORDER: ContextVar["Recorder | None"] = ContextVar("recorder", default=None)


def record(name: str, value: Any, units: str = "", interp: Interp = "linear") -> None:
    """Add a value to the step being recorded, if one is.

    Args:
        name (str): Channel name, such as "aero.force".
        value (Any): Number or array, in ``units``.
        units (str, optional): Pint units. Defaults to dimensionless.
        interp (Interp, optional): How the channel is read between rows.
            Defaults to linear.
    """
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.add(name, value, units, interp)


class Recorder:
    """Rows of what ``record`` was given within each ``step``."""

    def __init__(self) -> None:
        self._kinds: dict[str, tuple[str, Interp]] = {}
        self._rows: dict[str, list[np.ndarray]] = {}
        self._row: dict[str, np.ndarray] = {}

    @contextmanager
    def step(self) -> Iterator[None]:
        """Record what ``record`` is given within the block as one row.

        Raises:
            ValueError: If the row names other channels than the first row.
        """
        self._row = {}
        token = _RECORDER.set(self)
        try:
            yield
        finally:
            _RECORDER.reset(token)
        if self._rows and self._row.keys() != self._rows.keys():
            raise ValueError(
                f"Step recorded {sorted(self._row.keys() ^ self._rows.keys())}"
                " in only one of it and the first step"
            )
        for name, value in self._row.items():
            self._rows.setdefault(name, []).append(value)

    def add(self, name: str, value: Any, units: str, interp: Interp) -> None:
        """Add a value to the current row.

        Raises:
            ValueError: If the row has the channel already, or it was first
                recorded in other units or with another interpolation.
        """
        if name in self._row:
            raise ValueError(f"{name!r} recorded twice in one step")
        kind = self._kinds.setdefault(name, (units, interp))
        if kind != (units, interp):
            raise ValueError(f"{name!r} recorded as {(units, interp)}, first as {kind}")
        self._row[name] = np.array(value, dtype=float)

    def channels(self) -> dict[str, Channel]:
        """Return every channel recorded, in the order first recorded."""
        return {
            name: Channel(np.stack(rows), *self._kinds[name])
            for name, rows in self._rows.items()
        }


def _interpolate(channel: Channel, previous: np.ndarray, fraction: np.ndarray) -> Any:
    """Return a channel's values a fraction of the way from rows to the next."""
    start, end = channel.values[previous], channel.values[previous + 1]
    if channel.interp == "slerp":
        return np.array(
            [
                slerp(Quaternion(*a.tolist()), Quaternion(*b.tolist()), f).components()
                for a, b, f in zip(start, end, fraction.tolist(), strict=True)
            ]
        )
    if channel.interp == "wrapped":
        end = start + (end - start + 180.0) % 360.0 - 180.0
    fraction = fraction.reshape(-1, *[1] * (start.ndim - 1))
    if channel.interp == "nearest":
        return np.where(fraction < 0.5, start, end)
    blended = start * (1.0 - fraction) + end * fraction
    return blended % 360.0 if channel.interp == "wrapped" else blended


def resample(channels: Mapping[str, Channel], times: np.ndarray) -> dict[str, Channel]:
    """Read every channel at times within the "time" channel's.

    Args:
        channels (Mapping[str, Channel]): Recorded channels, with "time" in s.
        times (np.ndarray): Times to read them at, in s.

    Returns:
        dict[str, Channel]: One row per time, read as each channel's
            ``interp`` says.
    """
    recorded = channels["time"].values
    previous = np.clip(
        np.searchsorted(recorded, times, side="right") - 1, 0, len(recorded) - 2
    )
    span = recorded[previous + 1] - recorded[previous]
    fraction = np.divide(
        times - recorded[previous], span, out=np.zeros_like(times), where=span > 0.0
    )
    return {
        name: channel._replace(values=_interpolate(channel, previous, fraction))
        for name, channel in channels.items()
    }
