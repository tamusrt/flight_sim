"""Typed views, named channels and tables of the flight points a run records.

A channel is named by its attribute path in ``FlightPoint``, such as
"aero.force" or "state.orientation.q_w", and carries the units the field
declares. Quantities that are no field of it are under "derived.".
"""

import dataclasses
import fnmatch
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import NoneType
from typing import (
    Annotated,
    Any,
    NamedTuple,
    Self,
    get_args,
    get_type_hints,
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.flight_point import FlightPointSI
from flight_sim.integration import StateDerivative
from flight_sim.units import (
    Scalar,
    UnitChecked,
    Vector,
    annotated_units,
    parse_units,
    ureg,
)
from flight_sim.utilities.data_loader import AeroCoefficients
from flight_sim.utilities.dcm import body_to_world
from flight_sim.utilities.quaternion import Quaternion, slerp
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_state import RocketState


@dataclass(frozen=True)
class EnvironmentPoint(UnitChecked):
    """Air, wind and gravity around the rocket."""

    altitude: Annotated[Scalar, "m"]  # Above sea level
    atmosphere: AtmosphereConditions
    wind: Annotated[Vector, "m/s"]  # World components
    gravity: Annotated[Scalar, "m/s**2"]  # Acting along world -X


@dataclass(frozen=True)
class AeroPoint(UnitChecked):
    """Flow angles, coefficients and loads of the air on the rocket."""

    air_velocity: Annotated[Vector, "m/s"]  # Relative to the wind, body axes
    airspeed: Annotated[Scalar, "m/s"]
    mach: float
    alpha: Annotated[Scalar, "rad"]  # Total angle of attack
    phi: Annotated[Scalar, "rad"]  # Aerodynamic roll angle
    dynamic_pressure: Annotated[Scalar, "Pa"]
    coefficients: AeroCoefficients
    force: Annotated[Vector, "N"]  # Body axes
    torque: Annotated[Vector, "N*m"]  # About the CG, body axes


@dataclass(frozen=True)
class FlightPoint(UnitChecked):
    """State at one moment, with its rates and every load behind them."""

    time: Annotated[Scalar, "s"]
    state: RocketState
    derivative: StateDerivative
    environment: EnvironmentPoint
    aero: AeroPoint
    thrust: Annotated[Scalar, "N"]  # Along body +X
    mass: MassProperties

    # Non-gravitational force per unit mass, as an accelerometer reads it,
    # in body axes
    specific_force: Annotated[Vector, "m/s**2"]

    @classmethod
    def from_si(cls, point: FlightPointSI) -> Self:
        """Build the typed view of a recorded point."""
        result: Self = _typed(cls, point)
        return result


def _field_names(cls: type) -> list[str]:
    """Return the constructor fields of a dataclass or NamedTuple, in order."""
    if dataclasses.is_dataclass(cls):
        return [field.name for field in dataclasses.fields(cls) if field.init]
    return list(getattr(cls, "_fields", ()))


def _typed(cls: type, si: Any) -> Any:
    """Build a typed class from its SI mirror, field by field of the same name."""
    hints = get_type_hints(cls, include_extras=True)
    arguments = {}
    for name in _field_names(cls):
        value, hint = getattr(si, name), hints[name]
        units = annotated_units(hint)
        if value is None or isinstance(value, hint if isinstance(hint, type) else ()):
            arguments[name] = value
        elif units is not None:
            arguments[name] = ureg.Quantity(value, parse_units(units))
        elif _field_names(hint):
            arguments[name] = _typed(hint, value)
        else:
            arguments[name] = value
    return cls(**arguments)


class Channel(NamedTuple):
    """One recorded quantity: its units and how to read it from a point."""

    units: str  # Pint units of what ``read`` returns; "" for dimensionless
    read: Callable[[FlightPointSI], Any]

    # Reads the channel from the orientation alone; resampling reads it off
    # the slerped orientation
    of_orientation: Callable[[Quaternion], float] | None = None

    # Between points, resampling takes the nearer point's value, not a blend
    nearest: bool = False


def _orientation_channel(units: str, read: Callable[[Quaternion], float]) -> Channel:
    """Return a channel read from the orientation alone."""
    return Channel(
        units, lambda point: read(point.state.orientation), of_orientation=read
    )


# Quaternion components are dimensionless floats, which a rate in 1/s is not
_SKIPPED = ("derivative.orientation_derivative",)


def _leaves(cls: type, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield the path and units of every numeric field under a typed class.

    Fields that may be None, such as the rail start, are left out.
    """
    hints = get_type_hints(cls, include_extras=True)
    for name in _field_names(cls):
        path, hint = prefix + name, hints[name]
        if path in _SKIPPED:
            continue
        units = annotated_units(hint)
        if units is not None:
            if NoneType not in get_args(get_args(hint)[0]):
                yield path, units
        elif hint in (float, int):
            yield path, ""
        elif isinstance(hint, type):
            yield from _leaves(hint, f"{path}.")


def _norm(values: np.ndarray) -> float:
    """Return the length of a vector."""
    return math.sqrt(float(values @ values))


def _nose(orientation: Quaternion) -> np.ndarray:
    """Return the world-frame unit vector out the nose."""
    to_world: np.ndarray = body_to_world(orientation)
    return to_world[:, 0]


def _nose_elevation(orientation: Quaternion) -> float:
    """Return the nose's angle above the horizon in degrees."""
    return math.degrees(math.asin(min(max(float(_nose(orientation)[0]), -1.0), 1.0)))


def _nose_azimuth(orientation: Quaternion) -> float:
    """Return the nose's heading clockwise from north in degrees, in [0, 360)."""
    _, east, north = _nose(orientation).tolist()
    return math.degrees(math.atan2(east, north)) % 360.0


# Every channel a flight records, by name
CHANNELS: dict[str, Channel] = {
    **{
        path: Channel(units, operator.attrgetter(path))
        for path, units in _leaves(FlightPoint)
    },
    **{
        f"state.orientation.{name}": _orientation_channel("", operator.attrgetter(name))
        for name in _field_names(Quaternion)
    },
    "derived.speed": Channel("m/s", lambda p: _norm(p.state.velocity)),
    "derived.nose_elevation": _orientation_channel("deg", _nose_elevation),
    "derived.nose_azimuth": _orientation_channel("deg", _nose_azimuth),
    "derived.on_rail": Channel(
        "", lambda p: float(p.state.rail_start is not None), nearest=True
    ),
}


def channel_names(patterns: Sequence[str] = ("*",)) -> list[str]:
    """Return the channels matching any of some glob patterns, in table order.

    Raises:
        KeyError: If a pattern matches no channel.
    """
    for pattern in patterns:
        if not fnmatch.filter(CHANNELS, pattern):
            raise KeyError(f"No channel matches {pattern!r}")
    return [
        name
        for name in CHANNELS
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]


def column(points: Sequence[FlightPointSI], name: str) -> Any:
    """Read one channel at every point.

    Args:
        points (Sequence[FlightPointSI]): Recorded points.
        name (str): Channel name, a key of ``CHANNELS``.

    Returns:
        Quantity: Shape (points,) or (points, *field shape), in the channel's
            units.

    Raises:
        KeyError: If no channel has that name.
    """
    if name not in CHANNELS:
        raise KeyError(f"No channel {name!r}; see telemetry.CHANNELS")
    channel = CHANNELS[name]
    values = np.array([channel.read(point) for point in points], dtype=float)
    return ureg.Quantity(values, parse_units(channel.units))


class _Resampling(NamedTuple):
    """Where the rows of a resampled table fall among the recorded points."""

    times: np.ndarray  # Of the rows, in seconds
    previous: np.ndarray  # Recorded point at or before each row
    fraction: np.ndarray  # Each row's way from that point to the next, 0 to 1
    orientations: list[Quaternion]  # Slerped orientation at each row
    rows: dict[str, int]  # Row of each event


def _resampling(
    points: Sequence[FlightPointSI], events: Mapping[str, int], interval_s: float
) -> _Resampling:
    """Place rows every interval from launch, plus one at each event and the end."""
    times = np.array([point.time for point in points])
    grid = np.union1d(
        np.arange(times[0], times[-1], interval_s),
        np.append(times[list(events.values())], times[-1]),
    )
    previous = np.clip(
        np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2
    )
    span = times[previous + 1] - times[previous]
    fraction = np.divide(
        grid - times[previous], span, out=np.zeros_like(grid), where=span > 0.0
    )
    orientations = [
        slerp(points[i].state.orientation, points[i + 1].state.orientation, f)
        for i, f in zip(previous.tolist(), fraction.tolist(), strict=True)
    ]
    rows = {
        name: int(np.searchsorted(grid, times[row])) for name, row in events.items()
    }
    return _Resampling(grid, previous, fraction, orientations, rows)


def _resample(channel: Channel, values: np.ndarray, at: _Resampling) -> np.ndarray:
    """Return a channel's recorded values at the rows of a resampled table."""
    if channel.of_orientation is not None:
        return np.array([channel.of_orientation(q) for q in at.orientations])
    start, end = values[at.previous], values[at.previous + 1]
    fraction = at.fraction.reshape(-1, *[1] * (values.ndim - 1))
    resampled: np.ndarray = (
        np.where(fraction < 0.5, start, end)
        if channel.nearest
        else start * (1.0 - fraction) + end * fraction
    )
    return resampled


def _component_names(shape: tuple[int, ...]) -> list[str]:
    """Return column suffixes for the components of an array field."""
    if shape == (3,):
        return ["x", "y", "z"]
    if shape == (3, 3):
        return [row + col for row in "xyz" for col in "xyz"]
    return ["_".join(map(str, index)) for index in np.ndindex(shape)]


def table(
    points: Sequence[FlightPointSI],
    events: Mapping[str, int],
    patterns: Sequence[str] = ("*",),
    interval_s: float | None = None,
) -> pd.DataFrame:
    """Lay recorded points out as one column per channel component.

    Args:
        points (Sequence[FlightPointSI]): Recorded points.
        events (Mapping[str, int]): Row of each event, which fills the
            "event" column.
        patterns (Sequence[str], optional): Glob patterns of the channels to
            include. Defaults to every channel.
        interval_s (float | None, optional): Spacing of the rows in seconds,
            with a row kept at each event and at the end. Between points,
            orientations are slerped and other values blended linearly.
            Defaults to one row per point.

    Returns:
        pd.DataFrame: Columns such as "aero.force_x [N]", plus "event".
    """
    resampling = (
        _resampling(points, events, interval_s) if interval_s is not None else None
    )
    data: dict[str, Any] = {}
    for name in channel_names(patterns):
        values = column(points, name).magnitude
        if resampling is not None:
            values = _resample(CHANNELS[name], values, resampling)
        units = CHANNELS[name].units or "-"
        if values.ndim == 1:
            data[f"{name} [{units}]"] = values
            continue
        flat = values.reshape(len(values), -1)
        for index, suffix in enumerate(_component_names(values.shape[1:])):
            data[f"{name}_{suffix} [{units}]"] = flat[:, index]
    rows = events if resampling is None else resampling.rows
    labels = [""] * (len(points) if resampling is None else len(resampling.times))
    for name, row in rows.items():
        labels[row] = f"{labels[row]}, {name}" if labels[row] else name
    data["event"] = labels
    return pd.DataFrame(data)


def _lerp(start: Any, end: Any, fraction: float) -> Any:
    """Interpolate numbers, arrays, and tuples and dataclasses of them, linearly.

    Values that cannot be interpolated, such as None, take the nearer end.
    """
    if start is None or end is None or isinstance(start, bool):
        return start if fraction < 0.5 else end
    if isinstance(start, tuple) or dataclasses.is_dataclass(start):
        cls: Any = type(start)
        return cls(
            *(
                _lerp(getattr(start, name), getattr(end, name), fraction)
                for name in _field_names(cls)
            )
        )
    return start + fraction * (end - start)


def interpolate(
    start: FlightPointSI, end: FlightPointSI, fraction: float
) -> FlightPointSI:
    """Interpolate between two recorded points, by slerp for the orientation.

    Args:
        start (FlightPointSI): Point at ``fraction`` 0.
        end (FlightPointSI): Point at ``fraction`` 1.
        fraction (float): Position between them, from 0 to 1.

    Returns:
        FlightPointSI: Every quantity interpolated linearly, except the
            orientation.
    """
    point: FlightPointSI = _lerp(start, end, fraction)
    orientation = slerp(start.state.orientation, end.state.orientation, fraction)
    return point._replace(state=point.state._replace(orientation=orientation))
