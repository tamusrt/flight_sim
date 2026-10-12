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
from typing import (
    Annotated,
    Any,
    NamedTuple,
    Self,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.flight_point import FlightPointSI
from flight_sim.integration import StateDerivative
from flight_sim.units import Scalar, UnitChecked, Vector, parse_units, ureg
from flight_sim.utilities.data_loader import AeroCoefficients
from flight_sim.utilities.dcm import body_to_world
from flight_sim.utilities.quaternion import Quaternion
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


def _units(hint: Any) -> str | None:
    """Return the unit string of an ``Annotated`` hint, or None without one."""
    if get_origin(hint) is not Annotated:
        return None
    return next((meta for meta in get_args(hint)[1:] if isinstance(meta, str)), None)


def _typed(cls: type, si: Any) -> Any:
    """Build a typed class from its SI mirror, field by field of the same name."""
    hints = get_type_hints(cls, include_extras=True)
    arguments = {}
    for name in _field_names(cls):
        value, hint = getattr(si, name), hints[name]
        units = _units(hint)
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
        units = _units(hint)
        if units is not None:
            if type(None) not in get_args(get_args(hint)[0]):
                yield path, units
        elif hint in (float, int):
            yield path, ""
        elif isinstance(hint, type):
            yield from _leaves(hint, f"{path}.")


def _norm(values: np.ndarray) -> float:
    """Return the length of a vector."""
    return math.sqrt(float(values @ values))


def _nose(point: FlightPointSI) -> np.ndarray:
    """Return the world-frame unit vector out the nose."""
    to_world: np.ndarray = body_to_world(point.state.orientation)
    return to_world[:, 0]


def _nose_elevation(point: FlightPointSI) -> float:
    """Return the nose's angle above the horizon in degrees."""
    return math.degrees(math.asin(min(max(float(_nose(point)[0]), -1.0), 1.0)))


def _nose_azimuth(point: FlightPointSI) -> float:
    """Return the nose's heading clockwise from north in degrees, in [0, 360)."""
    _, east, north = _nose(point).tolist()
    return math.degrees(math.atan2(east, north)) % 360.0


# Every channel a flight records, by name
CHANNELS: dict[str, Channel] = {
    **{
        path: Channel(units, operator.attrgetter(path))
        for path, units in _leaves(FlightPoint)
    },
    "derived.speed": Channel("m/s", lambda p: _norm(p.state.velocity)),
    "derived.nose_elevation": Channel("deg", _nose_elevation),
    "derived.nose_azimuth": Channel("deg", _nose_azimuth),
    "derived.on_rail": Channel("", lambda p: float(p.state.rail_start is not None)),
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
) -> pd.DataFrame:
    """Lay recorded points out as one column per channel component.

    Args:
        points (Sequence[FlightPointSI]): Recorded points.
        events (Mapping[str, int]): Row of each event, which fills the
            "event" column.
        patterns (Sequence[str], optional): Glob patterns of the channels to
            include. Defaults to every channel.

    Returns:
        pd.DataFrame: Columns such as "aero.force_x [N]", plus "event".
    """
    data: dict[str, Any] = {}
    for name in channel_names(patterns):
        values = column(points, name).magnitude
        units = CHANNELS[name].units or "-"
        if values.ndim == 1:
            data[f"{name} [{units}]"] = values
            continue
        flat = values.reshape(len(points), -1)
        for index, suffix in enumerate(_component_names(values.shape[1:])):
            data[f"{name}_{suffix} [{units}]"] = flat[:, index]
    labels = [""] * len(points)
    for name, row in events.items():
        labels[row] = f"{labels[row]}, {name}" if labels[row] else name
    data["event"] = labels
    return pd.DataFrame(data)


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate between two unit quaternions along the shorter arc.

    Args:
        start (np.ndarray): Quaternion at ``fraction`` 0.
        end (np.ndarray): Quaternion at ``fraction`` 1.
        fraction (float): Position between them, from 0 to 1.

    Returns:
        np.ndarray: Unit quaternion in the same component order.
    """
    dot = float(start @ end)
    if dot < 0.0:
        end, dot = -end, -dot
    if dot > 0.9995:
        # Nearly parallel: a normalized straight line is accurate and stable
        result: np.ndarray = start + fraction * (end - start)
        return result / math.sqrt(float(result @ result))
    angle = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * angle) * start + math.sin(fraction * angle) * end
    ) / math.sin(angle)


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


def _components(orientation: Quaternion) -> np.ndarray:
    """Return a quaternion's (q_w, q_x, q_y, q_z) as an array."""
    return np.array(
        [orientation.q_w, orientation.q_x, orientation.q_y, orientation.q_z]
    )


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
    orientation = slerp(
        _components(start.state.orientation),
        _components(end.state.orientation),
        fraction,
    )
    return point._replace(
        state=point.state._replace(orientation=Quaternion(*orientation.tolist()))
    )
