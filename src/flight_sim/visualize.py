"""Log flight telemetry and write it into the 3D flight viewer."""

import json
import math
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flight_sim.descent import Parachute, RecoverySystem, ReefedParachute, air_at
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import CORD_TO_BODY_M, CORD_TO_NOSE_M
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_TEMPLATE = Path(__file__).with_name("viewer.html")
_HEAD = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    "<style>html,body{margin:0}</style></head><body>"
)

# Angle of attack at which the static centre of pressure is read, degrees
_CP_ALPHA_DEG = 4.0

# Width of the time window the propellant mass flow is measured over
_MASS_FLOW_STEP_S = 0.01


@dataclass(frozen=True)
class RecoveryFrame:
    """The recovery hardware at one sample, besides the rocket itself.

    Attributes:
        nose_dir (list[float] | None): After line stretch, the unit vector
            from the canopy attachment to the nose section.
        nose_sep (float): During ejection, the nose-to-body distance in m.
        line (list[float] | None): Unit vector along the line from the
            canopy to the rocket; None while there is no canopy.
        swing_deg (float): Angle between the line and the airflow.
        drag_fraction (float): Share of its drag the canopy gives.
    """

    nose_dir: list[float] | None = None
    nose_sep: float = 0.0
    line: list[float] | None = None
    swing_deg: float = 0.0
    drag_fraction: float = 1.0


def static_centre_of_pressure(properties: RocketProperties, mach: float) -> float:
    """Static centre of pressure in metres aft of the nose tip.

    The table's pitching moment about the nose is CMy = x_cp * Cz / L,
    read at a small angle of attack where Cz is not zero.

    Args:
        properties (RocketProperties): The rocket, with its aero table.
        mach (float): Mach number.

    Returns:
        float: Distance from the nose tip in metres, or NaN when the table gives
            no normal force there.
    """
    coefficients = properties.aero_table(mach, _CP_ALPHA_DEG, 0.0)
    if coefficients.cz == 0.0:
        return float("nan")
    length = float(properties.aero_table.reference_length.m_as("m"))
    return coefficients.cmy * length / coefficients.cz


def _json_safe(value: Any) -> Any:
    """Copy of ``value`` with every NaN or infinity replaced by None.

    ``json.dumps`` writes a NaN as the bare word ``NaN``, which is not valid
    JSON. None becomes ``null``, which the viewer reads as "unknown".
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def script_json(value: Any) -> str:
    """JSON text that is safe to put inside an HTML ``<script>`` element.

    A name such as ``</script>`` or ``<!--`` inside a JSON string would end the
    script (or hide the rest of the page) early. ``<\\/`` and ``\\u003c!--``
    mean the same characters to JSON and to JavaScript, so nothing else changes.
    """
    return escape_for_script(json.dumps(_json_safe(value), allow_nan=False))


def escape_for_script(text: str) -> str:
    """Make already-written JSON text safe inside an HTML ``<script>`` element."""
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


class TelemetryLog:
    """Collects one row per sample, in SI units, plus what the viewer draws.

    Besides the state, each row carries the Mach number, dynamic pressure,
    ambient pressure, propellant mass flow (kg/s),
    centre of gravity and static centre of pressure (both in metres aft of
    the nose tip). ``events``, ``rail`` and ``recovery`` describe the flight
    events, the launch rail and the recovery system for the 3D scene, and
    ``site`` where the pad is, for the satellite picture of the ground.
    """

    def __init__(self, properties: RocketProperties, config: IntegrationConfiguration):
        self._properties = properties
        self._config = config
        columns: tuple[str, ...] = ("t", "pos", "vel", "quat", "thrust", "mass", "mach")
        columns += ("q", "p", "mdot")
        columns += (
            "cg",
            "cp",
            "nose_dir",
            "nose_sep",
            "line",
            "swing",
            "drag",
            "wind_at",
        )
        self.rows: dict[str, list[Any]] = {k: [] for k in columns}
        self.wind = [0.0, 0.0, 0.0]
        self.events: list[dict[str, Any]] = []
        self.rail: dict[str, Any] | None = None
        self.recovery: dict[str, Any] | None = None
        self.vehicle: dict[str, Any] | None = None
        truth = config.truth
        self.site = {
            "lat_deg": float(truth.launch_latitude.m_as("deg")),
            "lon_deg": float(truth.launch_longitude.m_as("deg")),
            "elevation_m": float(truth.launch_elevation.m_as("m")),
        }

    def record(
        self,
        time_s: float,
        state: RocketState,
        frame: RecoveryFrame | None = None,
    ) -> None:
        """Append the state at ``time_s`` seconds after ignition.

        Args:
            time_s (float): Time since ignition in seconds.
            state (RocketState): The state.
            frame (RecoveryFrame | None): The recovery hardware to draw.
        """
        frame = frame or RecoveryFrame()
        pos = [float(x) for x in state.position.m_as("m")]
        velocity = [float(x) for x in state.velocity.m_as("m/s")]
        air, wind = air_at(self._config, pos[0])
        wind_row = [float(x) for x in wind]
        airspeed = math.dist(velocity, wind_row)
        mach = airspeed / air.speed_of_sound
        q = state.orientation
        thrust = float(self._properties.engine.get_thrust(time_s))
        mass_properties = self._properties.mass_properties(time_s)
        # The whole row is worked out first and added last, so a failure part
        # way through never leaves the columns with different lengths.
        row: dict[str, Any] = {
            "wind_at": [round(x, 3) for x in wind_row],
            "t": float(time_s),
            "pos": pos,
            "vel": velocity,
            "quat": [float(q.q_w), float(q.q_x), float(q.q_y), float(q.q_z)],
            "thrust": thrust,
            "mdot": self._mass_flow(time_s),
            "mass": mass_properties.mass,
            "mach": mach,
            "q": 0.5 * air.air_density * airspeed**2,
            "p": air.pressure,
            "nose_dir": frame.nose_dir or [0.0, 0.0, 0.0],
            "nose_sep": frame.nose_sep,
            "line": frame.line or [0.0, 0.0, 0.0],
            "swing": frame.swing_deg,
            "drag": frame.drag_fraction,
            "cg": -float(mass_properties.cg_location[0]),
            "cp": self._centre_of_pressure(mach),
        }
        if not self.rows["t"]:
            self.wind = wind_row
        for column, value in row.items():
            self.rows[column].append(value)

    def _mass_flow(self, time_s: float) -> float:
        """Propellant mass leaving the nozzle at a time, in kg/s (never negative)."""
        half = _MASS_FLOW_STEP_S / 2.0
        before = self._properties.mass_properties(time_s - half).mass
        after = self._properties.mass_properties(time_s + half).mass
        return max((before - after) / _MASS_FLOW_STEP_S, 0.0)

    def _centre_of_pressure(self, mach: float) -> float:
        return static_centre_of_pressure(self._properties, mach)

    def add_event(self, kind: str, name: str, time_s: float, **extra: float) -> None:
        """Note a flight event for the timeline, such as "rail exit"."""
        self.events.append({"kind": kind, "name": name, "t": float(time_s), **extra})

    def describe_rail(self, rail: LaunchRail) -> None:
        """Record the launch rail the scene draws."""
        self.rail = {
            "length": float(rail.length.m_as("m")),
            "direction": [float(x) for x in rail.direction()],
            "tilt_deg": 90.0 - float(rail.elevation.m_as("deg")),
        }

    def describe_recovery(self, recovery: RecoverySystem) -> None:
        """Record the canopy and shock cords the scene draws.

        The scene draws one canopy: the reefed one when there is one, otherwise
        the largest. A canopy that is not reefed is drawn fully open.
        """
        canopy: Parachute | ReefedParachute | None = next(
            (p for p in recovery.parachutes if isinstance(p, ReefedParachute)), None
        )
        if isinstance(canopy, ReefedParachute):
            reefed_opening = canopy.reefed_opening_diameter_m
        elif recovery.parachutes:
            canopy = max(recovery.parachutes, key=lambda p: p.diameter_m)
            reefed_opening = canopy.diameter_m
        else:
            return
        self.recovery = {
            "diameter": canopy.diameter_m,
            "spill_hole": canopy.spill_hole_diameter_m,
            "reefed_opening": reefed_opening,
            "cord_to_nose": CORD_TO_NOSE_M,
            "cord_to_body": CORD_TO_BODY_M,
        }

    def describe_vehicle(self, ork: Any) -> None:
        """Record the outside of the rocket from its OpenRocket design.

        The scene then draws the rocket that is flying (its parts, materials, fins
        and length) instead of its built-in model.
        """
        fins = ork.fins
        self.vehicle = {
            "length": float(ork.length_m),
            "radius": 0.5 * float(ork.reference_diameter_m),
            "segments": [dict(s) for s in ork.segments],
            "fins": {
                "n": int(fins.count),
                "root": float(fins.root_chord_m),
                "tip": float(fins.tip_chord_m),
                "span": float(fins.span_m),
                "sweep": float(fins.sweep_m),
                "t": float(fins.thickness_m),
                "xLE": float(fins.root_trailing_edge_x_m - fins.root_chord_m),
            },
        }

    def to_json(self) -> str:
        """Telemetry in the format the viewer expects.

        A NaN or infinite value (for example a centre of pressure the aero table
        cannot give) is written as ``null``, which the viewer reads as unknown.
        """
        return json.dumps(
            _json_safe(
                {
                    **self.rows,
                    "wind": self.wind,
                    "events": self.events,
                    "rail": self.rail,
                    "recovery": self.recovery,
                    "vehicle": self.vehicle,
                    "site": self.site,
                }
            ),
            allow_nan=False,
        )


def write_viewer(
    log: TelemetryLog,
    output: str | Path = "flight.html",
    open_browser: bool = True,
    real: dict[str, Any] | None = None,
    openrocket: dict[str, str] | None = None,
) -> Path:
    """Write a self-contained viewer page for the logged flight and open it.

    Args:
        log (TelemetryLog): The simulated flight.
        output (str | Path): Where to write the page.
        open_browser (bool): Whether to open the page.
        real (dict[str, Any] | None): Telemetry of the real flight; when given
            the viewer gets a button that toggles between it and the sim.
        openrocket (dict[str, str] | None): ``{"ork": file name, "sim": name}``
            of the OpenRocket simulation to offer next to the flight; on the
            dynamics site the viewer loads its path from the History tab's data.
    """
    # Names in the data (events, OpenRocket parts, the .ork file) must not be able
    # to end the script early, so each blob goes through the script-safe escape.
    scripts = (
        f"<script>window.TELEMETRY={escape_for_script(log.to_json())};"
        "window.TELEMETRY_NAME='simulation';</script>"
    )
    if real is not None:
        scripts += (
            f"<script>window.REAL_TELEMETRY={script_json(real)};"
            "window.REAL_TELEMETRY_NAME='real flight';</script>"
        )
    if openrocket is not None:
        scripts += (
            f"<script>window.OPENROCKET_SOURCE={script_json(openrocket)};</script>"
        )
    page = _TEMPLATE.read_text(encoding="utf-8").replace("<!--TELEMETRY-->", scripts)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_HEAD + page + "</body></html>", encoding="utf-8")
    if open_browser:
        webbrowser.open(output.as_uri())
    return output
