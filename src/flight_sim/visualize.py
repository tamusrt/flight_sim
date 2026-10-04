"""Log flight telemetry and write it into the 3D flight viewer."""

import json
import math
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flight_sim.descent import RecoverySystem, ReefedParachute, air_at
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import CORD_TO_BODY_M, CORD_TO_NOSE_M
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_TEMPLATE = Path(__file__).with_name("viewer.html")
_HEAD = (
    '<!doctype html><html><head><meta charset="utf-8">'
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


class TelemetryLog:
    """Collects one row per sample, in SI units, plus what the viewer draws.

    Besides the state, each row carries the Mach number, dynamic pressure,
    ambient pressure, propellant mass flow (kg/s),
    centre of gravity and static centre of pressure (both in metres aft of
    the nose tip). ``events``, ``rail`` and ``recovery`` describe the flight
    events, the launch rail and the recovery system for the 3D scene.
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
        if not self.rows["t"]:
            self.wind = [float(x) for x in wind]
        self.rows["wind_at"].append([round(float(x), 3) for x in wind])
        airspeed = math.dist(velocity, [float(x) for x in wind])
        mach = airspeed / air.speed_of_sound
        q = state.orientation
        self.rows["t"].append(float(time_s))
        self.rows["pos"].append(pos)
        self.rows["vel"].append(velocity)
        self.rows["quat"].append(
            [float(q.q_w), float(q.q_x), float(q.q_y), float(q.q_z)]
        )
        thrust = float(self._properties.engine.get_thrust(time_s))
        self.rows["thrust"].append(thrust)
        mass_properties = self._properties.mass_properties(time_s)
        self.rows["mdot"].append(self._mass_flow(time_s))
        self.rows["mass"].append(mass_properties.mass)
        self.rows["mach"].append(mach)
        self.rows["q"].append(0.5 * air.air_density * airspeed**2)
        self.rows["p"].append(air.pressure)
        self.rows["nose_dir"].append(frame.nose_dir or [0.0, 0.0, 0.0])
        self.rows["nose_sep"].append(frame.nose_sep)
        self.rows["line"].append(frame.line or [0.0, 0.0, 0.0])
        self.rows["swing"].append(frame.swing_deg)
        self.rows["drag"].append(frame.drag_fraction)
        self.rows["cg"].append(-float(mass_properties.cg_location[0]))
        self.rows["cp"].append(self._centre_of_pressure(mach))

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
        """Record the canopy and shock cords the scene draws."""
        canopy = next(
            (p for p in recovery.parachutes if isinstance(p, ReefedParachute)), None
        )
        if canopy is None:
            return
        self.recovery = {
            "diameter": canopy.diameter_m,
            "spill_hole": canopy.spill_hole_diameter_m,
            "reefed_opening": canopy.reefed_opening_diameter_m,
            "cord_to_nose": CORD_TO_NOSE_M,
            "cord_to_body": CORD_TO_BODY_M,
        }

    def to_json(self) -> str:
        """Telemetry in the format the viewer expects."""
        cp = [None if math.isnan(x) else x for x in self.rows["cp"]]
        return json.dumps(
            {
                **self.rows,
                "cp": cp,
                "wind": self.wind,
                "events": self.events,
                "rail": self.rail,
                "recovery": self.recovery,
            }
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
    scripts = (
        f"<script>window.TELEMETRY={log.to_json()};"
        "window.TELEMETRY_NAME='simulation';</script>"
    )
    if real is not None:
        scripts += (
            f"<script>window.REAL_TELEMETRY={json.dumps(real)};"
            "window.REAL_TELEMETRY_NAME='real flight';</script>"
        )
    if openrocket is not None:
        source = json.dumps(openrocket)
        scripts += f"<script>window.OPENROCKET_SOURCE={source};</script>"
    page = _TEMPLATE.read_text(encoding="utf-8").replace("<!--TELEMETRY-->", scripts)
    output = Path(output).resolve()
    output.write_text(_HEAD + page + "</body></html>", encoding="utf-8")
    if open_browser:
        webbrowser.open(output.as_uri())
    return output
