"""Log flight telemetry and write it into the 3D flight viewer."""

import json
import math
import webbrowser
from pathlib import Path
from typing import Any

from flight_sim.integration import IntegrationConfiguration
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_TEMPLATE = Path(__file__).with_name("viewer.html")
_HEAD = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    "<style>html,body{margin:0}</style></head><body>"
)


class TelemetryLog:
    """Collects one row per sample, in SI units, for the viewer."""

    def __init__(self, properties: RocketProperties, config: IntegrationConfiguration):
        self._properties = properties
        self._config = config
        self.rows: dict[str, list[Any]] = {
            k: [] for k in ("t", "pos", "vel", "quat", "thrust", "mass", "mach")
        }
        self.wind = [0.0, 0.0, 0.0]
        # Named moments, such as parachute releases, as (name, time in s)
        self.events: list[tuple[str, float]] = []

    def record(self, time_s: float, state: RocketState) -> None:
        """Append the state at ``time_s`` seconds after ignition."""
        pos = [float(x) for x in state.position.m_as("m")]
        velocity = [float(x) for x in state.velocity.m_as("m/s")]
        air = self._config.atmosphere.conditions(pos[0])
        if not self.rows["t"]:
            self.wind = [float(x) for x in air.wind]
        q = state.orientation
        airspeed = math.dist(velocity, [float(x) for x in air.wind])
        self.rows["t"].append(float(time_s))
        self.rows["pos"].append(pos)
        self.rows["vel"].append(velocity)
        self.rows["quat"].append(
            [float(q.q_w), float(q.q_x), float(q.q_y), float(q.q_z)]
        )
        self.rows["thrust"].append(float(self._properties.engine.get_thrust(time_s)))
        self.rows["mass"].append(float(state.current_mass.m_as("kg")))
        self.rows["mach"].append(airspeed / air.speed_of_sound)

    def to_json(self) -> str:
        """Telemetry in the format the viewer expects."""
        return json.dumps(
            {
                **self.rows,
                "wind": self.wind,
                "events": [[name, float(time)] for name, time in self.events],
            }
        )


def write_viewer(
    log: TelemetryLog, output: str | Path = "flight.html", open_browser: bool = True
) -> Path:
    """Write a self-contained viewer page for the logged flight and open it."""
    page = _TEMPLATE.read_text(encoding="utf-8").replace(
        "<!--TELEMETRY-->",
        f"<script>window.TELEMETRY={log.to_json()};window.TELEMETRY_NAME='simulation';</script>",
    )
    output = Path(output).resolve()
    output.write_text(_HEAD + page + "</body></html>", encoding="utf-8")
    if open_browser:
        webbrowser.open(output.as_uri())
    return output
