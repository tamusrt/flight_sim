"""Log flight telemetry and write it into the 3D flight viewer."""

import json
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
    """Collects one row per accepted integrator step, in SI units."""

    def __init__(self, properties: RocketProperties, config: IntegrationConfiguration):
        self._properties = properties
        self._config = config
        self.rows: dict[str, list[Any]] = {
            k: [] for k in ("t", "pos", "vel", "quat", "thrust", "mass")
        }
        self.wind = [0.0, 0.0, 0.0]

    def record(self, time_s: float, state: RocketState) -> None:
        """Append the state at ``time_s`` seconds after ignition."""
        pos = [float(x) for x in state.position.m_as("m")]
        q = state.orientation
        if not self.rows["t"]:
            self.wind = [
                float(x) for x in self._config.atmosphere.conditions(pos[0]).wind
            ]
        self.rows["t"].append(time_s)
        self.rows["pos"].append(pos)
        self.rows["vel"].append([float(x) for x in state.velocity.m_as("m/s")])
        self.rows["quat"].append([q.q_w, q.q_x, q.q_y, q.q_z])
        self.rows["thrust"].append(float(self._properties.engine.get_thrust(time_s)))
        self.rows["mass"].append(float(state.current_mass.m_as("kg")))

    def to_json(self) -> str:
        """Telemetry in the format the viewer expects."""
        return json.dumps({**self.rows, "wind": self.wind})


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
