"""Builders shared across the test modules."""

from typing import Any

import numpy as np

from flight_sim.environment.launch_rail import NoRail
from flight_sim.integration import (
    IntegrationConfiguration,
    TruthConfiguration,
    derivative_computation,
)
from flight_sim.telemetry import Recorder
from flight_sim.units import scalar
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


def truth(**fields: Any) -> TruthConfiguration:
    """Return a sea-level equatorial site with no rail, changed by ``fields``."""
    site: dict[str, Any] = {
        "launch_latitude": scalar(0.0, "deg"),
        "launch_elevation": scalar(0.0, "m"),
        "launch_rail": NoRail(),
    }
    return TruthConfiguration(**{**site, **fields})


def recorded(
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
    time: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return every channel the equations of motion record at one state."""
    recorder = Recorder()
    with recorder.step():
        derivative_computation(time, state, properties, config)
    return {name: channel.values[0] for name, channel in recorder.channels().items()}
