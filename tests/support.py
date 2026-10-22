"""Builders shared across the test modules."""

from typing import Any

from flight_sim.environment.launch_rail import NoRail
from flight_sim.integration import TruthConfiguration
from flight_sim.units import scalar


def truth(**fields: Any) -> TruthConfiguration:
    """Return a sea-level equatorial site with no rail, changed by ``fields``."""
    site: dict[str, Any] = {
        "launch_latitude": scalar(0.0, "deg"),
        "launch_elevation": scalar(0.0, "m"),
        "launch_rail": NoRail(),
    }
    return TruthConfiguration(**{**site, **fields})
