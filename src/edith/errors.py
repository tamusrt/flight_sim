"""Error raised for campaign files that cannot be read or describe no campaign."""

from flight_sim.config import ConfigError


class EdithConfigError(ConfigError):
    """Campaign file that cannot be read, or that describes no valid campaign."""
