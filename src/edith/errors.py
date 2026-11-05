"""Error raised for campaign files that cannot be read or describe no campaign."""

import difflib
from collections.abc import Sequence

from flight_sim.config import ConfigError


class EdithConfigError(ConfigError):
    """Campaign file that cannot be read, or that describes no valid campaign."""


def closest(key: str, known: Sequence[str]) -> str | None:
    """Return the known key closest to a misspelt one, preferring a case match."""
    same = [option for option in known if option.lower() == key.lower()]
    close = same or difflib.get_close_matches(key, known, n=1)
    return close[0] if close else None
