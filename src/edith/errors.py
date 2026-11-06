"""Error raised for invalid campaign files, with checks that raise it."""

import difflib
from collections.abc import Mapping, Sequence

from flight_sim.config import ConfigError
from flight_sim.config.toml_files import key_path
from flight_sim.config.writing import toml_key


class EdithConfigError(ConfigError):
    """Campaign file that cannot be read, or that describes no valid campaign."""


def closest(key: str, known: Sequence[str]) -> str | None:
    """Return the known key closest to a misspelt one, preferring a case match."""
    same = [option for option in known if option.lower() == key.lower()]
    close = same or difflib.get_close_matches(key, known, n=1)
    return close[0] if close else None


def quoted_key_path(where: str, key: str) -> str:
    """Return the key path of a key in the table at ``where``, quoted if need be."""
    return key_path(where, toml_key(key))


def check_keys(
    table: Mapping[str, object],
    where: str,
    required: Sequence[str],
    optional: Sequence[str],
    usage: str,
) -> None:
    """Check a table holds every required key and no key it does not take.

    Raises:
        EdithConfigError: Naming the first unknown key, or the missing keys.
    """
    known = (*required, *optional)
    for key in table:
        if key not in known:
            close = closest(key, known)
            hint = f" (did you mean {close}?)" if close is not None else ""
            raise EdithConfigError(
                f"{quoted_key_path(where, key)}: unknown key{hint}; {usage}"
            )
    missing = [key for key in required if key not in table]
    if missing:
        place = f"{where}: " if where else ""
        raise EdithConfigError(f"{place}missing {' and '.join(missing)}; {usage}")
