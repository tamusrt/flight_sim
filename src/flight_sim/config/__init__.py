"""Build objects from TOML input files, in the format configs/README.md describes."""

from flight_sim.config.building import build
from flight_sim.config.errors import ConfigError
from flight_sim.config.loading import InputSource, load, load_with_source
from flight_sim.config.toml_files import load_toml
from flight_sim.config.writing import write_toml

__all__ = [
    "ConfigError",
    "InputSource",
    "build",
    "load",
    "load_toml",
    "load_with_source",
    "write_toml",
]
