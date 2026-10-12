"""TOML input file loader tests."""

from pathlib import Path

import numpy as np
import pytest

from flight_sim.config import ConfigError, build, load, load_toml
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.environment.wind import UniformWind
from flight_sim.events import APOGEE, IMPACT
from flight_sim.flight_event import EventTime
from flight_sim.integration import TruthConfiguration
from flight_sim.simulation import Simulation
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties

_SIMS = Path("configs/sims")
_VEHICLES = Path("configs/vehicles")


def _write(directory: Path, name: str, text: str) -> Path:
    """Write a TOML file and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_nominal_sim_builds_every_part() -> None:
    """The nominal sim file builds the vehicle, world, rail and events."""
    simulation = load(_SIMS / "sol_invictus_nominal.toml", Simulation)

    assert simulation.name == "Sol Invictus - nominal"
    assert simulation.stop_event == "impact"
    assert simulation.initial_time_step.m_as("s") == pytest.approx(0.01)
    assert isinstance(simulation.properties.engine, SolidEngine)
    assert simulation.properties.engine.grain.mass.m_as("kg") == pytest.approx(5.0)
    rail = simulation.config.truth.launch_rail
    assert rail is not None
    assert rail.length.m_as("m") == pytest.approx(17 * 0.3048)
    assert simulation.config.truth.launch_elevation.m_as("ft") == pytest.approx(4595)
    assert simulation.events[1] is APOGEE
    assert simulation.events[2] is IMPACT
    assert simulation.event_names() == (
        "rail exit",
        "peak vertical velocity",
        "apogee",
        "impact",
    )


def test_extends_overrides_only_what_it_names() -> None:
    """The high wind sim changes the wind and keeps the rest of the nominal sim."""
    simulation = load(_SIMS / "sol_invictus_high_wind.toml", Simulation)
    wind = simulation.config.truth.wind

    assert isinstance(wind, UniformWind)
    assert wind.speed.m_as("mph") == pytest.approx(20.0)
    assert wind.from_azimuth.m_as("deg") == pytest.approx(270.0)
    assert simulation.config.truth.launch_rail is not None
    assert simulation.config.sim.relative_tolerance == pytest.approx(1e-6)


def test_changed_type_replaces_base_table_and_passes_kwargs() -> None:
    """A new engine type drops the base's grain and casing; extras reach the class."""
    properties = load(_VEHICLES / "sol_invictus_loki.toml", RocketProperties)
    engine = properties.engine

    assert isinstance(engine, SolidEngine)
    assert engine.grain.mass.m_as("kg") == pytest.approx(4.464)
    assert engine.reference_pressure.m_as("Pa") == pytest.approx(101325.0)
    assert engine.pending_ignition() is None


def test_event_ignition_builds_from_a_table(tmp_path: Path) -> None:
    """An ignition_time table builds an EventTime the engine waits on."""
    motor = Path("tests/test_data/Loki_M3464LB.eng")
    (tmp_path / "motor.eng").write_bytes(motor.read_bytes())
    path = _write(
        tmp_path,
        "engine.toml",
        """
        type = "flight_sim.vehicle.engine.SolidEngine.from_eng"
        motor_file_path = "motor.eng"
        ignition_time = { event = "apogee", delay = "1.5 s" }
        """,
    )
    engine = load(path, SolidEngine)
    ignition = engine.pending_ignition()

    assert isinstance(ignition, EventTime)
    assert ignition.event == "apogee"
    assert ignition.delay.m_as("s") == pytest.approx(1.5)
    assert engine.get_thrust(1.0, 101325.0) == 0.0


def test_paths_resolve_relative_to_their_file(tmp_path: Path) -> None:
    """A file included from another directory reads paths relative to itself."""
    data = Path("tests/test_data").resolve()
    vehicle_dir = tmp_path / "vehicles"
    vehicle_dir.mkdir()
    (vehicle_dir / "motor.rse").write_bytes((data / "Loki_M3464LB.rse").read_bytes())
    _write(
        vehicle_dir,
        "engine.toml",
        """
        type = "flight_sim.vehicle.engine.SolidEngine.from_rse"
        motor_file_path = "motor.rse"
        """,
    )
    path = _write(tmp_path, "top.toml", 'engine = { include = "vehicles/engine.toml" }')

    table = load_toml(path)
    engine = build(table["engine"], SolidEngine)

    assert isinstance(engine, SolidEngine)


def test_quantities_require_units(tmp_path: Path) -> None:
    """A bare number or unitless string for a quantity is rejected."""
    bare = _write(
        tmp_path,
        "bare.toml",
        """
        mass = 3.0
        cg_location = { value = [0, 0, 0], units = "m" }
        inertia = { diag = [1, 1, 1], units = "kg*m**2" }
        """,
    )
    unitless = _write(tmp_path, "unitless.toml", "speed = '5'")

    with pytest.raises(ConfigError, match=r"mass: 3\.0 needs units"):
        load(bare, MassProperties)
    with pytest.raises(ConfigError, match="needs units"):
        load(unitless, UniformWind)


def test_wrong_dimension_names_the_key(tmp_path: Path) -> None:
    """A quantity of the wrong dimension fails with the key that holds it."""
    path = _write(tmp_path, "wind.toml", "speed = '5 kg'")

    with pytest.raises(ConfigError, match="top level"):
        load(path, UniformWind)


def test_abstract_field_needs_a_type(tmp_path: Path) -> None:
    """A table filling an abstract field without a type is rejected."""
    path = _write(tmp_path, "truth.toml", "[wind]\nspeed = '5 m/s'")

    with pytest.raises(ConfigError, match="wind: WindModel is abstract"):
        load(path, TruthConfiguration)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A key the class does not take is reported with the keys it does take."""
    path = _write(tmp_path, "wind.toml", "speed = '5 m/s'\ngust = '2 m/s'")

    with pytest.raises(ConfigError, match=r"unknown keys \['gust'\]"):
        load(path, UniformWind)


def test_bad_dotted_paths_are_rejected(tmp_path: Path) -> None:
    """A type or ref naming nothing that exists is rejected."""
    no_attribute = _write(
        tmp_path, "a.toml", "[gravity]\ntype = 'flight_sim.environment.gravity.Moon'"
    )
    no_module = _write(tmp_path, "b.toml", "[gravity]\nref = 'not_a_module.X'")

    with pytest.raises(ConfigError, match=r"gravity: .*Moon does not exist"):
        load(no_attribute, TruthConfiguration)
    with pytest.raises(ConfigError, match="gravity: no module"):
        load(no_module, TruthConfiguration)


def test_ref_takes_no_other_keys(tmp_path: Path) -> None:
    """A ref table with other keys is rejected."""
    path = _write(
        tmp_path, "a.toml", "[gravity]\nref = 'flight_sim.events.APOGEE'\nx = 1"
    )

    with pytest.raises(ConfigError, match="ref takes no other keys"):
        load(path, TruthConfiguration)


def test_include_cycle_is_rejected(tmp_path: Path) -> None:
    """A file that includes itself is rejected."""
    path = _write(tmp_path, "loop.toml", "[gravity]\ninclude = 'loop.toml'")

    with pytest.raises(ConfigError, match="includes itself"):
        load_toml(path)


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    """A file that is not TOML is reported with its path."""
    path = _write(tmp_path, "broken.toml", "speed = ")

    with pytest.raises(ConfigError, match=r"broken\.toml"):
        load_toml(path)


def test_load_checks_the_built_class(tmp_path: Path) -> None:
    """A file whose type builds something other than the requested class fails."""
    path = _write(
        tmp_path,
        "gravity.toml",
        "type = 'flight_sim.environment.gravity.ConstantGravity'\nmagnitude_m_s2 = 9",
    )

    with pytest.raises(ConfigError, match="builds a ConstantGravity"):
        load(path, UniformWind)
    assert load(path, ConstantGravity).magnitude_m_s2 == 9.0


def test_array_quantities() -> None:
    """Value and diag tables build arrays; malformed ones are rejected."""
    inertia = build(
        {
            "mass": "1 kg",
            "cg_location": {"value": [1, 2, 3], "units": "cm"},
            "inertia": {"diag": [1, 2, 3], "units": "kg*m**2"},
        },
        MassProperties,
    )

    assert np.allclose(inertia.cg_location.m_as("m"), [0.01, 0.02, 0.03])
    assert np.allclose(inertia.inertia.m_as("kg*m**2"), np.diag([1, 2, 3]))
    with pytest.raises(ConfigError, match="needs units"):
        build({"value": [1, 2, 3]}, MassProperties.__annotations__["cg_location"])
    with pytest.raises(ConfigError, match="one of value or diag"):
        build({"units": "m"}, MassProperties.__annotations__["cg_location"])
    with pytest.raises(ConfigError, match="cannot read units"):
        build(
            {"value": [1], "units": "furlongs per"},
            MassProperties.__annotations__["cg_location"],
        )


def test_unreadable_quantity_is_rejected() -> None:
    """A string that is not a quantity is rejected."""
    with pytest.raises(ConfigError, match="cannot read 'fast' as a quantity"):
        build({"speed": "fast"}, UniformWind)
