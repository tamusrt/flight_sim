"""TOML input file loader tests."""

from pathlib import Path

import numpy as np
import pytest

from flight_sim.config import ConfigError, build, load, load_toml, load_with_source
from flight_sim.environment.gravity import ConstantGravity
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import UniformWind
from flight_sim.events import APOGEE, IMPACT
from flight_sim.flight_event import EventTime
from flight_sim.integration import TruthConfiguration
from flight_sim.simulation import Simulation
from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties

_EXAMPLE = Path("examples/sol_invictus")


def _write(directory: Path, name: str, text: str) -> Path:
    """Write a TOML file and return its path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_example_sim_builds_every_part() -> None:
    """The example sim file builds the vehicle, world, rail and events."""
    simulation = load(_EXAMPLE / "sim.toml", Simulation)

    assert simulation.name == "Sol Invictus example"
    assert simulation.stop_event == "impact"
    assert simulation.initial_time_step.m_as("s") == pytest.approx(0.01)
    assert isinstance(simulation.properties.engine, SolidEngine)
    assert simulation.properties.engine.grain.mass.m_as("kg") == pytest.approx(4.464)
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


def test_extends_overrides_only_what_it_names(tmp_path: Path) -> None:
    """A sim extending the example changes the wind and keeps the rest."""
    base = (_EXAMPLE / "sim.toml").resolve().as_posix()
    path = _write(
        tmp_path,
        "high_wind.toml",
        f"""
        extends = "{base}"

        [config.truth.wind]
        speed = "20 mph"
        from_azimuth = "270 deg"
        """,
    )
    simulation = load(path, Simulation)
    wind = simulation.config.truth.wind

    assert isinstance(wind, UniformWind)
    assert wind.speed.m_as("mph") == pytest.approx(20.0)
    assert wind.from_azimuth.m_as("deg") == pytest.approx(270.0)
    assert simulation.config.truth.launch_rail is not None
    assert simulation.config.sim.relative_tolerance == pytest.approx(1e-6)


def test_changed_type_replaces_base_table_and_passes_kwargs(tmp_path: Path) -> None:
    """A new engine type drops the base's keys; extras reach the class."""
    base = (_EXAMPLE / "vehicle.toml").resolve().as_posix()
    motor = Path("tests/test_data/Loki_M3464LB.rse").resolve().as_posix()
    path = _write(
        tmp_path,
        "vehicle.toml",
        f"""
        extends = "{base}"

        [engine]
        type = "flight_sim.vehicle.engine.SolidEngine.from_rse"
        motor_file_path = "{motor}"
        forward_end_location = {{ value = [-2.08, 0.0, 0.0], units = "m" }}
        reference_pressure = "101325 Pa"
        """,
    )
    properties = load(path, RocketProperties)
    engine = properties.engine

    assert isinstance(engine, SolidEngine)
    assert engine.grain.mass.m_as("kg") == pytest.approx(4.464)
    assert engine.grain.cg_location.m_as("m") == pytest.approx([-2.5991, 0.0, 0.0])
    assert engine.reference_pressure.m_as("Pa") == pytest.approx(101325.0)
    assert engine.pending_ignition is None


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
        forward_end_location = { value = [-2, 0, 0], units = "m" }
        ignition_time = { event = "apogee", delay = "1.5 s" }
        """,
    )
    engine = load(path, SolidEngine)
    ignition = engine.pending_ignition

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
        forward_end_location = { value = [-2, 0, 0], units = "m" }
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

    with pytest.raises(ConfigError, match=r"wind in .*truth\.toml: WindModel is ab"):
        load(path, TruthConfiguration)


def test_abstract_field_error_names_the_file_and_the_options(tmp_path: Path) -> None:
    """An included table's error names its own file and the classes to pick from."""
    _write(tmp_path, "site.toml", "[gravity]\n")
    path = _write(tmp_path, "truth.toml", 'include = "site.toml"\n')

    with pytest.raises(ConfigError) as error:
        load(path, TruthConfiguration)

    message = str(error.value)
    assert "gravity in " in message and "site.toml: GravityModel" in message
    assert "truth.toml" not in message
    assert '"flight_sim.environment.gravity.WGS84Gravity"' in message


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


def test_missing_files_name_the_file_that_names_them(tmp_path: Path) -> None:
    """A missing file, or a missing include, is reported against its includer."""
    path = _write(tmp_path, "top.toml", 'engine = { include = "nope.toml" }')

    with pytest.raises(ConfigError, match=r"cannot read .*missing\.toml"):
        load_toml(tmp_path / "missing.toml")
    with pytest.raises(ConfigError, match=r"top\.toml: cannot read .*nope\.toml"):
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

    cg_hint = MassProperties.__annotations__["cg_location"]
    assert np.allclose(inertia.cg_location.m_as("m"), [0.01, 0.02, 0.03])
    assert np.allclose(inertia.inertia.m_as("kg*m**2"), np.diag([1, 2, 3]))
    with pytest.raises(ConfigError, match="needs units"):
        build({"value": [1, 2, 3]}, MassProperties.__annotations__["cg_location"])
    with pytest.raises(ConfigError, match="one of value or diag"):
        build({"units": "m"}, MassProperties.__annotations__["cg_location"])
    with pytest.raises(ConfigError, match="cannot read units"):
        build(
            {"value": [1, 2, 3], "units": "furlongs per"},
            MassProperties.__annotations__["cg_location"],
        )
    with pytest.raises(ConfigError, match=r"cg\.value: expected numbers"):
        build({"value": ["a", 0, 0], "units": "m"}, cg_hint, "cg")
    with pytest.raises(ConfigError, match="cg: value needs 3 numbers"):
        build({"value": [0, 0], "units": "m"}, cg_hint, "cg")
    with pytest.raises(ConfigError, match="cg: diag needs 3 numbers"):
        build({"diag": [[1, 0], [0, 1]], "units": "m"}, cg_hint, "cg")


def test_unreadable_quantity_is_rejected() -> None:
    """A string that is not a quantity is rejected."""
    with pytest.raises(ConfigError, match="cannot read 'fast' as a quantity"):
        build({"speed": "fast"}, UniformWind)


def test_missing_arguments_are_named_for_the_callable_used() -> None:
    """A missing argument is reported with what that callable needs."""
    with pytest.raises(ConfigError) as from_rse:
        build(
            {
                "type": "flight_sim.vehicle.engine.SolidEngine.from_rse",
                "motor_file_path": "tests/test_data/Loki_M3464LB.rse",
            },
            SolidEngine,
            "engine",
        )
    with pytest.raises(ConfigError) as from_csv:
        build(
            {
                "type": "flight_sim.vehicle.engine.solid_engine_from_csv",
                "motor_file_path": "tests/test_data/standard_motor.csv",
            },
            SolidEngine,
            "engine",
        )

    assert str(from_rse.value) == (
        "engine: missing forward_end_location; SolidEngine.from_rse needs "
        "motor_file_path, forward_end_location and takes other SolidEngine fields"
    )
    assert "grain" not in str(from_rse.value)
    assert str(from_csv.value).startswith("engine: missing grain, casing;")


@pytest.mark.parametrize(
    ("friction", "expected"),
    [('"abc"', "a number, got 'abc'"), ("true", "a number, got True")],
)
def test_plain_fields_check_their_type(
    tmp_path: Path, friction: str, expected: str
) -> None:
    """A float field takes only numbers."""
    path = _write(
        tmp_path,
        "rail.toml",
        f"""
        length = "10 ft"
        elevation = "85 deg"
        azimuth = "0 deg"
        friction_coefficient = {friction}
        """,
    )

    with pytest.raises(ConfigError, match=f"friction_coefficient: expected {expected}"):
        load(path, LaunchRail)


def test_literal_fields_name_their_options() -> None:
    """A literal field lists the values it takes."""
    with pytest.raises(ConfigError, match="frame: expected one of 'body', 'missile'"):
        build(
            {
                "type": "flight_sim.utilities.data_loader.aero_table_from_csv",
                "filepath": "tests/test_data/standard_aero.csv",
                "reference_area": "1 m**2",
                "reference_length": "1 m",
                "reference_point": {"value": [0, 0, 0], "units": "m"},
                "frame": "wind",
            },
            AeroTable,
            "aero",
        )


def test_defaults_are_recorded_in_the_resolved_input() -> None:
    """Arguments left to their defaults are filled into the resolved input."""
    _, source = load_with_source(_EXAMPLE / "sim.toml", Simulation)
    table = source.table

    assert "properties.engine.ignition_time" in source.defaults
    assert table["properties"]["engine"]["ignition_time"] == "0.0 s"
    assert table["initial_state"]["orientation"]["type"].endswith("Quaternion")
    assert "properties.engine.grain" not in source.defaults
    assert source.path == (_EXAMPLE / "sim.toml").resolve()
    # Only arguments built as paths are made absolute
    assert Path(table["properties"]["engine"]["motor_file_path"]).is_absolute()
    assert table["name"] == "Sol Invictus example"
