"""Rocket shape tests."""

from pathlib import Path

import pytest

from flight_sim.config import ConfigError, build, load, load_toml
from flight_sim.vehicle.rocket_properties import RocketProperties

_VEHICLE = Path("examples/sol_invictus/vehicle.toml")


def test_example_shape_is_cached_in_si_units() -> None:
    """The example's shape, written in inches, reads into metres."""
    properties = load(_VEHICLE, RocketProperties)

    assert properties.enable_pitch_damping is True
    assert properties.shape.length_m == pytest.approx(206.19 * 0.0254)
    assert properties.shape.num_fins == 4
    assert properties.shape.fin_area_m2 == pytest.approx(67.5 * 0.0254**2)
    assert properties.shape.fin_pos_m == pytest.approx(-196.41 * 0.0254)


@pytest.mark.parametrize("key", ["enable_pitch_damping", "shape"])
def test_vehicle_needs_its_shape_and_damping_switch(key: str) -> None:
    """A vehicle missing the shape table or the damping switch fails, naming it."""
    vehicle = load_toml(_VEHICLE)
    del vehicle[key]

    with pytest.raises(
        ConfigError, match=rf"vehicle\.toml: missing {key}; RocketProperties needs"
    ):
        build(vehicle, RocketProperties, "properties")


@pytest.mark.parametrize("key", ["length", "num_fins", "fin_area", "fin_pos"])
def test_shape_keys_have_no_defaults(key: str) -> None:
    """A shape missing any key fails, naming it."""
    vehicle = load_toml(_VEHICLE)
    del vehicle["shape"][key]

    with pytest.raises(ConfigError, match=rf"missing {key}; RocketShape needs"):
        build(vehicle, RocketProperties, "properties")


def test_damping_switch_must_be_a_bool() -> None:
    """enable_pitch_damping takes true or false, not a number."""
    vehicle = load_toml(_VEHICLE)
    vehicle["enable_pitch_damping"] = 1

    with pytest.raises(ConfigError, match="expected true or false, got 1"):
        build(vehicle, RocketProperties, "properties")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("num_fins", 4.0, "expected an integer, got 4.0"),
        ("length", 206.19, "206.19 needs units"),
        ("fin_area", "67.5 in", "assigned to RocketShape.fin_area"),
        ("length", "0 m", "length must be finite and positive, got 0.0 m"),
        ("num_fins", -1, "num_fins must be 0 or more, got -1"),
        ("fin_area", "-1 m**2", "fin_area must be finite and 0 or more"),
        ("fin_pos", "196.41 in", "fin_pos must lie on the body"),
        ("fin_pos", "-207 in", "fin_pos must lie on the body"),
    ],
)
def test_shape_values_are_checked(key: str, value: object, message: str) -> None:
    """A shape value of the wrong type, units or size fails, naming the key."""
    vehicle = load_toml(_VEHICLE)
    vehicle["shape"][key] = value

    with pytest.raises(ConfigError, match=message):
        build(vehicle, RocketProperties, "properties")
