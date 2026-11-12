"""Recorder, channel and Flight tests."""

import math
from dataclasses import replace

import numpy as np
import pytest
from support import recorded, truth

from flight_sim.environment.gravity import ConstantGravity
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.events import at_time
from flight_sim.integration import IntegrationConfiguration, derivative_computation
from flight_sim.simulation import Flight, Simulation
from flight_sim.telemetry import Channel, Recorder, record, resample
from flight_sim.units import scalar, vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_CONFIG = IntegrationConfiguration(truth=truth())
_WEIGHTLESS = IntegrationConfiguration(truth=truth(gravity=ConstantGravity(0.0)))

# Moving up and a little sideways, so the air loads are not zero
_MOVING = RocketState(velocity=vector((100.0, 5.0, 0.0), "m/s"))


@pytest.fixture(name="flight")
def _flight(unpowered_rocket_properties: RocketProperties) -> Flight:
    """Provides a tenth of a second of unpowered, weightless flight."""
    return Simulation(
        properties=unpowered_rocket_properties,
        config=_WEIGHTLESS,
        events=(at_time("half", scalar(0.05, "s")), at_time("end", scalar(0.1, "s"))),
        initial_state=_MOVING,
        stop_event="end",
        max_time=scalar(1.0, "s"),
    ).run()


def test_recorded_loads_match_the_derivative(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """A recorded state holds the derivative's rates and the loads behind them."""
    point = recorded(_MOVING, unpowered_rocket_properties, _CONFIG)
    derivative = derivative_computation(
        0.0, _MOVING, unpowered_rocket_properties, _CONFIG
    )
    table_area = unpowered_rocket_properties.aero_table.reference_area_m2
    coefficients = [point[f"aero.coefficients.c{axis}"] for axis in "xyz"]

    assert np.allclose(
        point["derivative.acceleration"], derivative.acceleration.m_as("m/s**2")
    )
    assert np.allclose(
        point["derivative.angular_acceleration"],
        derivative.angular_acceleration.m_as("rad/s**2"),
    )
    assert point["aero.dynamic_pressure"] == pytest.approx(
        0.5 * point["environment.atmosphere.air_density"] * point["aero.airspeed"] ** 2
    )
    assert np.allclose(
        point["aero.force"],
        point["aero.dynamic_pressure"] * table_area * np.array(coefficients),
    )
    assert point["aero.airspeed"] == pytest.approx(math.sqrt(100.0**2 + 5.0**2))
    assert point["aero.mach"] == pytest.approx(
        point["aero.airspeed"] / point["environment.atmosphere.speed_of_sound"]
    )


def test_specific_force_on_the_pad_is_the_rails_push(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """A motionless rocket held by the rail feels one g up, and is on the rail."""
    rail = LaunchRail(
        length=scalar(5.0, "m"), friction_coefficient=0.0, elevation=scalar(85.0, "deg")
    )
    config = IntegrationConfiguration(truth=truth(launch_rail=rail))

    point = recorded(rail.mount(RocketState()), unpowered_rocket_properties, config)

    assert np.allclose(point["derivative.acceleration"], 0.0)
    assert np.linalg.norm(point["specific_force"]) == pytest.approx(
        point["environment.gravity"]
    )
    assert point["derived.on_rail"] == 1.0
    assert point["derived.nose_elevation"] == pytest.approx(85.0)
    assert point["derived.nose_azimuth"] == pytest.approx(0.0)


def test_record_outside_a_step_is_ignored() -> None:
    """Values recorded with no step open are dropped."""
    recorder = Recorder()
    record("time", 1.0, "s")
    with recorder.step():
        record("time", 2.0, "s")

    assert recorder.channels()["time"].values.tolist() == [2.0]


def test_steps_must_record_the_same_channels() -> None:
    """A step naming other channels than the first is refused."""
    recorder = Recorder()
    with recorder.step():
        record("time", 0.0, "s")
        record("thrust", 0.0, "N")

    with pytest.raises(ValueError, match=r"\['thrust'\]"), recorder.step():
        record("time", 1.0, "s")


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (("time", 1.0, "s"), "recorded twice"),
        (("thrust", 1.0, "lbf"), "first as \\('N', 'linear'\\)"),
    ],
)
def test_a_channel_keeps_one_value_and_kind(
    second: tuple[str, float, str], message: str
) -> None:
    """A channel is recorded once per step, always in the same units."""
    recorder = Recorder()
    with recorder.step():
        record("time", 0.0, "s")
        record("thrust", 0.0, "N")

    with pytest.raises(ValueError, match=message), recorder.step():
        record("time", 1.0, "s")
        record(*second)


def test_channels_are_quantity_arrays(flight: Flight) -> None:
    """Channels read every step, in their units."""
    steps = len(flight["time"])
    force = flight["aero.force"]

    assert force.shape == (steps, 3)
    assert force.check("[force]")
    assert flight["aero.mach"].shape == (steps,)
    assert flight["aero.alpha"].units == "radian"
    assert flight["mass.inertia"].shape == (steps, 3, 3)
    assert flight["state.orientation"].shape == (steps, 4)
    assert flight["time"].m_as("s")[-1] == pytest.approx(0.1)
    with pytest.raises(KeyError, match=r"aero\.lift"):
        flight["aero.lift"]  # pylint: disable=pointless-statement


def test_every_channel_reads(flight: Flight) -> None:
    """Every channel holds a finite value at every step."""
    for name, channel in flight.channels.items():
        assert channel.values.shape[0] == len(flight["time"]), name
        assert np.all(np.isfinite(channel.values)), name


def test_at_events_and_times(flight: Flight) -> None:
    """Events give their recorded row; times between steps interpolate."""
    half = flight.at("half")
    between = flight.at(scalar(73.0, "ms"))
    end = flight["time"][-1]

    assert half["time"].m_as("s") == pytest.approx(0.05, abs=1e-6)
    assert half["aero.force"].m_as("N") == pytest.approx(
        flight["aero.force"][flight.events["half"]].m_as("N")
    )
    assert between["time"].m_as("s") == pytest.approx(0.073)
    assert np.linalg.norm(between["state.orientation"].magnitude) == pytest.approx(1)
    assert flight.at(scalar(0.0, "s"))["time"].m_as("s") == 0.0
    assert flight.at(end)["time"] == end
    with pytest.raises(KeyError, match="apogee"):
        flight.at("apogee")
    with pytest.raises(ValueError, match="outside the flight"):
        flight.at(scalar(1.0, "s"))


def test_dataframe_lays_out_components_and_events(flight: Flight) -> None:
    """Each component gets a column with its units, and events label their rows."""
    frame = flight.to_dataframe()

    assert list(frame.columns[:5]) == [
        "time [s]",
        "state.position_x [m]",
        "state.position_y [m]",
        "state.position_z [m]",
        "state.velocity_x [m/s]",
    ]
    assert {
        "aero.mach [-]",
        "aero.force_z [N]",
        "mass.inertia_xy [kg*m**2]",
        "state.orientation_w [-]",
    } <= set(frame.columns)
    assert frame.columns[-1] == "event"
    assert frame["event"].iloc[flight.events["half"]] == "half"
    assert frame["event"].iloc[-1] == "end"


def test_dataframe_joins_events_on_one_row(flight: Flight) -> None:
    """Two events on the same row share its label."""
    frame = replace(flight, events={"a": 1, "b": 1}).to_dataframe()

    assert frame["event"].iloc[1] == "a, b"


def test_resample_reads_each_channel_as_its_interp_says() -> None:
    """Linear blends, nearest snaps, wrapped turns the short way, slerp stays unit."""
    half_turn = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    channels = {
        "time": Channel(np.array([0.0, 1.0]), "s"),
        "linear": Channel(np.array([0.0, 4.0]), ""),
        "nearest": Channel(np.array([0.0, 1.0]), "", "nearest"),
        "wrapped": Channel(np.array([359.0, 3.0]), "deg", "wrapped"),
        "slerp": Channel(np.array([[1.0, 0.0, 0.0, 0.0], half_turn]), "", "slerp"),
    }

    sample = resample(channels, np.array([0.0, 0.25, 0.75, 1.0]))

    assert sample["time"].values == pytest.approx([0.0, 0.25, 0.75, 1.0])
    assert sample["linear"].values == pytest.approx([0.0, 1.0, 3.0, 4.0])
    assert sample["nearest"].values.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert sample["wrapped"].values == pytest.approx([359.0, 0.0, 2.0, 3.0])
    assert np.linalg.norm(sample["slerp"].values, axis=1) == pytest.approx(1.0)
    assert sample["slerp"].values[1] == pytest.approx(
        [math.cos(math.pi / 16), 0.0, 0.0, math.sin(math.pi / 16)]
    )
