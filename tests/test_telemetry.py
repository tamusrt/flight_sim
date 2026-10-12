"""Flight point, channel and Flight tests."""

import math
import operator

import numpy as np
import pytest

from flight_sim.environment.gravity import ConstantGravity
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.events import at_time
from flight_sim.integration import (
    IntegrationConfiguration,
    TruthConfiguration,
    derivative_computation,
    flight_point_si,
)
from flight_sim.simulation import Flight, Simulation
from flight_sim.telemetry import (
    CHANNELS,
    FlightPoint,
    channel_names,
    interpolate,
    slerp,
    table,
)
from flight_sim.units import scalar, vector
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_CONFIG = IntegrationConfiguration()
_WEIGHTLESS = IntegrationConfiguration(
    truth=TruthConfiguration(gravity=ConstantGravity(0.0))
)

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
    ).run()


def test_flight_point_matches_derivative_and_loads(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """A flight point holds the derivative's rates and the loads behind them."""
    point = flight_point_si(0.0, _MOVING, unpowered_rocket_properties, _CONFIG)
    derivative = derivative_computation(
        0.0, _MOVING, unpowered_rocket_properties, _CONFIG
    )
    table_area = unpowered_rocket_properties.aero_table.reference_area_m2

    aero, air = point.aero, point.environment.atmosphere
    coefficients = aero.coefficients

    assert np.allclose(
        point.derivative.acceleration, derivative.acceleration.m_as("m/s**2")
    )
    assert np.allclose(
        point.derivative.angular_acceleration,
        derivative.angular_acceleration.m_as("rad/s**2"),
    )
    assert aero.dynamic_pressure == pytest.approx(
        0.5 * air.air_density * aero.airspeed**2
    )
    assert np.allclose(
        aero.force,
        aero.dynamic_pressure
        * table_area
        * np.array([coefficients.cx, coefficients.cy, coefficients.cz]),
    )
    assert aero.airspeed == pytest.approx(math.sqrt(100.0**2 + 5.0**2))
    assert aero.mach == pytest.approx(aero.airspeed / air.speed_of_sound)


def test_specific_force_on_the_pad_is_the_rails_push(
    unpowered_rocket_properties: RocketProperties,
) -> None:
    """A motionless rocket held by the rail feels one g up, and is on the rail."""
    rail = LaunchRail(length=scalar(5.0, "m"), elevation=scalar(85.0, "deg"))
    config = IntegrationConfiguration(truth=TruthConfiguration(launch_rail=rail))
    state = rail.mount(RocketState())

    point = flight_point_si(0.0, state, unpowered_rocket_properties, config)
    nose_elevation = CHANNELS["derived.nose_elevation"].read(point)
    nose_azimuth = CHANNELS["derived.nose_azimuth"].read(point)

    assert np.allclose(point.derivative.acceleration, 0.0)
    assert np.linalg.norm(point.specific_force) == pytest.approx(
        point.environment.gravity
    )
    assert CHANNELS["derived.on_rail"].read(point) == 1.0
    assert nose_elevation == pytest.approx(85.0)
    assert nose_azimuth == pytest.approx(0.0)


def test_typed_point_round_trips_the_state(flight: Flight) -> None:
    """A typed point's state feeds straight back into the integrator."""
    point = flight.point(-1)
    derivative = derivative_computation(
        point.time.m_as("s"), point.state, flight.simulation.properties, _WEIGHTLESS
    )

    assert isinstance(point, FlightPoint)
    assert isinstance(point.state.orientation, Quaternion)
    assert np.allclose(
        derivative.acceleration.m_as("m/s**2"),
        point.derivative.acceleration.m_as("m/s**2"),
    )
    assert point.mass.mass.m_as("kg") == pytest.approx(20.0)
    assert point.aero.force.m_as("N") == pytest.approx(flight.points[-1].aero.force)
    assert point.state.rail_start is None


def test_channels_are_quantity_arrays(flight: Flight) -> None:
    """Channels read every point, in their units, and are read once."""
    force = flight["aero.force"]

    assert force.shape == (len(flight), 3)
    assert force.check("[force]")
    assert flight["aero.mach"].shape == (len(flight),)
    assert flight["aero.alpha"].units == "radian"
    assert flight["mass.inertia"].shape == (len(flight), 3, 3)
    assert flight["aero.force"] is force
    assert flight.time.m_as("s")[-1] == pytest.approx(0.1)
    assert all(isinstance(q, Quaternion) for q in flight.orientations)
    with pytest.raises(KeyError, match=r"No channel 'aero\.lift'"):
        flight["aero.lift"]  # pylint: disable=pointless-statement


def test_every_channel_reads(flight: Flight) -> None:
    """Every channel reads a finite value at every point."""
    for name in CHANNELS:
        values = flight[name].magnitude

        assert values.shape[0] == len(flight), name
        assert np.all(np.isfinite(values)), name


def test_channels_are_the_typed_attribute_paths(flight: Flight) -> None:
    """Each field channel holds what its attribute path holds in a FlightPoint."""
    point = flight.point(-1)

    for name, channel in CHANNELS.items():
        if name.startswith("derived."):
            continue
        typed = operator.attrgetter(name)(point)
        recorded = flight[name][-1]

        assert recorded.to(channel.units or "").magnitude == pytest.approx(
            getattr(typed, "magnitude", typed)
        ), name


def test_at_events_and_times(flight: Flight) -> None:
    """Events give their recorded point; times between points interpolate."""
    half = flight.at("half")
    between = flight.at(scalar(73.0, "ms"))
    start = flight.at(scalar(0.0, "s"))

    assert half.time.m_as("s") == pytest.approx(0.05, abs=1e-6)
    assert between.time.m_as("s") == pytest.approx(0.073)
    assert start.time.m_as("s") == 0.0
    assert flight.at(flight.time[-1]).time == flight.time[-1]
    with pytest.raises(KeyError, match="No event 'apogee'"):
        flight.at("apogee")
    with pytest.raises(ValueError, match="outside the flight"):
        flight.at(scalar(1.0, "s"))


def test_dataframe_lays_out_components_and_events(flight: Flight) -> None:
    """Each component gets a column with its units, and events label their rows."""
    frame = flight.to_dataframe(["aero.force", "time", "aero.mach", "mass.inertia"])

    assert list(frame.columns) == [
        "time [s]",
        "aero.mach [-]",
        "aero.force_x [N]",
        "aero.force_y [N]",
        "aero.force_z [N]",
        *(f"mass.inertia_{r}{c} [kg*m**2]" for r in "xyz" for c in "xyz"),
        "event",
    ]
    assert frame["event"].iloc[flight.events["half"]] == "half"
    assert frame["event"].iloc[-1] == "end"
    assert len(flight.to_dataframe().columns) > 60


def test_table_joins_events_on_one_row(flight: Flight) -> None:
    """Two events on the same row share its label."""
    frame = table(flight.points, {"a": 1, "b": 1}, ["time"])

    assert frame["event"].iloc[1] == "a, b"


def test_unknown_channel_pattern_is_rejected() -> None:
    """A pattern matching no channel is rejected."""
    assert channel_names(["aero.coefficients.c?"]) == [
        "aero.coefficients.cx",
        "aero.coefficients.cy",
        "aero.coefficients.cz",
    ]
    with pytest.raises(KeyError, match="No channel matches 'drag'"):
        channel_names(["drag"])


def test_slerp_follows_the_short_arc() -> None:
    """Halfway between identity and a quarter turn is an eighth turn."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    quarter = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    eighth = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])

    assert np.allclose(slerp(identity, quarter, 0.5), eighth)
    # The same rotation written with the opposite sign
    assert np.allclose(slerp(identity, -quarter, 0.5), eighth)
    # Nearly equal quaternions still give a unit quaternion
    nearby = np.array([1.0, 0.0, 0.0, 1e-4]) / math.sqrt(1.0 + 1e-8)
    assert np.linalg.norm(slerp(identity, nearby, 0.3)) == pytest.approx(1.0)


def test_interpolate_takes_the_nearer_rail_start(flight: Flight) -> None:
    """Numbers interpolate linearly; a rail start present at one end is not mixed."""
    start = flight.points[0]
    end = flight.points[1]._replace(
        state=flight.points[1].state._replace(rail_start=np.zeros(3))
    )

    early = interpolate(start, end, 0.25)
    late = interpolate(start, end, 0.75)

    assert early.time == pytest.approx(start.time + 0.25 * (end.time - start.time))
    assert early.state.rail_start is None
    assert late.state.rail_start is not None
    assert early.mass.mass == pytest.approx(start.mass.mass)
