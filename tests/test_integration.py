"""Integration kernel tests."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.__main__ import main
from flight_sim.environment.atmosphere import StandardAtmosphere1976, VacuumAtmosphere
from flight_sim.environment.gravity import ConstantGravity, GravityModel, WGS84Gravity
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import UniformWind
from flight_sim.events import IMPACT
from flight_sim.integration import (
    IntegrationConfiguration,
    SimConfiguration,
    TruthConfiguration,
    adaptive_step,
    derivative_computation,
    locate_event,
    step,
)
from flight_sim.units import scalar, vector, zero_vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.utilities.dcm import body_to_world
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_properties import RocketProperties, TrapezoidFinSet
from flight_sim.vehicle.rocket_state import RocketState

_CONFIG = IntegrationConfiguration()


def _baseline_state() -> RocketState:
    """Launchpad state of the 25 kg test rocket the unit tests are built on."""
    return RocketState(
        position=vector((0.0, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
        current_mass=scalar(25.0, "kg"),
        inertia=vector((2.5, 150.0, 150.0), "kg*m**2"),
        cg_location=vector((-1.5, 0.0, 0.0), "m"),
        angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
        orientation=Quaternion(q_x=0.0, q_y=0.0, q_z=0.0, q_w=1.0),
        on_rail=True,
    )


def _bare_state() -> RocketState:
    """Return a massless state at rest at the origin.

    Zero mass switches off thrust and aerodynamics, leaving only gravity.
    """
    return RocketState(
        current_mass=scalar(0.0, "kg"),
        inertia=vector((0.1, 2.5, 2.5), "kg*m**2"),
        cg_location=vector((-2.5, 0.0, 0.0), "m"),
    )


@patch("flight_sim.__main__.adaptive_step")
def test_main_echoes_test_input(mock_adaptive_step: MagicMock) -> None:
    """Verify the executive simulation loop runs from launch to impact."""

    # Force the physics step to immediately return an underground state.
    mock_state = _baseline_state()
    mock_state.position = vector((-10.0, 0.0, 0.0), "m")
    mock_adaptive_step.return_value = (
        mock_state,
        scalar(0.2, "s"),
        scalar(0.2, "s"),
        IMPACT,
    )

    main()

    # Verify the simulation successfully fired the physics engine before exiting
    mock_adaptive_step.assert_called()


@patch("flight_sim.__main__.RocketProperties")
def test_main_rejects_unknown_argument(mock_properties: MagicMock) -> None:
    """Verify the module behaves predictably if aerodynamic files are missing."""
    mock_properties.side_effect = FileNotFoundError("Missing standard_aero.csv")
    with pytest.raises(FileNotFoundError):
        main()


def test_step_zero_force_keeps_velocity_constant(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Tests that the step function keeps velocity constant"""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    state = _bare_state()
    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(0.01, "s"))
    assert np.allclose(next_state.velocity.m_as("m/s"), np.zeros(3))


def test_step_with_gravity_changes_velocity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Tests that gravity correctly accelerates rocket downwards"""
    state = _bare_state()
    next_state = step(
        0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.01, "s")
    )
    assert next_state.velocity[0].m_as("m/s") < 0


def test_step_result_keeps_expected_units(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The integrated state stays in the units its fields declare."""
    next_state = step(
        0.0,
        _bare_state(),
        baseline_rocket_properties,
        _CONFIG,
        scalar(0.01, "s"),
    )

    assert next_state.position.check("[length]")
    assert next_state.velocity.check("[length] / [time]")
    assert next_state.angular_velocity.check("1 / [time]")
    assert next_state.current_mass.check("[mass]")


def test_step_accepts_any_time_unit(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A dt given in milliseconds integrates the same as the equivalent seconds."""
    from_ms = step(
        0.0,
        _bare_state(),
        baseline_rocket_properties,
        _CONFIG,
        scalar(10.0, "ms"),
    )
    from_s = step(
        0.0,
        _bare_state(),
        baseline_rocket_properties,
        _CONFIG,
        scalar(0.01, "s"),
    )

    assert np.allclose(from_ms.velocity.m_as("m/s"), from_s.velocity.m_as("m/s"))


def test_step_rejects_dt_that_is_not_a_time(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A dt in the wrong dimension is rejected rather than silently integrated."""
    with pytest.raises(DimensionalityError):
        step(
            0.0,
            _bare_state(),
            baseline_rocket_properties,
            _CONFIG,
            scalar(0.01, "m"),
        )


def test_step_adaptive_scaling_and_rejection(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Stress adaptive rejection, scaling, and overshoot limits."""

    class _CubicGravity(GravityModel):
        """Gravity growing with the cube of the altitude."""

        def magnitude(self, latitude_rad: float, altitude_m: float) -> float:
            return altitude_m**3

    config = IntegrationConfiguration(truth=TruthConfiguration(gravity=_CubicGravity()))

    state = _bare_state()
    state.position = vector((10.0, 0.0, 0.0), "m")
    state.velocity = vector((50.0, 0.0, 0.0), "m/s")
    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(5.0, "s"))

    assert next_state is not None


def test_step_triggers_aerodynamic_calculations(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Ensures the integration engine runs the drag/lift physics block."""
    state = _bare_state()

    state.current_mass = scalar(25.0, "kg")
    state.velocity = vector((50.0, 0.0, 0.0), "m/s")

    next_state = step(0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.1, "s"))

    assert next_state is not None


@pytest.mark.parametrize(
    "attitude", [Quaternion(), Quaternion(q_w=0.5, q_x=-0.5, q_y=0.5, q_z=0.5)]
)
def test_aero_loads_match_a_table_row(
    baseline_rocket_properties: RocketProperties, attitude: Quaternion
) -> None:
    """On a grid point the loads are the row's coefficients scaled by q*S and L."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    air = config.truth.atmosphere.conditions(0.0)
    speed = 0.1 * air.speed_of_sound
    alpha = np.radians(5.0)
    to_world = body_to_world(attitude)

    state = _bare_state()
    state.current_mass = scalar(20.0, "kg")
    state.orientation = attitude
    # Mach 0.1, alpha 5 degrees, crossflow along body +Z so phi_a is 0
    state.velocity = vector(
        to_world @ (speed * np.array([np.cos(alpha), 0.0, np.sin(alpha)])), "m/s"
    )
    coasting = 100.0

    derivative = derivative_computation(
        coasting, state, baseline_rocket_properties, config
    )

    force_scale = 0.5 * air.air_density * speed**2 * 0.0182414692
    force = force_scale * np.array([-0.6, 0.0, -0.1])
    lever_arm = np.array([2.5, 0.0, 0.0])  # Nose-tip reference point from the CG
    torque = force_scale * 0.1524 * np.array([0.0, -2.0, 0.0]) + np.cross(
        lever_arm, force
    )
    assert derivative.acceleration.m_as("m/s**2") == pytest.approx(
        to_world @ force / 20.0
    )
    assert derivative.angular_acceleration.m_as("rad/s**2") == pytest.approx(
        torque / np.array([0.1, 2.5, 2.5])
    )


def test_step_angular_velocity_rotates_orientation(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A constant body rate about X turns the orientation by rate * time."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    state = _bare_state()
    state.angular_velocity = vector((np.pi / 2, 0.0, 0.0), "rad/s")

    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(0.1, "s"))

    half_angle = (np.pi / 2) * 0.1 / 2
    assert next_state.orientation.q_w == pytest.approx(np.cos(half_angle), abs=1e-6)
    assert next_state.orientation.q_x == pytest.approx(np.sin(half_angle), abs=1e-6)
    assert next_state.orientation.q_y == pytest.approx(0.0, abs=1e-6)
    assert next_state.orientation.q_z == pytest.approx(0.0, abs=1e-6)


def test_step_keeps_orientation_normalized(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The orientation stays a unit quaternion while the rocket is pitching."""
    state = _bare_state()
    state.velocity = vector((200.0, 50.0, 0.0), "m/s")
    state.angular_velocity = vector((0.3, -1.2, 0.8), "rad/s")
    state.current_mass = scalar(20.0, "kg")

    next_state = step(0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.1, "s"))

    orientation = next_state.orientation
    norm = np.linalg.norm(
        [orientation.q_w, orientation.q_x, orientation.q_y, orientation.q_z]
    )
    assert norm == pytest.approx(1.0)
    assert orientation.q_w != 1.0  # Orientation actually moved off the pad attitude


def test_torque_free_rotation_conserves_angular_momentum(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """With no torque, the world-frame angular momentum and spin energy hold."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    state = _bare_state()
    state.inertia = vector((1.0, 2.0, 3.0), "kg*m**2")
    state.angular_velocity = vector((0.3, 1.0, 0.5), "rad/s")

    def momentum_and_energy(spinning: RocketState) -> tuple[np.ndarray, float]:
        """Return the world-frame angular momentum and the rotational energy."""
        inertia = spinning.inertia.m_as("kg*m**2")
        rate = spinning.angular_velocity.m_as("rad/s")
        momentum = body_to_world(spinning.orientation) @ (inertia * rate)
        return momentum, 0.5 * float(rate @ (inertia * rate))

    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(5.0, "s"))

    momentum, energy = momentum_and_energy(state)
    next_momentum, next_energy = momentum_and_energy(next_state)
    assert next_momentum == pytest.approx(momentum, rel=1e-5)
    assert next_energy == pytest.approx(energy, rel=1e-5)


def test_step_length_is_limited_by_the_attitude_error(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A rocket turning in place still integrates its attitude accurately."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    state = _bare_state()
    state.angular_velocity = vector((0.0, 0.5, 0.0), "rad/s")

    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(10.0, "s"))

    half_angle = 0.5 * 10.0 / 2
    orientation = next_state.orientation
    assert [
        orientation.q_w,
        orientation.q_x,
        orientation.q_y,
        orientation.q_z,
    ] == pytest.approx([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], abs=1e-6)


def test_launch_elevation_sets_the_altitude_of_the_air_and_gravity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A rocket h above a pad at elevation e flies as if h + e above sea level."""
    state = _bare_state()
    state.current_mass = scalar(20.0, "kg")
    state.velocity = vector((200.0, 30.0, 0.0), "m/s")
    state.position = vector((100.0, 0.0, 0.0), "m")
    raised = replace(state, position=vector((1500.0, 0.0, 0.0), "m"))
    coasting = 100.0

    def rates(rocket: RocketState, config: IntegrationConfiguration) -> np.ndarray:
        """Return the linear and angular accelerations side by side."""
        derivative = derivative_computation(
            coasting, rocket, baseline_rocket_properties, config
        )
        return np.concatenate(
            (
                derivative.acceleration.m_as("m/s**2"),
                derivative.angular_acceleration.m_as("rad/s**2"),
            )
        )

    high_pad = IntegrationConfiguration(
        truth=TruthConfiguration(launch_elevation=scalar(1400.0, "m"))
    )
    assert rates(state, high_pad) == pytest.approx(rates(raised, _CONFIG))
    assert rates(state, high_pad) != pytest.approx(rates(state, _CONFIG))


_RAIL = LaunchRail(
    length=scalar(5.0, "m"), elevation=scalar(80.0, "deg"), azimuth=scalar(90.0, "deg")
)


def _rail_state() -> RocketState:
    """Return the default rocket at rest on the test rail."""
    state = _baseline_state()
    state.orientation = _RAIL.orientation()
    return state


def test_rail_guides_the_rocket_up_it(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """On the rail the rocket slides up it without turning, despite a crosswind."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(
            wind=UniformWind(speed=scalar(10.0, "m/s")), launch_rail=_RAIL
        )
    )
    state = _rail_state()
    rail = _RAIL.direction()

    next_state = step(0.0, state, baseline_rocket_properties, config, scalar(0.2, "s"))

    position = next_state.position.m_as("m")
    assert next_state.on_rail
    assert float(position @ rail) > 0.0
    assert np.cross(rail, position) == pytest.approx(np.zeros(3), abs=1e-12)
    assert next_state.angular_velocity.m_as("rad/s") == pytest.approx(np.zeros(3))
    assert next_state.orientation == state.orientation


def test_rail_holds_an_unpowered_rocket_on_the_pad(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Without thrust, gravity cannot push the rocket down through the pad."""
    config = IntegrationConfiguration(truth=TruthConfiguration(launch_rail=_RAIL))
    after_burnout = 100.0

    next_state = step(
        after_burnout,
        _rail_state(),
        baseline_rocket_properties,
        config,
        scalar(1.0, "s"),
    )

    assert next_state.on_rail
    assert next_state.position.m_as("m") == pytest.approx(np.zeros(3))
    assert next_state.velocity.m_as("m/s") == pytest.approx(np.zeros(3))


def test_default_rocket_turns_over_and_lands_nose_first() -> None:
    """With the estimated aero, a tilted launch falls nose-first after apogee."""
    rail = LaunchRail(length=scalar(17.0, "ft"), elevation=scalar(85.0, "deg"))
    properties = RocketProperties(
        aero_table=aero_table_from_csv(
            "data/aero/estimated_aero.csv",
            reference_area=scalar(0.0182414692, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=zero_vector("m"),
            frame="missile",
        ),
        motor_file_path="tests/test_data/standard_motor.csv",
        propellant_mass=5.0,
    )
    config = IntegrationConfiguration(truth=TruthConfiguration(launch_rail=rail))
    state = _baseline_state()
    state.orientation = rail.orientation()

    time = 0.0
    dt = scalar(0.01, "s")
    hit = None
    while hit is not IMPACT:
        state, dt_taken, dt, hit = adaptive_step(
            time, state, properties, config, dt, events=(IMPACT,)
        )
        time += float(dt_taken.m_as("s"))
        assert time < 100.0

    nose = body_to_world(state.orientation)[:, 0]
    velocity = state.velocity.m_as("m/s")
    angle_of_attack = np.arccos(nose @ velocity / np.linalg.norm(velocity))
    assert velocity[0] < 0.0
    assert np.degrees(angle_of_attack) < 5.0


def test_six_dof_aerodynamic_response(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Verify that all six coefficients drive every axis."""

    baseline_rocket_properties.aero_table.values[:] = 1.0

    baseline_rocket_properties.engine = MagicMock(
        get_thrust=MagicMock(return_value=0.0),
        get_mass_flow=MagicMock(return_value=0.0),
    )

    state = _baseline_state()
    state.position = vector((1000.0, 0.0, 0.0), "m")
    state.velocity = vector((100.0, 50.0, 0.0), "m/s")

    derivative = derivative_computation(
        time=0.0,
        state=state,
        properties=baseline_rocket_properties,
        config=_CONFIG,
    )

    angular_accel = derivative.angular_acceleration.m_as("rad/s**2")
    linear_accel = derivative.acceleration.m_as("m/s**2")

    assert np.all(np.abs(angular_accel) > 0.0)

    assert abs(linear_accel[1]) > 0.0
    assert abs(linear_accel[2]) > 0.0


@pytest.mark.parametrize("vertical_velocity", [100.0, -100.0])
def test_aerodynamics_finite_at_zero_and_reversed_angle_of_attack(
    baseline_rocket_properties: RocketProperties, vertical_velocity: float
) -> None:
    """A vertical rocket flying nose-first or tail-first gets finite loads."""
    baseline_rocket_properties.aero_table.values[:] = 1.0
    baseline_rocket_properties.engine = MagicMock(
        get_thrust=MagicMock(return_value=0.0),
        get_mass_flow=MagicMock(return_value=0.0),
    )
    state = _baseline_state()
    state.position = vector((1000.0, 0.0, 0.0), "m")
    state.velocity = vector((vertical_velocity, 0.0, 0.0), "m/s")

    derivative = derivative_computation(0.0, state, baseline_rocket_properties, _CONFIG)

    angular_accel = derivative.angular_acceleration.m_as("rad/s**2")
    linear_accel = derivative.acceleration.m_as("m/s**2")
    assert np.all(np.isfinite(angular_accel))
    assert np.all(np.isfinite(linear_accel))
    assert abs(angular_accel[0]) > 0.0


def test_locate_event_finds_ballistic_apogee(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A drag-free coast peaks at the time and height the closed form gives."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(9.81))
    )
    start = _bare_state()
    start.velocity = vector((50.0, 0.0, 0.0), "m/s")
    dt = scalar(10.0, "s")
    end = step(0.0, start, baseline_rocket_properties, config, dt)

    apogee, time_to_apogee = locate_event(
        0.0,
        start,
        end,
        baseline_rocket_properties,
        config,
        dt=dt,
        event=lambda _time, state: float(state.velocity.m_as("m/s")[0]),
    )

    assert time_to_apogee.m_as("s") == pytest.approx(50.0 / 9.81, abs=1e-8)
    assert apogee.position.m_as("m")[0] == pytest.approx(50.0**2 / (2 * 9.81))
    assert apogee.velocity.m_as("m/s")[0] == pytest.approx(0.0, abs=1e-7)


def test_step_warns_at_minimum_step_length(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """A step whose error never meets tolerance is accepted at the minimum length."""
    config = IntegrationConfiguration(
        sim=SimConfiguration(
            position_tolerance=scalar(1e-7, "m"), min_time_step=scalar(1.0, "ns")
        )
    )

    def over_tolerance(
        _time: float, values: np.ndarray, _dt: float, _inputs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        values_4th = values.copy()
        values_4th[0] += 1e-6
        return values.copy(), values_4th

    monkeypatch.setattr("flight_sim.integration.rkf45_step", over_tolerance)

    with pytest.warns(RuntimeWarning, match="minimum step length") as record:
        next_state = step(
            1.0, _bare_state(), baseline_rocket_properties, config, scalar(3.0, "ns")
        )

    assert len(record) == 3  # One warning per minimum-length step
    assert next_state.position.m_as("m") == pytest.approx([0.0, 0.0, 0.0])


def test_step_raises_on_nan_error(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """A NaN error estimate stops the integration instead of looping."""

    def nan_step(
        _time: float, values: np.ndarray, _dt: float, _inputs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        return values.copy(), np.full_like(values, np.nan)

    monkeypatch.setattr("flight_sim.integration.rkf45_step", nan_step)

    with pytest.raises(FloatingPointError, match="nan"):
        step(0.0, _bare_state(), baseline_rocket_properties, _CONFIG, scalar(0.01, "s"))


def test_configured_atmosphere_is_used(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """In a vacuum atmosphere a coasting rocket feels no aerodynamic force."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(
            atmosphere=VacuumAtmosphere(), gravity=ConstantGravity(0.0)
        )
    )
    state = _bare_state()
    state.current_mass = scalar(20.0, "kg")
    state.velocity = vector((200.0, 50.0, 0.0), "m/s")

    next_state = step(
        100.0, state, baseline_rocket_properties, config, scalar(1.0, "s")
    )

    assert next_state.velocity.m_as("m/s") == pytest.approx([200.0, 50.0, 0.0])


def test_configured_launch_latitude_sets_gravity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Gravity follows the configured latitude through the default WGS84 model."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(launch_latitude=scalar(90.0, "deg"))
    )

    next_state = step(
        0.0, _bare_state(), baseline_rocket_properties, config, scalar(1.0, "s")
    )

    polar_gravity = WGS84Gravity().magnitude(np.pi / 2, 0.0)
    assert polar_gravity > WGS84Gravity().magnitude(0.0, 0.0)
    assert next_state.velocity.m_as("m/s")[0] == pytest.approx(-polar_gravity, rel=1e-5)


def test_uniform_wind_loads_a_rocket_at_rest(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A rocket at rest feels no drag in still air and is pushed along a wind."""
    properties = baseline_rocket_properties
    state = _baseline_state()
    state.position = vector((1000.0, 0.0, 0.0), "m")
    coasting = 100.0

    still_air = IntegrationConfiguration(
        truth=TruthConfiguration(
            atmosphere=StandardAtmosphere1976(), gravity=ConstantGravity(0.0)
        )
    )
    in_still_air = derivative_computation(coasting, state, properties, still_air)
    assert in_still_air.acceleration.m_as("m/s**2") == pytest.approx(np.zeros(3))

    # A west wind, blowing toward +Y
    windy = IntegrationConfiguration(
        truth=TruthConfiguration(
            wind=UniformWind(
                speed=scalar(20.0, "m/s"), from_azimuth=scalar(270.0, "deg")
            ),
            gravity=ConstantGravity(0.0),
        )
    )
    in_wind = derivative_computation(coasting, state, properties, windy)
    acceleration = in_wind.acceleration.m_as("m/s**2")
    assert acceleration[1] > 0.0  # Blown downwind
    assert acceleration[2] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "crossflow_direction",
    [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)],
)
def test_table_loads_are_symmetric_about_the_nose(
    baseline_rocket_properties: RocketProperties,
    crossflow_direction: tuple[float, float, float],
) -> None:
    """The same alpha gives the same loads whichever way the crossflow points."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(gravity=ConstantGravity(0.0))
    )
    nose = np.array([1.0, 0.0, 0.0])
    speed, alpha_tot = 150.0, np.radians(5.0)
    coasting = 100.0

    def loads(crossflow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the accelerations of the pad-attitude rocket in one crossflow."""
        state = _baseline_state()
        state.position = vector((1000.0, 0.0, 0.0), "m")
        state.velocity = vector(
            speed * (np.cos(alpha_tot) * nose + np.sin(alpha_tot) * crossflow), "m/s"
        )
        derivative = derivative_computation(
            coasting, state, baseline_rocket_properties, config
        )
        return (
            derivative.acceleration.m_as("m/s**2"),
            derivative.angular_acceleration.m_as("rad/s**2"),
        )

    crossflow = np.array(crossflow_direction)
    reference_linear, reference_angular = loads(np.array([0.0, 1.0, 0.0]))
    linear, angular = loads(crossflow)

    # The side force opposes the +Y crossflow and the stable rocket turns into
    # it, and these loads rotate with the crossflow
    assert reference_linear[1] < 0.0
    assert reference_angular[2] > 0.0
    assert linear == pytest.approx(
        reference_linear[0] * nose + reference_linear[1] * crossflow, abs=1e-9
    )
    assert angular == pytest.approx(
        reference_angular[2] * np.cross(nose, crossflow), abs=1e-9
    )


def test_configuration_rejects_wrong_units() -> None:
    """A tolerance that is not a length is rejected on construction."""
    with pytest.raises(DimensionalityError, match="position_tolerance"):
        SimConfiguration(position_tolerance=scalar(1.0, "s"))


def test_gyroscopic_coupling_turns_roll_and_pitch_into_yaw(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Torque-free spin about X and Y accelerates Z by (Ix - Iy) / Iz * wx * wy."""
    config = IntegrationConfiguration(
        truth=TruthConfiguration(
            atmosphere=VacuumAtmosphere(), gravity=ConstantGravity(0.0)
        )
    )
    state = _bare_state()
    state.current_mass = scalar(20.0, "kg")
    state.inertia = vector((1.0, 2.0, 3.0), "kg*m**2")
    state.angular_velocity = vector((1.0, 1.0, 0.0), "rad/s")

    next_state = step(
        1000.0, state, baseline_rocket_properties, config, scalar(1e-3, "s")
    )

    rates = next_state.angular_velocity.m_as("rad/s")
    assert rates[2] == pytest.approx(-1e-3 / 3.0, rel=1e-3)


def test_fin_misalignment_spins_up_roll_and_damping_limits_it(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A fin defect rolls the rocket; with none, or at the balance rate, it does not."""
    fins = TrapezoidFinSet(
        fin_count=4,
        root_chord_m=0.3,
        tip_chord_m=0.1,
        span_m=0.15,
        sweep_length_m=0.2,
        body_radius_m=0.08,
    )
    state = _bare_state()
    state.velocity = vector((200.0, 0.0, 0.0), "m/s")
    state.current_mass = scalar(20.0, "kg")

    def roll_rate(misalignment_rad: float, initial_rate: float = 0.0) -> float:
        baseline_rocket_properties.fins = replace(
            fins, misalignment_rad=misalignment_rad
        )
        state.angular_velocity = vector((initial_rate, 0.0, 0.0), "rad/s")
        step_state = step(
            0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.01, "s")
        )
        return float(step_state.angular_velocity.m_as("rad/s")[0])

    assert roll_rate(0.0) == pytest.approx(0.0, abs=1e-12)
    assert roll_rate(1e-3) > 0.0
    assert roll_rate(-1e-3) == pytest.approx(-roll_rate(1e-3))
    # Forcing and damping balance at p = V * misalignment * F / K
    balance = 200.0 * 1e-3 * fins.roll_forcing_integral / fins.roll_damping_integral
    assert roll_rate(1e-3, balance) == pytest.approx(balance, rel=1e-3)
    # Without misalignment a roll decays
    assert 0.0 < roll_rate(0.0, 2.0) < 2.0


def test_fin_set_geometry_and_normal_force_slope() -> None:
    """Closed-form strip integrals, and a slope continuous through Mach 1."""
    fins = TrapezoidFinSet(
        fin_count=3,
        root_chord_m=0.2,
        tip_chord_m=0.2,
        span_m=0.1,
        sweep_length_m=0.0,
        body_radius_m=0.05,
    )
    # Rectangular fins: N * c * (outer**(n+1) - inner**(n+1)) / (n + 1)
    assert fins.roll_forcing_integral == pytest.approx(
        3 * 0.2 * (0.15**2 - 0.05**2) / 2
    )
    assert fins.roll_damping_integral == pytest.approx(
        3 * 0.2 * (0.15**3 - 0.05**3) / 3
    )
    assert fins.aspect_ratio == pytest.approx(0.5)
    assert fins.midchord_sweep_rad == pytest.approx(0.0)
    slender = np.pi * fins.aspect_ratio
    assert fins.normal_force_slope(0.999) == pytest.approx(slender, rel=1e-2)
    assert fins.normal_force_slope(1.0) == pytest.approx(slender)
    assert fins.normal_force_slope(1.001) == pytest.approx(slender)
    assert fins.normal_force_slope(0.0) < slender
    assert fins.normal_force_slope(5.0) == pytest.approx(4.0 / np.sqrt(24.0))
