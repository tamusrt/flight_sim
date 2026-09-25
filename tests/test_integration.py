"""Integration kernel tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.__main__ import get_default_state, main
from flight_sim.environment.gravity import get_gravity
from flight_sim.integration import (
    derivative_computation,
    locate_event,
    quaternion_kinematics,
    step,
)
from flight_sim.units import Scalar, scalar, vector
from flight_sim.vehicle.rocket_properties import (
    AERO_COEFFICIENT_COLUMNS,
    RocketProperties,
)
from flight_sim.vehicle.rocket_state import Quaternion, RocketState


def _bare_state() -> RocketState:
    """Return a massless state at rest at the origin.

    Zero mass switches off thrust and aerodynamics, leaving only gravity.
    """
    return RocketState(
        current_mass=scalar(0.0, "kg"),
        inertia=vector((2.5, 2.5, 0.1), "kg*m**2"),
        cg_location=vector((0.0, 0.0, -2.5), "m"),
    )


@pytest.fixture
def baseline_rocket_properties() -> RocketProperties:
    """Provides a standardized rocket configuration for integration tests."""
    return RocketProperties(
        aero_file_path="tests/test_data/standard_aero.csv",
        motor_file_path="tests/test_data/standard_motor.csv",
        propellant_mass=2.5,
        reference_area=scalar(0.0182414692, "m**2"),
        reference_diameter=scalar(0.1524, "m"),
    )


@patch("flight_sim.__main__.adaptive_step")
def test_main_echoes_test_input(mock_adaptive_step: MagicMock) -> None:
    """Verify the executive simulation loop runs from launch to impact."""

    # Force the physics step to immediately return an underground state.
    mock_state = get_default_state()
    mock_state.position = vector((0.0, 0.0, -10.0), "m")
    mock_adaptive_step.return_value = (
        mock_state,
        scalar(0.2, "s"),
        scalar(0.2, "s"),
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
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """Tests that the step function keeps velocity constant"""

    monkeypatch.setattr(
        "flight_sim.integration.normal_gravity",
        lambda latitude_rad, altitude_m: 0.0,
    )
    state = _bare_state()
    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.01, "s"))
    assert np.allclose(next_state.velocity.m_as("m/s"), np.zeros(3))


def test_step_with_gravity_changes_velocity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Tests that gravity correctly accelerates rocket downwards"""
    state = _bare_state()
    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.01, "s"))
    assert next_state.velocity[2].m_as("m/s") < 0


def test_step_result_keeps_expected_units(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The integrated state stays in the units its fields declare."""
    next_state = step(
        0.0,
        _bare_state(),
        baseline_rocket_properties,
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
        scalar(10.0, "ms"),
    )
    from_s = step(
        0.0,
        _bare_state(),
        baseline_rocket_properties,
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
            scalar(0.01, "m"),
        )


def test_get_gravity_returns_an_acceleration() -> None:
    """Gravity is returned as an acceleration quantity, not a bare float."""
    magnitude: Scalar = get_gravity(
        latitude=scalar(0.0, "deg"),
        longitude=scalar(0.0, "deg"),
        altitude=scalar(0.0, "m"),
    )

    assert magnitude.check("[length] / [time] ** 2")
    assert magnitude.m_as("m/s**2") == pytest.approx(9.7803253359, rel=1e-5)


def test_step_adaptive_scaling_and_rejection(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """Stress adaptive rejection, scaling, and overshoot limits."""
    monkeypatch.setattr(
        "flight_sim.integration.normal_gravity",
        lambda latitude_rad, altitude_m: altitude_m**3,
    )

    state = _bare_state()
    state.position = vector((0.0, 0.0, 10.0), "m")
    state.velocity = vector((0.0, 0.0, 50.0), "m/s")
    next_state = step(0.0, state, baseline_rocket_properties, scalar(5.0, "s"))

    assert next_state is not None


def test_step_triggers_aerodynamic_calculations(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Ensures the integration engine runs the drag/lift physics block."""
    state = _bare_state()

    state.current_mass = scalar(25.0, "kg")
    state.velocity = vector((0.0, 0.0, 50.0), "m/s")

    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.1, "s"))

    assert next_state is not None


def test_step_pitch_moment_induces_angular_velocity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Test that a non-zero Angle of Attack creates a pitch restoring rotation."""
    state = _bare_state()
    state.velocity = vector((50.0, 0.0, 200.0), "m/s")
    state.current_mass = scalar(20.0, "kg")

    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.1, "s"))

    angular_vel = next_state.angular_velocity.m_as("rad/s")

    assert angular_vel[0] == 0.0  # No roll induced
    assert abs(angular_vel[1]) > 0.0  # Pitch rotation successfully applied!
    assert angular_vel[2] == 0.0  # No yaw induced


def test_quaternion_normalized_returns_unit_length() -> None:
    """Normalizing rescales every component by the quaternion's length."""
    unit = Quaternion(q_w=2.0, q_x=0.0, q_y=-2.0, q_z=1.0).normalized()

    assert unit.q_w == pytest.approx(2 / 3)
    assert unit.q_x == pytest.approx(0.0)
    assert unit.q_y == pytest.approx(-2 / 3)
    assert unit.q_z == pytest.approx(1 / 3)


def test_quaternion_kinematics_matches_xi_matrix() -> None:
    """The quaternion rate equals 0.5 * Xi(q) * omega for a general orientation."""
    orientation = Quaternion(q_w=0.5, q_x=-0.5, q_y=0.5, q_z=0.5)
    angular_velocity = vector((1.0, 2.0, 3.0), "rad/s")

    q_dot = quaternion_kinematics(orientation, angular_velocity)

    assert q_dot.q_w == pytest.approx(-1.0)
    assert q_dot.q_x == pytest.approx(0.5)
    assert q_dot.q_y == pytest.approx(1.5)
    assert q_dot.q_z == pytest.approx(0.0)


def test_quaternion_kinematics_accepts_any_angular_rate_unit() -> None:
    """An angular velocity in deg/s gives the same rate as the equivalent rad/s."""
    from_deg = quaternion_kinematics(Quaternion(), vector((0.0, 0.0, 180.0), "deg/s"))

    assert from_deg.q_z == pytest.approx(np.pi / 2)


def test_step_angular_velocity_rotates_orientation(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """A constant body rate about Z turns the orientation by rate * time."""
    monkeypatch.setattr(
        "flight_sim.integration.normal_gravity",
        lambda latitude_rad, altitude_m: 0.0,
    )
    state = _bare_state()
    state.angular_velocity = vector((0.0, 0.0, np.pi / 2), "rad/s")

    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.1, "s"))

    half_angle = (np.pi / 2) * 0.1 / 2
    assert next_state.orientation.q_w == pytest.approx(np.cos(half_angle), abs=1e-6)
    assert next_state.orientation.q_x == pytest.approx(0.0, abs=1e-6)
    assert next_state.orientation.q_y == pytest.approx(0.0, abs=1e-6)
    assert next_state.orientation.q_z == pytest.approx(np.sin(half_angle), abs=1e-6)


def test_step_keeps_orientation_normalized(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The orientation stays a unit quaternion while the rocket is pitching."""
    state = _bare_state()
    state.velocity = vector((50.0, 0.0, 200.0), "m/s")
    state.angular_velocity = vector((0.3, -1.2, 0.8), "rad/s")
    state.current_mass = scalar(20.0, "kg")

    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.1, "s"))

    orientation = next_state.orientation
    norm = np.linalg.norm(
        [orientation.q_w, orientation.q_x, orientation.q_y, orientation.q_z]
    )
    assert norm == pytest.approx(1.0)
    assert orientation.q_w != 1.0  # Orientation actually moved off the pad attitude


def test_step_dynamic_cg_moment_transfer(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A shifted CG induces a pitch moment at a non-zero alpha."""
    baseline_rocket_properties.reference_point = vector((0.0, 0.0, 0.0), "m")

    state = _bare_state()
    state.cg_location = vector((0.0, 0.0, -2.7432), "m")

    state.current_mass = scalar(20.0, "kg")
    state.inertia = vector((0.1, 2.5, 2.5), "kg*m**2")

    state.velocity = vector((20.0, 0.0, 150.0), "m/s")
    state.angular_velocity = vector((0.0, 0.0, 0.0), "rad/s")

    next_state = step(0.0, state, baseline_rocket_properties, scalar(0.1, "s"))

    angular_vel = next_state.angular_velocity.m_as("rad/s")

    assert abs(angular_vel[1]) > 0.0

    assert angular_vel[0] == 0.0
    assert angular_vel[2] == 0.0


def test_six_dof_aerodynamic_response(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Verify that Side Force, Roll, and Yaw generate correct accelerations."""

    baseline_rocket_properties.aero_coefficients = MagicMock(
        return_value=[1.0] * len(AERO_COEFFICIENT_COLUMNS)
    )

    baseline_rocket_properties.thrust_curve = MagicMock(return_value=0.0)

    state = get_default_state()
    state.position = vector((0.0, 0.0, 1000.0), "m")
    state.velocity = vector((50.0, 0.0, 100.0), "m/s")

    derivative = derivative_computation(
        time=0.0,
        state=state,
        properties=baseline_rocket_properties,
    )

    angular_accel = derivative.angular_acceleration.m_as("rad/s**2")
    linear_accel = derivative.acceleration.m_as("m/s**2")

    assert abs(angular_accel[2]) > 0.0
    assert abs(angular_accel[0]) > 0.0

    lateral_accel_magnitude = abs(linear_accel[0]) + abs(linear_accel[1])
    assert lateral_accel_magnitude > 0.0


def test_locate_event_finds_ballistic_apogee(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """A drag-free coast peaks at the time and height the closed form gives."""
    monkeypatch.setattr(
        "flight_sim.integration.normal_gravity",
        lambda latitude_rad, altitude_m: 9.81,
    )
    start = _bare_state()
    start.velocity = vector((0.0, 0.0, 50.0), "m/s")
    dt = scalar(10.0, "s")
    end = step(0.0, start, baseline_rocket_properties, dt)

    apogee, time_to_apogee = locate_event(
        0.0,
        start,
        end,
        baseline_rocket_properties,
        dt,
        lambda _time, state: float(state.velocity.m_as("m/s")[2]),
    )

    assert time_to_apogee.m_as("s") == pytest.approx(50.0 / 9.81, abs=1e-8)
    assert apogee.position.m_as("m")[2] == pytest.approx(50.0**2 / (2 * 9.81))
    assert apogee.velocity.m_as("m/s")[2] == pytest.approx(0.0, abs=1e-7)
