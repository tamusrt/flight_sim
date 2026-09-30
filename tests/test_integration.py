"""Integration kernel tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.__main__ import get_default_state, main
from flight_sim.environment.atmosphere import StandardAtmosphere1976, VacuumAtmosphere
from flight_sim.environment.gravity import ConstantGravity, GravityModel, WGS84Gravity
from flight_sim.events import IMPACT
from flight_sim.integration import (
    IntegrationConfiguration,
    derivative_computation,
    locate_event,
    step,
)
from flight_sim.units import scalar, vector
from flight_sim.utilities.data_loader import AeroCoefficients
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_CONFIG = IntegrationConfiguration()


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
    mock_state = get_default_state()
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
    config = IntegrationConfiguration(gravity=ConstantGravity(0.0))
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

    config = IntegrationConfiguration(gravity=_CubicGravity())

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


def test_step_pitch_moment_induces_angular_velocity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Test that a non-zero Angle of Attack creates a pitch restoring rotation."""
    state = _bare_state()
    state.velocity = vector((200.0, 50.0, 0.0), "m/s")
    state.current_mass = scalar(20.0, "kg")

    next_state = step(0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.1, "s"))

    angular_vel = next_state.angular_velocity.m_as("rad/s")

    # X and Y carry only rounding residue from 270 degrees landing off its grid line
    assert angular_vel[0] == pytest.approx(0.0, abs=1e-12)  # No roll about the nose
    assert angular_vel[1] == pytest.approx(0.0, abs=1e-12)  # No torque out of XY
    assert abs(angular_vel[2]) > 0.0  # Pitch rotation successfully applied!


def test_step_angular_velocity_rotates_orientation(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A constant body rate about X turns the orientation by rate * time."""
    config = IntegrationConfiguration(gravity=ConstantGravity(0.0))
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


def test_step_dynamic_cg_moment_transfer(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A shifted CG induces a pitch moment at a non-zero alpha."""
    baseline_rocket_properties.reference_point = vector((0.0, 0.0, 0.0), "m")

    state = _bare_state()
    state.cg_location = vector((-2.7432, 0.0, 0.0), "m")

    state.current_mass = scalar(20.0, "kg")
    state.inertia = vector((0.1, 2.5, 2.5), "kg*m**2")

    state.velocity = vector((150.0, 20.0, 0.0), "m/s")
    state.angular_velocity = vector((0.0, 0.0, 0.0), "rad/s")

    next_state = step(0.0, state, baseline_rocket_properties, _CONFIG, scalar(0.1, "s"))

    angular_vel = next_state.angular_velocity.m_as("rad/s")

    assert abs(angular_vel[2]) > 0.0

    # X and Y carry only rounding residue from 270 degrees landing off its grid line
    assert angular_vel[0] == pytest.approx(0.0, abs=1e-12)
    assert angular_vel[1] == pytest.approx(0.0, abs=1e-12)


def test_six_dof_aerodynamic_response(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Verify that all six coefficients drive every axis."""

    baseline_rocket_properties.aero_coefficients = MagicMock(
        return_value=AeroCoefficients(cx=1.0, cy=1.0, cz=1.0, cmx=1.0, cmy=1.0, cmz=1.0)
    )

    baseline_rocket_properties.engine = MagicMock(
        get_thrust=MagicMock(return_value=0.0),
        get_mass_flow=MagicMock(return_value=0.0),
    )

    state = get_default_state()
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
    baseline_rocket_properties.aero_coefficients = MagicMock(
        return_value=AeroCoefficients(cx=1.0, cy=1.0, cz=1.0, cmx=1.0, cmy=1.0, cmz=1.0)
    )
    baseline_rocket_properties.engine = MagicMock(
        get_thrust=MagicMock(return_value=0.0),
        get_mass_flow=MagicMock(return_value=0.0),
    )
    state = get_default_state()
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
    config = IntegrationConfiguration(gravity=ConstantGravity(9.81))
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
        dt,
        lambda _time, state: float(state.velocity.m_as("m/s")[0]),
    )

    assert time_to_apogee.m_as("s") == pytest.approx(50.0 / 9.81, abs=1e-8)
    assert apogee.position.m_as("m")[0] == pytest.approx(50.0**2 / (2 * 9.81))
    assert apogee.velocity.m_as("m/s")[0] == pytest.approx(0.0, abs=1e-7)


def test_step_warns_at_minimum_step_length(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """A step whose error never meets tolerance is accepted at the minimum length."""
    config = IntegrationConfiguration(
        position_tolerance=scalar(1e-7, "m"), min_time_step=scalar(1.0, "ns")
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
        atmosphere=VacuumAtmosphere(), gravity=ConstantGravity(0.0)
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
    config = IntegrationConfiguration(launch_latitude=scalar(90.0, "deg"))

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
    state = get_default_state()
    state.position = vector((1000.0, 0.0, 0.0), "m")
    coasting = 100.0

    still_air = IntegrationConfiguration(
        atmosphere=StandardAtmosphere1976(), gravity=ConstantGravity(0.0)
    )
    in_still_air = derivative_computation(coasting, state, properties, still_air)
    assert in_still_air.acceleration.m_as("m/s**2") == pytest.approx(np.zeros(3))

    windy = IntegrationConfiguration(
        atmosphere=StandardAtmosphere1976(wind_m_s=np.array([0.0, 20.0, 0.0])),
        gravity=ConstantGravity(0.0),
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
    config = IntegrationConfiguration(gravity=ConstantGravity(0.0))
    nose = np.array([1.0, 0.0, 0.0])
    speed, alpha_tot = 150.0, np.radians(5.0)
    coasting = 100.0

    def loads(crossflow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the accelerations of the pad-attitude rocket in one crossflow."""
        state = get_default_state()
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

    # The +Y loads rotated with the crossflow
    assert abs(reference_linear[1]) > 0.0
    assert abs(reference_angular[2]) > 0.0
    assert linear == pytest.approx(
        reference_linear[0] * nose + reference_linear[1] * crossflow, abs=1e-9
    )
    assert angular == pytest.approx(
        reference_angular[2] * np.cross(nose, crossflow), abs=1e-9
    )


def test_configuration_rejects_wrong_units() -> None:
    """A tolerance that is not a length is rejected on construction."""
    with pytest.raises(DimensionalityError, match="position_tolerance"):
        IntegrationConfiguration(position_tolerance=scalar(1.0, "s"))
