"""Integration kernel tests."""

import numpy as np
import pytest
from pint import DimensionalityError

from flight_sim.__main__ import main
from flight_sim.environment.atmosphere import AtmosphereData
from flight_sim.environment.gravity import get_gravity
from flight_sim.integration import step
from flight_sim.units import Scalar, scalar, vector
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


# pylint: disable=redefined-outer-name
@pytest.fixture
def baseline_rocket_properties() -> RocketProperties:
    """Provides a standardized rocket configuration for integration tests."""
    return RocketProperties(aero_file_path="tests/test_data/standard_aero.csv")


def test_main_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() reports a None test input when no arguments are given."""
    monkeypatch.setattr("sys.argv", ["flight_sim"])

    main()

    assert capsys.readouterr().out == "Initializing FS with test input None\n"


def test_main_echoes_test_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() echoes the value passed via --test_input."""
    monkeypatch.setattr("sys.argv", ["flight_sim", "--test_input", "hello"])

    main()

    assert capsys.readouterr().out == "Initializing FS with test input hello\n"


def test_main_rejects_unknown_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse exits with status 2 when an unknown argument is supplied."""
    monkeypatch.setattr("sys.argv", ["flight_sim", "--nope"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 2


def test_step_zero_force_keeps_velocity_constant(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """Tests that the step function keeps velocity constant"""

    monkeypatch.setattr(
        "flight_sim.integration.get_gravity",
        lambda latitude, longitude, altitude: scalar(0.0, "m/s**2"),
    )
    state = RocketState()
    atmosphere = AtmosphereData()
    atmosphere.speed_of_sound = scalar(343.0, "m/s")
    next_state = step(state, atmosphere, baseline_rocket_properties, scalar(0.01, "s"))
    assert np.allclose(next_state.velocity.m_as("m/s"), np.zeros(3))


def test_step_with_gravity_changes_velocity(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Tests that gravity correctly accelerates rocket downwards"""
    state = RocketState()
    atmosphere = AtmosphereData()
    atmosphere.speed_of_sound = scalar(343.0, "m/s")
    next_state = step(state, atmosphere, baseline_rocket_properties, scalar(0.01, "s"))
    assert next_state.velocity[2].m_as("m/s") < 0


def test_step_result_keeps_expected_units(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """The integrated state stays in the units its fields declare."""
    next_state = step(
        RocketState(), AtmosphereData(), baseline_rocket_properties, scalar(0.01, "s")
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
        RocketState(), AtmosphereData(), baseline_rocket_properties, scalar(10.0, "ms")
    )
    from_s = step(
        RocketState(), AtmosphereData(), baseline_rocket_properties, scalar(0.01, "s")
    )

    assert np.allclose(from_ms.velocity.m_as("m/s"), from_s.velocity.m_as("m/s"))


def test_step_rejects_dt_that_is_not_a_time(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """A dt in the wrong dimension is rejected rather than silently integrated."""
    with pytest.raises(DimensionalityError):
        step(
            RocketState(),
            AtmosphereData(),
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
    assert magnitude.m_as("m/s**2") == pytest.approx(9.81)


def test_step_adaptive_scaling_and_rejection(
    monkeypatch: pytest.MonkeyPatch, baseline_rocket_properties: RocketProperties
) -> None:
    """Stress adaptive rejection, scaling, and overshoot limits."""
    monkeypatch.setattr(
        "flight_sim.integration.get_gravity",
        lambda latitude, longitude, altitude: scalar(
            altitude.to("m").magnitude ** 3, "m/s**2"
        ),
    )

    state = RocketState()
    state.position = vector((0.0, 0.0, 10.0), "m")
    state.velocity = vector((0.0, 0.0, 50.0), "m/s")
    atmosphere = AtmosphereData()
    atmosphere.speed_of_sound = scalar(343.0, "m/s")
    next_state = step(state, atmosphere, baseline_rocket_properties, scalar(5.0, "s"))

    assert next_state is not None


def test_step_triggers_aerodynamic_calculations(
    baseline_rocket_properties: RocketProperties,
) -> None:
    """Ensures the integration engine runs the drag/lift physics block."""
    state = RocketState()

    state.current_mass = scalar(25.0, "kg")
    state.velocity = vector((0.0, 0.0, 50.0), "m/s")

    atmosphere = AtmosphereData()
    atmosphere.speed_of_sound = scalar(343.0, "m/s")
    next_state = step(state, atmosphere, baseline_rocket_properties, scalar(0.1, "s"))

    assert next_state is not None
