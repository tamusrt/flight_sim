"""Engine model tests."""

import numpy as np
import pytest

from flight_sim.vehicle.engine import SolidEngine, solid_engine_from_csv


def test_solid_engine_mass_flow_integrates_to_the_propellant_mass() -> None:
    """Burning a constant thrust over the burn time consumes the propellant."""
    burn_time = 4.0
    thrust = 500.0
    engine = SolidEngine(
        thrust_curve=lambda time: thrust if 0.0 <= time <= burn_time else 0.0,
        total_propellant_mass=2.0,
        total_impulse=thrust * burn_time,
    )
    times = np.linspace(0.0, burn_time, 401)
    mass_flow = [engine.get_mass_flow(time, engine.get_thrust(time)) for time in times]

    assert engine.get_thrust(1.0) == thrust
    assert engine.get_thrust(burn_time + 1.0) == 0.0
    assert np.trapezoid(mass_flow, times) == pytest.approx(-2.0)


def test_solid_engine_without_impulse_burns_nothing() -> None:
    """A motor with no impulse reports no mass flow."""
    engine = SolidEngine(
        thrust_curve=lambda time: 0.0, total_propellant_mass=1.0, total_impulse=0.0
    )

    assert engine.get_mass_flow(0.0, 0.0) == 0.0


def test_solid_engine_from_csv_integrates_the_thrust_curve() -> None:
    """The total impulse is the area under the CSV thrust curve."""
    engine = solid_engine_from_csv("tests/test_data/standard_motor.csv", 2.5)

    # 2000 N for 2 s, then a linear tail-off to zero over 2 s
    assert engine.total_impulse == pytest.approx(6000.0)
    assert engine.get_thrust(1.0) == pytest.approx(2000.0)
    assert engine.get_mass_flow(1.0, 2000.0) == pytest.approx(-2000.0 * 2.5 / 6000.0)
