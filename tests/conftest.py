"""Fixtures shared across the test modules."""

import matplotlib
import numpy as np
import pytest

from flight_sim.units import matrix, scalar, vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.vehicle.engine import Engine, PropellantGrain, solid_engine_from_csv
from flight_sim.vehicle.mass_properties import MassProperties, MassPropertiesSI
from flight_sim.vehicle.rocket_properties import RocketProperties

# Tests draw figures without a display
matplotlib.use("Agg")


@pytest.fixture(name="baseline_rocket_properties")
def _baseline_rocket_properties() -> RocketProperties:
    """Provides the standard rocket with the hand-made test aero table.

    The table's center of pressure is 3.048 m aft of the nose tip. The motor
    is the standard test curve, 25 kg at ignition burning down to 20 kg.
    """
    grain = PropellantGrain(
        mass=scalar(5.0, "kg"),
        length=scalar(0.75, "m"),
        outer_diameter=scalar(3.0, "in"),
        core_diameter=scalar(30.0, "mm"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
    )
    # Thin tube of radius 0.045 m and length 0.9 m
    casing = MassProperties(
        mass=scalar(3.0, "kg"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
        inertia=matrix(np.diag((0.0061, 0.21, 0.21)), "kg*m**2"),
    )
    return RocketProperties(
        aero_table=aero_table_from_csv(
            "tests/test_data/standard_aero.csv",
            reference_area=scalar(0.0182414692, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=vector((0.0, 0.0, 0.0), "m"),
            frame="body",
            extrapolation="linear",
        ),
        engine=solid_engine_from_csv(
            "tests/test_data/standard_motor.csv",
            grain,
            casing,
            nozzle_exit_area=scalar(0.0, "m**2"),
            reference_pressure=scalar(101325.0, "Pa"),
        ),
        dry_mass_properties=MassProperties(
            mass=scalar(17.0, "kg"),
            cg_location=vector((-1.0, 0.0, 0.0), "m"),
            inertia=matrix(np.diag((2.4, 135.0, 135.0)), "kg*m**2"),
        ),
    )


class _InertEngine(Engine):
    """Massless motor that never fires."""

    def get_thrust(self, time: float, ambient_pressure: float) -> float:
        return 0.0

    def mass_properties(self, time: float) -> MassPropertiesSI:
        return MassPropertiesSI(0.0, np.zeros(3), np.zeros((3, 3)))

    def burnout_time(self) -> float:
        return 0.0


@pytest.fixture(name="unpowered_rocket_properties")
def _unpowered_rocket_properties(
    baseline_rocket_properties: RocketProperties,
) -> RocketProperties:
    """Provides the test-table rocket as a fixed 20 kg body with no motor.

    Its CG is 2.5 m aft of the nose tip and its inertia is diag(0.1, 2.5, 2.5).
    """
    return RocketProperties(
        aero_table=baseline_rocket_properties.aero_table,
        engine=_InertEngine(),
        dry_mass_properties=MassProperties(
            scalar(20.0, "kg"),
            vector((-2.5, 0.0, 0.0), "m"),
            matrix(np.diag((0.1, 2.5, 2.5)), "kg*m**2"),
        ),
    )
