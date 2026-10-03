"""Fixtures shared across the test modules."""

import numpy as np
import pytest

from flight_sim.__main__ import get_default_properties
from flight_sim.units import matrix, scalar, vector, zero_vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.vehicle.engine import PropellantGrain, SolidEngine
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties


@pytest.fixture(name="baseline_rocket_properties")
def _baseline_rocket_properties() -> RocketProperties:
    """Provides the standard rocket with the hand-made test aero table.

    The table's center of pressure is 3.048 m aft of the nose tip.
    """
    properties = get_default_properties()
    properties.aero_table = aero_table_from_csv(
        "tests/test_data/standard_aero.csv",
        reference_area=scalar(0.0182414692, "m**2"),
        reference_length=scalar(0.1524, "m"),
        reference_point=zero_vector("m"),
    )
    return properties


@pytest.fixture(name="unpowered_rocket_properties")
def _unpowered_rocket_properties(
    baseline_rocket_properties: RocketProperties,
) -> RocketProperties:
    """Provides the test-table rocket with a motor that never fires.

    Every part's CG is 2.5 m aft of the nose tip, so the 20 kg rocket's mass
    properties are fixed and its inertia diagonal.
    """
    cg_location = vector((-2.5, 0.0, 0.0), "m")
    engine = SolidEngine(
        times=np.array([0.0, 1.0]),
        thrusts=np.zeros(2),
        grain=PropellantGrain(
            mass=scalar(2.0, "kg"),
            length=scalar(0.5, "m"),
            outer_diameter=scalar(0.06, "m"),
            core_diameter=scalar(0.02, "m"),
            cg_location=cg_location,
        ),
        casing=MassProperties(
            scalar(2.0, "kg"), cg_location, matrix(np.zeros((3, 3)), "kg*m**2")
        ),
    )
    return RocketProperties(
        aero_table=baseline_rocket_properties.aero_table,
        engine=engine,
        dry_mass_properties=MassProperties(
            scalar(16.0, "kg"),
            cg_location,
            matrix(np.diag((0.1, 2.5, 2.5)), "kg*m**2"),
        ),
    )
