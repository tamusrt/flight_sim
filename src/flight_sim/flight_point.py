"""Everything the equations of motion compute at one moment, in SI units.

Each ``XSI`` holds the fields of the typed ``X`` it mirrors, under the same
names, as floats and arrays in SI units: ``FlightPointSI`` mirrors
``telemetry.FlightPoint`` and ``RocketStateSI`` mirrors ``RocketState``.
Vector frames and units are those of the typed fields.
"""

from typing import NamedTuple

import numpy as np

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.utilities.data_loader import AeroCoefficients
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.mass_properties import MassPropertiesSI


class RocketStateSI(NamedTuple):
    """``RocketState`` in SI units."""

    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    orientation: Quaternion
    rail_start: np.ndarray | None


class StateDerivativeSI(NamedTuple):
    """``StateDerivative`` in SI units."""

    velocity: np.ndarray
    acceleration: np.ndarray
    angular_acceleration: np.ndarray
    orientation_derivative: Quaternion


class EnvironmentPointSI(NamedTuple):
    """``telemetry.EnvironmentPoint`` in SI units."""

    altitude: float
    atmosphere: AtmosphereConditions
    wind: np.ndarray
    gravity: float


class AeroPointSI(NamedTuple):
    """``telemetry.AeroPoint`` in SI units."""

    air_velocity: np.ndarray
    airspeed: float
    mach: float
    alpha: float
    phi: float
    dynamic_pressure: float
    coefficients: AeroCoefficients
    force: np.ndarray
    torque: np.ndarray


class FlightPointSI(NamedTuple):
    """``telemetry.FlightPoint`` in SI units."""

    time: float
    state: RocketStateSI
    derivative: StateDerivativeSI
    environment: EnvironmentPointSI
    aero: AeroPointSI
    thrust: float
    mass: MassPropertiesSI
    specific_force: np.ndarray
