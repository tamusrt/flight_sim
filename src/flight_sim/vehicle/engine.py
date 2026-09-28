"""Engine models giving the thrust and mass flow of a motor over its burn."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flight_sim.utilities.data_loader import time_interpolator_from_csv


class Engine(ABC):
    """Motor the integrator samples at each time it visits."""

    @abstractmethod
    def get_thrust(self, time: float) -> float:
        """Return the thrust in N at a time since ignition in seconds."""

    @abstractmethod
    def get_mass_flow(self, time: float, current_thrust: float) -> float:
        """Return the mass flow rate in kg/s, negative while burning.

        Args:
            time (float): Time since ignition in seconds.
            current_thrust (float): Thrust at that time in N.

        Returns:
            float: Rate of change of the vehicle mass in kg/s.
        """


@dataclass
class SolidEngine(Engine):
    """Solid motor whose mass flow is proportional to its thrust."""

    thrust_curve: Callable[[float], float]  # N from seconds since ignition
    total_propellant_mass: float  # kg
    total_impulse: float  # N*s

    def get_thrust(self, time: float) -> float:
        """Return the thrust in N at a time since ignition in seconds."""
        return float(self.thrust_curve(time))

    def get_mass_flow(self, time: float, current_thrust: float) -> float:
        """Return the mass flow rate in kg/s, negative while burning.

        Args:
            time (float): Time since ignition in seconds.
            current_thrust (float): Thrust at that time in N.

        Returns:
            float: Rate of change of the vehicle mass in kg/s, sharing the
                propellant across the burn in proportion to the thrust.
        """
        if self.total_impulse > 0:
            return -current_thrust * (self.total_propellant_mass / self.total_impulse)
        return 0.0


def solid_engine_from_csv(motor_file_path: str, propellant_mass: float) -> SolidEngine:
    """Build a solid motor from a thrust curve CSV.

    Args:
        motor_file_path (str): CSV with "Time" and "Thrust" columns, in seconds
            and newtons.
        propellant_mass (float): Propellant mass burned over the curve in kg.

    Returns:
        SolidEngine: Motor with the interpolated thrust curve and the total
            impulse integrated from the samples.
    """
    motor_data = np.genfromtxt(motor_file_path, delimiter=",", names=True)
    total_impulse = float(np.trapezoid(motor_data["Thrust"], motor_data["Time"]))
    return SolidEngine(
        thrust_curve=time_interpolator_from_csv(motor_file_path, "Time", "Thrust"),
        total_propellant_mass=propellant_mass,
        total_impulse=total_impulse,
    )
