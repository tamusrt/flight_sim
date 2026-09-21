"""Stores data for solid, hybrid, liquid engines"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


class Engine(ABC):
    """Initialize rocket class"""

    @abstractmethod
    def get_thrust(self, time: float) -> float:
        """Calculate the thrust produced by the engine at a given time."""
        pass

    @abstractmethod
    def get_mass_flow(self, time: float, current_thrust: float) -> float:
        """Calculate the rate of mass depletion for the engine."""
        pass


@dataclass
class SolidEngine(Engine):
    """Implementation for solid motors"""

    thrust_curve: Callable[[float], float]
    total_propellant_mass: float
    total_impulse: float

    def get_thrust(self, time: float) -> float:
        """Calculate the thrust produced by the solid engine at a given time."""
        return float(self.thrust_curve(time))

    def get_mass_flow(self, time: float, current_thrust: float) -> float:
        """Calculate the rate of mass depletion for the solid engine."""
        if self.total_impulse > 0:
            return -current_thrust * (self.total_propellant_mass / self.total_impulse)
        return 0.0
