"""Engine models giving the thrust and mass properties of a motor over its burn."""

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.units import Scalar, UnitChecked, Vector, scalar
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)


class Engine(ABC):
    """Motor the integrator samples at each time it visits."""

    @abstractmethod
    def get_thrust(self, time: float) -> float:
        """Return the thrust in N along body +X at a simulation time in seconds."""

    @abstractmethod
    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the motor's mass properties at a simulation time in seconds.

        Args:
            time (float): Simulation time in seconds.

        Returns:
            MassPropertiesSI: Casing plus remaining propellant, in rocket body
                axes.
        """


@dataclass(frozen=True)
class PropellantGrain(UnitChecked):
    """Hollow cylindrical grain along body X, burning radially out from its core."""

    mass: Annotated[Scalar, "kg"]
    length: Annotated[Scalar, "m"]
    outer_diameter: Annotated[Scalar, "m"]
    core_diameter: Annotated[Scalar, "m"]

    # From the nose tip in body axes; fixed as the grain burns
    cg_location: Annotated[Vector, "m"]

    _mass_kg: float = field(init=False, repr=False, compare=False)
    _outer_radius_sq: float = field(init=False, repr=False, compare=False)
    _core_radius_sq: float = field(init=False, repr=False, compare=False)
    _length_sq: float = field(init=False, repr=False, compare=False)
    _cg_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the geometry and cache it in SI units.

        Raises:
            ValueError: If the mass is negative, the length is not positive, or
                the core is not within 0 and the outer diameter.
        """
        super().__post_init__()
        mass = float(self.mass.m_as("kg"))
        length = float(self.length.m_as("m"))
        outer = float(self.outer_diameter.m_as("m"))
        core = float(self.core_diameter.m_as("m"))
        cg_m = np.array(self.cg_location.m_as("m"), dtype=float)
        if mass < 0.0:
            raise ValueError(f"Grain mass must not be negative, got {mass} kg")
        if length <= 0.0:
            raise ValueError(f"Grain length must be positive, got {length} m")
        if not 0.0 <= core < outer:
            raise ValueError(
                f"Grain core diameter must be in [0, {outer}) m, got {core} m"
            )
        if cg_m.shape != (3,):
            raise ValueError(f"Expected a 3-vector CG, got shape {cg_m.shape}")
        cg_m.flags.writeable = False
        object.__setattr__(self, "_mass_kg", mass)
        object.__setattr__(self, "_outer_radius_sq", (outer / 2) ** 2)
        object.__setattr__(self, "_core_radius_sq", (core / 2) ** 2)
        object.__setattr__(self, "_length_sq", length**2)
        object.__setattr__(self, "_cg_m", cg_m)

    def mass_properties(self, burned_fraction: float) -> MassPropertiesSI:
        """Return the mass properties of the propellant left after part of the burn.

        Args:
            burned_fraction (float): Fraction of the propellant burned, in [0, 1].

        Returns:
            MassPropertiesSI: The remaining hollow cylinder, about its own CG.
        """
        remaining = 1.0 - burned_fraction
        outer_sq = self._outer_radius_sq
        # The core widens until the remaining volume matches the remaining mass
        inner_sq = outer_sq - remaining * (outer_sq - self._core_radius_sq)
        mass = self._mass_kg * remaining
        axial = 0.5 * mass * (outer_sq + inner_sq)
        transverse = mass * (3.0 * (outer_sq + inner_sq) + self._length_sq) / 12.0
        return MassPropertiesSI(
            mass, self._cg_m, np.diag((axial, transverse, transverse))
        )


@dataclass(frozen=True)
class SolidEngine(Engine, UnitChecked):
    """Solid motor burning its propellant in proportion to the impulse delivered.

    The thrust is linear between the samples of the thrust curve and zero
    outside them.
    """

    times: np.ndarray  # s from ignition, strictly increasing
    thrusts: np.ndarray  # N at each of the times
    grain: PropellantGrain
    casing: MassProperties  # Everything but the propellant

    ignition_time: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(0.0, "s")
    )

    _ignition_s: float = field(init=False, repr=False, compare=False)
    _sample_times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _sample_thrusts: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _slopes: tuple[float, ...] = field(init=False, repr=False, compare=False)

    # Impulse delivered by each sample time, in N*s
    _impulses: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the thrust curve and cache its cumulative impulse.

        Raises:
            ValueError: If the curve has fewer than two samples, mismatched
                lengths, non-increasing times or negative thrust, or the
                casing has no mass.
        """
        super().__post_init__()
        times = np.array(self.times, dtype=float)
        thrusts = np.array(self.thrusts, dtype=float)
        if times.ndim != 1 or times.shape != thrusts.shape or len(times) < 2:
            raise ValueError(
                "Expected matching 1-D times and thrusts of at least two samples,"
                f" got shapes {times.shape} and {thrusts.shape}"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("Thrust curve times must be strictly increasing")
        if np.any(thrusts < 0.0):
            raise ValueError("Thrust curve must not be negative")
        if not self.casing.si.mass > 0.0:
            raise ValueError("Casing mass must be positive")
        intervals = np.diff(times)
        impulses = np.concatenate(
            ([0.0], np.cumsum(0.5 * (thrusts[:-1] + thrusts[1:]) * intervals))
        )
        times.flags.writeable = False
        thrusts.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "thrusts", thrusts)
        object.__setattr__(self, "_ignition_s", float(self.ignition_time.m_as("s")))
        object.__setattr__(self, "_sample_times", tuple(times.tolist()))
        object.__setattr__(self, "_sample_thrusts", tuple(thrusts.tolist()))
        object.__setattr__(
            self, "_slopes", tuple((np.diff(thrusts) / intervals).tolist())
        )
        object.__setattr__(self, "_impulses", tuple(impulses.tolist()))

    @property
    def total_impulse(self) -> float:
        """Return the impulse of the whole thrust curve in N*s."""
        return self._impulses[-1]

    def get_thrust(self, time: float) -> float:
        """Return the thrust in N along body +X at a simulation time in seconds."""
        return float(
            np.interp(
                time - self._ignition_s, self.times, self.thrusts, left=0.0, right=0.0
            )
        )

    def burned_fraction(self, time: float) -> float:
        """Return the fraction of the propellant burned by a simulation time.

        Args:
            time (float): Simulation time in seconds.

        Returns:
            float: Impulse delivered by ``time`` over the total impulse, exact
                for the linear thrust curve; 0 for a motor with no impulse.
        """
        total = self._impulses[-1]
        burn_time = time - self._ignition_s
        times = self._sample_times
        if total <= 0.0 or burn_time <= times[0]:
            return 0.0
        if burn_time >= times[-1]:
            return 1.0
        index = bisect.bisect_right(times, burn_time) - 1
        into = burn_time - times[index]
        impulse = self._impulses[index] + into * (
            self._sample_thrusts[index] + 0.5 * self._slopes[index] * into
        )
        return impulse / total

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the casing plus remaining propellant at a simulation time in s."""
        return combine(
            self.casing.si, self.grain.mass_properties(self.burned_fraction(time))
        )


def solid_engine_from_csv(
    motor_file_path: str, grain: PropellantGrain, casing: MassProperties
) -> SolidEngine:
    """Build a solid motor from a thrust curve CSV.

    Args:
        motor_file_path (str): CSV with "Time" and "Thrust" columns, in seconds
            from ignition and newtons.
        grain (PropellantGrain): Propellant burned over the curve.
        casing (MassProperties): Everything in the motor but the propellant.

    Returns:
        SolidEngine: Motor igniting at time zero.
    """
    motor_data = pd.read_csv(motor_file_path)
    return SolidEngine(
        times=motor_data["Time"].to_numpy(dtype=float),
        thrusts=motor_data["Thrust"].to_numpy(dtype=float),
        grain=grain,
        casing=casing,
    )
