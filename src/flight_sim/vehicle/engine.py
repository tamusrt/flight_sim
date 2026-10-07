"""Engine models giving the thrust and mass properties of a motor over its burn."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Self
from xml.etree import ElementTree

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
    def get_thrust(self, time: float, ambient_pressure: float) -> float:
        """Return the thrust along body +X.

        Args:
            time (float): Simulation time in seconds.
            ambient_pressure (float): Air pressure around the nozzle in Pa.

        Returns:
            float: Thrust in N.
        """

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
        """Check the core fits in the grain and cache the geometry in SI units.

        Raises:
            ValueError: If the core is not within 0 and the outer diameter.
        """
        super().__post_init__()
        outer = float(self.outer_diameter.m_as("m"))
        core = float(self.core_diameter.m_as("m"))
        if not 0.0 <= core < outer:
            raise ValueError(
                f"Grain core diameter must be in [0, {outer}) m, got {core} m"
            )
        object.__setattr__(self, "_mass_kg", float(self.mass.m_as("kg")))
        object.__setattr__(self, "_outer_radius_sq", (outer / 2) ** 2)
        object.__setattr__(self, "_core_radius_sq", (core / 2) ** 2)
        object.__setattr__(self, "_length_sq", float(self.length.m_as("m")) ** 2)
        object.__setattr__(self, "_cg_m", self.cg_location.m_as("m"))

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

    The thrust curve is linear between its samples and zero outside them.
    During the burn, the nozzle adds (reference - ambient pressure) * exit
    area to it.
    """

    times: np.ndarray  # s from ignition, strictly increasing
    thrusts: np.ndarray  # N at each of the times
    grain: PropellantGrain
    casing: MassProperties  # Everything but the propellant

    ignition_time: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(0.0, "s")
    )

    nozzle_exit_area: Annotated[Scalar, "m**2"] = field(
        default_factory=lambda: scalar(0.0, "m**2")
    )

    # Ambient pressure the thrust curve was measured at
    reference_pressure: Annotated[Scalar, "Pa"] = field(
        default_factory=lambda: scalar(101325.0, "Pa")
    )

    _ignition_s: float = field(init=False, repr=False, compare=False)
    _exit_area_m2: float = field(init=False, repr=False, compare=False)
    _reference_pressure_pa: float = field(init=False, repr=False, compare=False)

    # Impulse delivered by each sample time, in N*s
    _impulses: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the thrust curve and cache its cumulative impulse.

        Raises:
            ValueError: If the times are not strictly increasing, the curve
                delivers no impulse or the nozzle exit area is negative.
        """
        super().__post_init__()
        intervals = np.diff(self.times)
        if np.any(intervals <= 0.0):
            raise ValueError("Thrust curve times must be strictly increasing")
        impulses = np.concatenate(
            ([0.0], np.cumsum(0.5 * (self.thrusts[:-1] + self.thrusts[1:]) * intervals))
        )
        if not impulses[-1] > 0.0:
            raise ValueError("Thrust curve must deliver some impulse")
        exit_area = float(self.nozzle_exit_area.m_as("m**2"))
        if exit_area < 0.0:
            raise ValueError(
                f"Nozzle exit area must be non-negative, got {exit_area} m**2"
            )
        object.__setattr__(self, "_ignition_s", float(self.ignition_time.m_as("s")))
        object.__setattr__(self, "_exit_area_m2", exit_area)
        object.__setattr__(
            self, "_reference_pressure_pa", float(self.reference_pressure.m_as("Pa"))
        )
        object.__setattr__(self, "_impulses", impulses)

    @classmethod
    def from_eng(
        cls, motor_file_path: str, grain: PropellantGrain, casing: MassProperties
    ) -> Self:
        """Build a solid motor from the first motor in a RASP .eng file.

        Args:
            motor_file_path (str): RASP file whose header line is followed by
                "time thrust" lines, in seconds from ignition and newtons.
            grain (PropellantGrain): Propellant burned over the curve.
            casing (MassProperties): Everything in the motor but the propellant.

        Returns:
            SolidEngine: Motor igniting at time zero.

        Raises:
            ValueError: If the file holds no thrust curve.
        """
        with open(motor_file_path, encoding="utf-8", errors="replace") as file:
            # Drop ";" comments and blank lines
            lines = [fields for line in file if (fields := line.split(";")[0].split())]
        samples = []
        # The curve follows the header and ends at the next motor's header
        for fields in lines[1:]:
            try:
                samples.append((float(fields[0]), float(fields[1])))
            except (ValueError, IndexError):
                break
        return cls._from_samples(samples, grain, casing)

    @classmethod
    def from_rse(
        cls, motor_file_path: str, grain: PropellantGrain, casing: MassProperties
    ) -> Self:
        """Build a solid motor from the first motor in a RockSim .rse file.

        Args:
            motor_file_path (str): RockSim XML file whose "eng-data" elements
                give the time "t" in seconds from ignition and thrust "f" in N.
            grain (PropellantGrain): Propellant burned over the curve.
            casing (MassProperties): Everything in the motor but the propellant.

        Returns:
            SolidEngine: Motor igniting at time zero.

        Raises:
            ValueError: If the file holds no thrust curve.
        """
        motor = ElementTree.parse(motor_file_path).find(".//engine")
        samples = (
            []
            if motor is None
            else [
                (float(point.attrib["t"]), float(point.attrib["f"]))
                for point in motor.iter("eng-data")
            ]
        )
        return cls._from_samples(samples, grain, casing)

    @classmethod
    def _from_samples(
        cls,
        samples: list[tuple[float, float]],
        grain: PropellantGrain,
        casing: MassProperties,
    ) -> Self:
        """Build a motor from (time, thrust) samples, adding (0, 0) if they start later.

        Raises:
            ValueError: If there are no samples.
        """
        if not samples:
            raise ValueError("Motor file holds no thrust curve")
        if samples[0][0] > 0.0:
            samples = [(0.0, 0.0), *samples]
        times, thrusts = np.array(samples, dtype=float).T
        return cls(times=times, thrusts=thrusts, grain=grain, casing=casing)

    @property
    def total_impulse(self) -> float:
        """Return the impulse of the whole thrust curve in N*s."""
        return float(self._impulses[-1])

    def get_thrust(self, time: float, ambient_pressure: float) -> float:
        """Return the thrust in N along body +X, never negative.

        Args:
            time (float): Simulation time in seconds.
            ambient_pressure (float): Air pressure around the nozzle in Pa.

        Returns:
            float: Curve thrust plus pressure thrust during the burn, else zero.
        """
        burn_time = time - self._ignition_s
        if not self.times[0] <= burn_time <= self.times[-1]:
            return 0.0
        curve = float(np.interp(burn_time, self.times, self.thrusts))
        pressure_thrust = (
            self._reference_pressure_pa - ambient_pressure
        ) * self._exit_area_m2
        return max(curve + pressure_thrust, 0.0)

    def burned_fraction(self, time: float) -> float:
        """Return the fraction of the propellant burned by a simulation time.

        Args:
            time (float): Simulation time in seconds.

        Returns:
            float: Impulse delivered by ``time`` over the total impulse, exact
                for the linear thrust curve.
        """
        burn_time = time - self._ignition_s
        times, thrusts = self.times, self.thrusts
        if burn_time <= times[0]:
            return 0.0
        if burn_time >= times[-1]:
            return 1.0
        index = int(np.searchsorted(times, burn_time, side="right")) - 1
        into = burn_time - times[index]
        slope = (thrusts[index + 1] - thrusts[index]) / (
            times[index + 1] - times[index]
        )
        impulse = self._impulses[index] + into * (thrusts[index] + 0.5 * slope * into)
        return float(impulse / self._impulses[-1])

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
