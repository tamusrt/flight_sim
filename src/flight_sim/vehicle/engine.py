"""Engine models giving the thrust and mass properties of a motor over its burn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Self
from xml.etree import ElementTree

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.units import Scalar, UnitChecked, Vector, matrix, scalar, vector
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)


def _motor_parts(
    diameter_mm: float,
    length_mm: float,
    propellant_mass_kg: float,
    total_mass_kg: float,
    cg_mm: float | None = None,
) -> tuple[PropellantGrain, MassProperties]:
    """Build propellant and casing properties from motor metadata."""
    if diameter_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("Motor header dimensions must be positive")

    casing_mass_kg = total_mass_kg - propellant_mass_kg

    if propellant_mass_kg < 0.0 or casing_mass_kg < 0.0:
        raise ValueError("Motor header masses must be non-negative")

    if cg_mm is None:
        cg_mm = length_mm / 2.0

    center = vector((-cg_mm, 0.0, 0.0), "mm")

    grain = PropellantGrain(
        mass=scalar(propellant_mass_kg, "kg"),
        length=scalar(length_mm, "mm"),
        outer_diameter=scalar(diameter_mm, "mm"),
        core_diameter=scalar(0.0, "mm"),
        cg_location=center,
    )

    casing = MassProperties(
        mass=scalar(casing_mass_kg, "kg"),
        cg_location=center,
        inertia=matrix(np.zeros((3, 3)), "kg*m**2"),
    )

    return grain, casing


class Engine(ABC):
    """Motor the integrator samples at each time it visits."""

    @abstractmethod
    def get_thrust(self, time: float, ambient_pressure: float) -> float:
        """Return thrust along body +X, in newtons."""

    @abstractmethod
    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return motor mass properties at a simulation time."""


@dataclass(frozen=True)
class PropellantGrain(UnitChecked):
    """Hollow cylindrical grain along body X, burning radially from its core."""

    mass: Annotated[Scalar, "kg"]
    length: Annotated[Scalar, "m"]
    outer_diameter: Annotated[Scalar, "m"]
    core_diameter: Annotated[Scalar, "m"]
    cg_location: Annotated[Vector, "m"]

    _mass_kg: float = field(init=False, repr=False, compare=False)
    _outer_radius_sq: float = field(init=False, repr=False, compare=False)
    _core_radius_sq: float = field(init=False, repr=False, compare=False)
    _length_sq: float = field(init=False, repr=False, compare=False)
    _cg_m: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate grain geometry and cache SI quantities."""
        super().__post_init__()

        outer_diameter_m = float(self.outer_diameter.m_as("m"))
        core_diameter_m = float(self.core_diameter.m_as("m"))

        if not 0.0 <= core_diameter_m < outer_diameter_m:
            raise ValueError(
                "Grain core diameter must be non-negative and smaller "
                "than outer diameter"
            )

        object.__setattr__(self, "_mass_kg", float(self.mass.m_as("kg")))
        object.__setattr__(
            self,
            "_outer_radius_sq",
            (outer_diameter_m / 2.0) ** 2,
        )
        object.__setattr__(
            self,
            "_core_radius_sq",
            (core_diameter_m / 2.0) ** 2,
        )
        object.__setattr__(
            self,
            "_length_sq",
            float(self.length.m_as("m")) ** 2,
        )
        object.__setattr__(
            self,
            "_cg_m",
            self.cg_location.m_as("m"),
        )

    def mass_properties(self, burned_fraction: float) -> MassPropertiesSI:
        """Return properties of remaining propellant."""
        remaining_fraction = float(np.clip(1.0 - burned_fraction, 0.0, 1.0))

        outer_sq = self._outer_radius_sq

        inner_sq = outer_sq - remaining_fraction * (
            outer_sq - self._core_radius_sq
        )

        remaining_mass_kg = self._mass_kg * remaining_fraction

        axial_inertia = 0.5 * remaining_mass_kg * (outer_sq + inner_sq)

        transverse_inertia = (
            remaining_mass_kg
            * (3.0 * (outer_sq + inner_sq) + self._length_sq)
            / 12.0
        )

        return MassPropertiesSI(
            remaining_mass_kg,
            self._cg_m,
            np.diag(
                (
                    axial_inertia,
                    transverse_inertia,
                    transverse_inertia,
                )
            ),
        )


@dataclass(frozen=True)
class SolidEngine(Engine, UnitChecked):
    """Solid motor with a linearly interpolated thrust curve."""

    # Seconds from ignition.
    times: np.ndarray

    # Newtons at each time sample.
    thrusts: np.ndarray

    grain: PropellantGrain
    casing: MassProperties

    ignition_time: Annotated[Scalar, "s"] = field(
        default_factory=lambda: scalar(0.0, "s")
    )

    nozzle_exit_area: Annotated[Scalar, "m**2"] = field(
        default_factory=lambda: scalar(0.0, "m**2")
    )

    reference_pressure: Annotated[Scalar, "Pa"] = field(
        default_factory=lambda: scalar(101325.0, "Pa")
    )

    _ignition_s: float = field(init=False, repr=False, compare=False)
    _exit_area_m2: float = field(init=False, repr=False, compare=False)
    _reference_pressure_pa: float = field(
        init=False,
        repr=False,
        compare=False,
    )
    _impulses: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate thrust data and calculate cumulative impulse."""
        super().__post_init__()

        if len(self.times) < 2:
            raise ValueError("Thrust curve must contain at least two samples")

        if len(self.times) != len(self.thrusts):
            raise ValueError("Thrust and time arrays must have equal lengths")

        intervals = np.diff(self.times)

        if np.any(intervals <= 0.0):
            raise ValueError("Thrust curve times must be strictly increasing")

        impulses = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    0.5
                    * (self.thrusts[:-1] + self.thrusts[1:])
                    * intervals
                ),
            )
        )

        if impulses[-1] <= 0.0:
            raise ValueError("Thrust curve must deliver positive impulse")

        exit_area_m2 = float(self.nozzle_exit_area.m_as("m**2"))

        if exit_area_m2 < 0.0:
            raise ValueError("Nozzle exit area must not be negative")

        object.__setattr__(
            self,
            "_ignition_s",
            float(self.ignition_time.m_as("s")),
        )

        object.__setattr__(self, "_exit_area_m2", exit_area_m2)

        object.__setattr__(
            self,
            "_reference_pressure_pa",
            float(self.reference_pressure.m_as("Pa")),
        )

        object.__setattr__(self, "_impulses", impulses)

    @classmethod
    def from_eng(cls, motor_file_path: str) -> Self:
        """Build a solid motor from the first motor in a RASP .eng file."""
        with open(
            motor_file_path,
            encoding="utf-8",
            errors="replace",
        ) as file:
            lines = [
                fields
                for line in file
                if (fields := line.split(";")[0].split())
            ]

        if not lines:
            raise ValueError("Motor file holds no thrust curve or header")

        try:
            diameter_mm = float(lines[0][1])
            length_mm = float(lines[0][2])
            propellant_mass_kg = float(lines[0][4])
            total_mass_kg = float(lines[0][5])
        except (IndexError, ValueError) as error:
            raise ValueError("Motor file holds an invalid motor header") from error

        grain, casing = _motor_parts(
            diameter_mm,
            length_mm,
            propellant_mass_kg,
            total_mass_kg,
        )

        samples: list[tuple[float, float]] = []

        for fields in lines[1:]:
            try:
                samples.append((float(fields[0]), float(fields[1])))
            except (IndexError, ValueError):
                break

        return cls._from_samples(samples, grain, casing)

    @classmethod
    def from_rse(
        cls,
        motor_file_path: str,
        *,
        motor_front_location: Vector | None = None,
    ) -> Self:
        """Build a solid motor from the first motor in a RockSim .rse file.

        RSE cg values are measured aft from the FORWARD END of the motor.
        flight_sim CG values are measured from the ROCKET NOSE, with aft
        locations represented by negative body-X values.
        """
        motor = ElementTree.parse(motor_file_path).find(".//engine")

        if motor is None:
            raise ValueError("Motor file holds no thrust curve or motor header")

        try:
            first_point = next(motor.iter("eng-data"))

            # RSE cg is millimetres aft of the motor's forward end.
            rse_cg_from_motor_front_m = (
                float(first_point.attrib["cg"]) / 1000.0
            )

            if motor_front_location is None:
                # Backward-compatible behavior, but normally wrong for a
                # motor mounted near the rocket tail.
                motor_front_x_m = 0.0
            else:
                motor_front_x_m = float(
                    motor_front_location.m_as("m")[0]
                )

            # Body +X points to the nose.
            # Moving aft is negative X.
            #
            # Example:
            #   motor front = -2.4003 m
            #   motor local CG = 1.3976 m aft
            #   installed motor CG = -3.7979 m
            installed_motor_cg_x_m = (
                motor_front_x_m - rse_cg_from_motor_front_m
            )

            grain, casing = _motor_parts(
                float(motor.attrib["dia"]),
                float(motor.attrib["len"]),
                float(motor.attrib["propWt"]) / 1000.0,
                float(motor.attrib["initWt"]) / 1000.0,

                # _motor_parts expects a positive distance in mm and then
                # makes it negative internally. Pass the installed distance
                # aft of the ROCKET nose.
                -installed_motor_cg_x_m * 1000.0,
            )

            samples = [
                (float(point.attrib["t"]), float(point.attrib["f"]))
                for point in motor.iter("eng-data")
            ]

        except (KeyError, StopIteration, ValueError) as error:
            raise ValueError("Motor file holds an invalid motor header") from error

        return cls._from_samples(samples, grain, casing)
    @classmethod
    def _from_samples(
        cls,
        samples: list[tuple[float, float]],
        grain: PropellantGrain,
        casing: MassProperties,
    ) -> Self:
        """Build a motor from (time, thrust) samples."""

        if not samples:
            raise ValueError("Motor file holds no thrust curve")

        # Ensure the motor has an explicit zero-thrust point at ignition.
        if samples[0][0] > 0.0:
            samples = [(0.0, 0.0), *samples]

        times, thrusts = np.array(samples, dtype=float).T

        return cls(
            times=times,
            thrusts=thrusts,
            grain=grain,
            casing=casing,
        )

    @property
    def total_impulse(self) -> float:
        """Return total motor impulse in N*s."""
        return float(self._impulses[-1])

    def get_thrust(self, time: float, ambient_pressure: float) -> float:
        """Return non-negative thrust along body +X, in N."""
        burn_time_s = time - self._ignition_s

        if not self.times[0] <= burn_time_s <= self.times[-1]:
            return 0.0

        thrust_curve_n = float(
            np.interp(
                burn_time_s,
                self.times,
                self.thrusts,
            )
        )

        pressure_thrust_n = (
            self._reference_pressure_pa - ambient_pressure
        ) * self._exit_area_m2

        return max(thrust_curve_n + pressure_thrust_n, 0.0)

    def burned_fraction(self, time: float) -> float:
        """Return propellant fraction burned using delivered impulse."""
        burn_time_s = time - self._ignition_s

        if burn_time_s <= self.times[0]:
            return 0.0

        if burn_time_s >= self.times[-1]:
            return 1.0

        index = int(
            np.searchsorted(self.times, burn_time_s, side="right")
        ) - 1

        time_into_interval_s = burn_time_s - self.times[index]

        thrust_slope = (
            self.thrusts[index + 1] - self.thrusts[index]
        ) / (
            self.times[index + 1] - self.times[index]
        )

        impulse = self._impulses[index] + time_into_interval_s * (
            self.thrusts[index]
            + 0.5 * thrust_slope * time_into_interval_s
        )

        return float(impulse / self._impulses[-1])

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return casing plus remaining propellant mass properties."""
        return combine(
            self.casing.si,
            self.grain.mass_properties(self.burned_fraction(time)),
        )


def solid_engine_from_csv(
    motor_file_path: str,
    grain: PropellantGrain,
    casing: MassProperties,
) -> SolidEngine:
    """Build a motor from CSV columns called Time and Thrust."""
    motor_data = pd.read_csv(motor_file_path)

    return SolidEngine(
        times=motor_data["Time"].to_numpy(dtype=float),
        thrusts=motor_data["Thrust"].to_numpy(dtype=float),
        grain=grain,
        casing=casing,
    )