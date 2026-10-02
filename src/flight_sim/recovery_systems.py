"""The ways a rocket can come down, one class for each.

A recovery architecture is more than the canopies. It also decides how many
times the airframe comes apart, which charge fires at which event, what the
flight computer commands, and what the rocket does between events. Each
class here is one mainstream architecture, and the one a rocket's profile
names is the one ``visual_run`` flies:

* ``SingleDeploy``: one canopy comes out at apogee, on one separation.
* ``ReefedSingleSeparation``: one canopy comes out reefed at apogee on one
  separation and is cut open at the main altitude (Sol Invictus).
* ``DualDeploy``: a drogue at apogee on one separation, then a main at the
  main altitude on a second one (Morpheus).
* ``Drogueless``: no canopy at apogee; the rocket falls tumbling to the
  main altitude, where one separation releases the main.

The canopies, drag and flight computer logic are the ones in ``descent`` and
``flight_computer``; each class decides how they are wired together. The
single-separation classes fly the descent as ``flight_computer`` always has,
so Sol Invictus is unchanged.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import (
    DescentResult,
    Parachute,
    RecoverySystem,
    ReefedParachute,
    simulate_descent,
)
from flight_sim.flight_computer import (
    FlightComputer,
    RecoveryPlan,
    Sample,
    _eject,
    _state_at,
    plan_recovery,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import (
    CORD_TO_BODY_M,
    CORD_TO_NOSE_M,
    EjectionCharge,
    NoseSwing,
    Separation,
)
from flight_sim.visualize import RecoveryFrame

_KG_PER_LB = 0.45359237
_J_PER_IN_LBF = 0.112984829
# Black powder gas constant and combustion temperature, from the recovery
# team's charge sizing sheet: 265.92 in-lbf/(lbm R) at 3307 R
_BP_GAS_CONSTANT = 265.92 * _J_PER_IN_LBF / _KG_PER_LB * 1.8  # J/(kg K)
_BP_TEMPERATURE_K = 3307.0 / 1.8

# Stiffness times length of the shock cord, N: the 5100 N/m of the 13.87 m
# cord that Sol Invictus flies (assumed, typical of tubular Kevlar)
CORD_EA_N = 5100.0 * (CORD_TO_NOSE_M + CORD_TO_BODY_M)

# A canopy that is never released by height, so a descent can be flown with
# the main held back
_NEVER_M = -1.0


@dataclass(frozen=True)
class BlackPowderCharge:
    """A black powder charge in the bay it pressurises.

    The pressure is the ideal-gas one of the charge's mass burnt in the bay
    volume, as in the recovery team's charge sizing sheet.

    Attributes:
        grams (float): Mass of black powder.
        bay_diameter_m (float): Inside diameter of the bay, in metres.
        bay_length_m (float): Length of bay the gas fills, in metres.
    """

    grams: float
    bay_diameter_m: float
    bay_length_m: float

    @property
    def pressure_pa(self) -> float:
        """Pressure the burnt charge adds to the bay, in pascals."""
        volume = math.pi * self.bay_diameter_m**2 / 4.0 * self.bay_length_m
        return self.grams * 1e-3 * _BP_GAS_CONSTANT * _BP_TEMPERATURE_K / volume

    def ejection(
        self,
        *,
        stroke_m: float,
        shear_pins: int,
        pin_strength_n: float,
        section_mass_kg: float,
        section_drag_area_m2: float,
        other_drag_area_m2: float,
        cord_length_m: float,
    ) -> EjectionCharge:
        """The separation this charge makes of a section and the cord on it.

        Args:
            stroke_m (float): Shoulder length the gas pushes the section out.
            shear_pins (int): Pins holding the section.
            pin_strength_n (float): Shear strength of one pin.
            section_mass_kg (float): Mass of the section that leaves.
            section_drag_area_m2 (float): Its drag coefficient times area.
            other_drag_area_m2 (float): The same for the rest of the rocket.
            cord_length_m (float): Shock cord the section runs out.
        """
        return EjectionCharge(
            pressure_pa=self.pressure_pa,
            bore_radius_m=self.bay_diameter_m / 2.0,
            stroke_m=stroke_m,
            shear_pins=shear_pins,
            pin_strength_n=pin_strength_n,
            nose_mass_kg=section_mass_kg,
            nose_drag_area_m2=section_drag_area_m2,
            body_drag_area_m2=other_drag_area_m2,
            cord_length_m=cord_length_m,
            cord_stiffness_n_m=CORD_EA_N / cord_length_m,
        )


@dataclass(frozen=True)
class RecoveryScheme(ABC):
    """One recovery architecture: canopies, separations and their commands.

    Attributes:
        recovery (RecoverySystem): The canopies, with nominal settings; the
            scheme sets their release times from the flight computer.
        computer (FlightComputer): The avionics and the settings it was
            programmed with.
        apogee_charge (EjectionCharge | None): The separation fired after
            apogee is detected, or None when nothing separates there.
        main_charge (EjectionCharge | None): The separation fired when the
            main altitude is commanded, or None when there is none.
        main_delay_s (float): Programmed delay from the main command to the
            main charge.
    """

    kind: ClassVar[str] = ""
    separations: ClassVar[int] = 1

    recovery: RecoverySystem
    computer: FlightComputer
    apogee_charge: EjectionCharge | None = None
    main_charge: EjectionCharge | None = None
    main_delay_s: float = 0.0

    @abstractmethod
    def plan(
        self, flight: list[Sample], config: IntegrationConfiguration
    ) -> RecoveryPlan:
        """Fly the descent with the events where this scheme puts them."""

    @abstractmethod
    def frames(self, plan: RecoveryPlan) -> Iterator[tuple[float, RecoveryFrame]]:
        """The recovery hardware at each sample of the descent."""


@dataclass(frozen=True)
class SingleSeparation(RecoveryScheme):
    """The rocket comes apart once, at apogee, and hangs under one canopy.

    This is the base of ``SingleDeploy`` and ``ReefedSingleSeparation``. The
    nose leaves the body at the joint and runs out its leg of the shock
    cord; the canopy comes out with it and the rocket then swings from the
    canopy on the body's leg.

    Attributes:
        separation_station_m (float): Nose tip to the joint, where both
            harnesses are.
        nose_harness_to_cg_m (float): The nose section's centre of mass from
            its harness.
        line_share (float): Suspension line length as a share of the canopy
            diameter.
        cord_to_nose_m (float): The nose's leg of the shock cord.
        cord_to_body_m (float): The body's leg of the shock cord.
    """

    kind: ClassVar[str] = "single separation"

    separation_station_m: float = 0.9144
    nose_harness_to_cg_m: float = 0.45
    line_share: float = 0.9
    cord_to_nose_m: float = CORD_TO_NOSE_M
    cord_to_body_m: float = CORD_TO_BODY_M

    def __post_init__(self) -> None:
        if self.apogee_charge is None:
            raise ValueError(f"A {self.kind} needs an apogee charge")

    @property
    def lines_m(self) -> float:
        """Length of the suspension lines of the biggest canopy, in metres."""
        return self.line_share * max(
            (p.diameter_m for p in self.recovery.parachutes), default=0.0
        )

    def swing(self, centre_of_gravity_m: float) -> CanopySwing:
        """The two-body model: the line from the canopy to the rocket's CG.

        It runs through the suspension lines and the body's leg of the cord
        to the harness at the joint, and on to the centre of gravity.
        """
        return CanopySwing(
            line_length_m=self.cord_to_body_m
            + self.lines_m
            + centre_of_gravity_m
            - self.separation_station_m
        )

    def plan(
        self, flight: list[Sample], config: IntegrationConfiguration
    ) -> RecoveryPlan:
        """Fly the descent as ``plan_recovery`` does, with this scheme's parts."""
        assert self.apogee_charge is not None
        apogee = flight[-1][1]
        return plan_recovery(
            flight,
            config,
            self.recovery,
            self.computer,
            self.apogee_charge,
            swing=self.swing(-float(apogee.cg_location.m_as("m")[0])),
        )

    def frames(self, plan: RecoveryPlan) -> Iterator[tuple[float, RecoveryFrame]]:
        """The nose coming out, then the line, the canopy and the nose swinging."""
        descent, swing = plan.descent, plan.swing
        if swing is None:
            raise ValueError("The plan was not flown with the swing model")
        times = descent.times_s
        stretch = int(np.searchsorted(times, plan.line_stretch_s))
        lines = np.array(swing.line_directions)
        nose = NoseSwing(
            length_m=self.cord_to_nose_m + self.lines_m + self.nose_harness_to_cg_m
        )
        nose_dirs = nose.swing(
            times,
            np.array(swing.canopy_velocities),
            start_s=plan.line_stretch_s,
            start=lines[min(stretch, len(times) - 1)],
        )
        curve = plan.separation
        for i, time in enumerate(times):
            if time >= plan.line_stretch_s and float(np.linalg.norm(lines[i])) > 0.0:
                yield (
                    time,
                    RecoveryFrame(
                        nose_dir=[float(x) for x in nose_dirs[i]],
                        line=[float(x) for x in lines[i]],
                        swing_deg=float(np.degrees(swing.angles_rad[i])),
                        drag_fraction=swing.drag_fractions[i],
                    ),
                )
            elif time >= plan.fire_s:
                yield (
                    time,
                    RecoveryFrame(
                        nose_sep=float(
                            np.interp(
                                time - plan.fire_s, curve.times_s, curve.distances_m
                            )
                        )
                    ),
                )
            else:
                yield time, RecoveryFrame()


@dataclass(frozen=True)
class SingleDeploy(SingleSeparation):
    """One plain canopy at apogee: no drogue, no reefing, one separation."""

    kind: ClassVar[str] = "single deploy"

    def __post_init__(self) -> None:
        super().__post_init__()
        canopies = self.recovery.parachutes
        if len(canopies) != 1 or not isinstance(canopies[0], Parachute):
            raise ValueError("A single deploy flies one plain canopy")
        if canopies[0].deploy_altitude_m is not None:
            raise ValueError("A single deploy's canopy comes out at apogee")


@dataclass(frozen=True)
class ReefedSingleSeparation(SingleSeparation):
    """One reefed canopy at apogee, cut open at the main altitude."""

    kind: ClassVar[str] = "reefed single separation"

    def __post_init__(self) -> None:
        super().__post_init__()
        canopies = self.recovery.parachutes
        if len(canopies) != 1 or not isinstance(canopies[0], ReefedParachute):
            raise ValueError("A reefed single separation flies one reefed canopy")


@dataclass(frozen=True)
class _MainSeparation(RecoveryScheme):
    """Base of the schemes whose main comes out on its own separation.

    The descent is flown twice. First the main is held back, so the flight
    computer can read the barometer on the way down and find when it
    commands the main altitude. The main charge fires after its delay, the
    section runs out the cord, and the main inflates at line stretch, which
    is where the second flight releases it. The descent is flown as a point
    mass: the drag the rocket sheds in the swing is not modelled for these
    schemes.
    """

    def __post_init__(self) -> None:
        if self.main_charge is None:
            raise ValueError(f"A {self.kind} needs a main charge")

    def _canopies(self) -> tuple[Parachute | None, Parachute]:
        """The apogee canopy, if any, and the main."""
        raise NotImplementedError

    def _apogee_event(
        self, coasting: list[Sample], config: IntegrationConfiguration, detected: float
    ) -> tuple[float, Separation, float]:
        """Charge time, separation and line stretch of the apogee event."""
        fire = detected + self.computer.apogee_delay_s
        if self.apogee_charge is None:
            return (
                fire,
                Separation(False, 0.0, math.inf, 0.0, 0.0, [0.0], [0.0]),
                math.inf,
            )
        separation = _eject(_state_at(coasting, fire), config, self.apogee_charge)
        return fire, separation, fire + separation.line_stretch_s

    def _canopies_at(
        self, stretch_delay_s: float, release_altitude_m: float
    ) -> RecoverySystem:
        """The recovery with the apogee canopy on its delay and the main on a height."""
        first, main = self._canopies()
        released: list[Parachute | ReefedParachute] = []
        if first is not None:
            released.append(replace(first, deploy_delay_s=stretch_delay_s))
        released.append(replace(main, deploy_altitude_m=release_altitude_m))
        return replace(self.recovery, parachutes=tuple(released))

    def plan(  # pylint: disable=too-many-locals
        self, flight: list[Sample], config: IntegrationConfiguration
    ) -> RecoveryPlan:
        """Fly the descent: apogee event, then the main on its own separation."""
        assert self.main_charge is not None
        apogee_time, apogee_state = flight[-1]
        computer = self.computer
        coast = simulate_descent(
            apogee_time,
            apogee_state,
            config,
            replace(self.recovery, parachutes=()),
            max_time_s=computer.apogee_delay_s + 30.0,
        )
        coasting = flight + list(zip(coast.times_s, coast.states, strict=True))
        detection = computer.detect_apogee(computer.sense(coasting, config))
        detected = (
            detection.detected_s if detection.detected_s is not None else apogee_time
        )
        fire, separation, stretch = self._apogee_event(coasting, config, detected)

        def fly(release_altitude_m: float) -> tuple[RecoverySystem, DescentResult]:
            recovery = self._canopies_at(stretch - apogee_time, release_altitude_m)
            return recovery, simulate_descent(
                apogee_time, apogee_state, config, recovery
            )

        _, held = fly(_NEVER_M)
        command = computer.main_command(
            computer.sense(
                flight + list(zip(held.times_s, held.states, strict=True)), config
            ),
            after_s=detected,
        )
        altitudes = [float(s.position.m_as("m")[0]) for s in held.states]
        main_fire = main_separation = main_stretch = None
        release = _NEVER_M
        if command is not None:
            main_fire = command + self.main_delay_s
            index = int(np.argmin(np.abs(np.array(held.times_s) - main_fire)))
            main_separation = _eject(held.states[index], config, self.main_charge)
            main_stretch = main_fire + main_separation.line_stretch_s
            release = float(np.interp(main_stretch, held.times_s, altitudes))
        recovery, descent = fly(release)
        return RecoveryPlan(
            recovery=recovery,
            descent=descent,
            swing=None,
            apogee=detection,
            fire_s=fire,
            separation=separation,
            line_stretch_s=stretch,
            main_command_s=command,
            main_true_altitude_m=(
                None
                if command is None
                else float(np.interp(command, held.times_s, altitudes))
            ),
            main_fire_s=main_fire,
            main_separation=main_separation,
            main_line_stretch_s=main_stretch,
        )

    def frames(self, plan: RecoveryPlan) -> Iterator[tuple[float, RecoveryFrame]]:
        """The nose cone coming out of the body at the main separation."""
        curve = plan.main_separation
        for time in plan.descent.times_s:
            if (
                curve is not None
                and plan.main_fire_s is not None
                and plan.main_line_stretch_s is not None
                and plan.main_fire_s <= time < plan.main_line_stretch_s
            ):
                distance = float(
                    np.interp(time - plan.main_fire_s, curve.times_s, curve.distances_m)
                )
                yield time, RecoveryFrame(nose_sep=distance)
            else:
                yield time, RecoveryFrame()


@dataclass(frozen=True)
class DualDeploy(_MainSeparation):
    """A drogue at apogee on one separation, a main at altitude on another.

    The first canopy is the drogue, released at line stretch of the apogee
    separation; the second is the main, released by height when the main
    separation's cord comes tight.
    """

    kind: ClassVar[str] = "dual deploy"
    separations: ClassVar[int] = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.apogee_charge is None:
            raise ValueError("A dual deploy needs an apogee charge")
        canopies = self.recovery.parachutes
        if len(canopies) != 2 or not all(isinstance(c, Parachute) for c in canopies):
            raise ValueError("A dual deploy flies a drogue and a main, both plain")

    def _canopies(self) -> tuple[Parachute | None, Parachute]:
        drogue, main = self.recovery.parachutes
        assert isinstance(drogue, Parachute) and isinstance(main, Parachute)
        return drogue, main


@dataclass(frozen=True)
class Drogueless(_MainSeparation):
    """No canopy at apogee: the rocket tumbles to the main altitude.

    Nothing separates at apogee, so the body falls on its own drag area
    (``recovery.body_drag_area_m2``, the tumbling one) until the main charge
    fires, as the recovery team's drogueless velocities sheet assumes.
    """

    kind: ClassVar[str] = "drogueless"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.apogee_charge is not None:
            raise ValueError("A drogueless rocket does not separate at apogee")
        canopies = self.recovery.parachutes
        if len(canopies) != 1 or not isinstance(canopies[0], Parachute):
            raise ValueError("A drogueless rocket flies one plain main")

    def _canopies(self) -> tuple[Parachute | None, Parachute]:
        main = self.recovery.parachutes[0]
        assert isinstance(main, Parachute)
        return None, main
