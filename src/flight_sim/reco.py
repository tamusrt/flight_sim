"""The recovery model (RECO) as a swappable part.

A *recovery scheme* (``recovery_systems``) is the hardware: the canopies, the
charges and the flight computer. A *RECO version* is the way the descent under
that hardware is flown. Every version takes the same request (the ascent that
ended at apogee) and gives back the same outcome, so the code that uses a
descent does not care which version made it:

* ``FullRECO`` is the model as it has always been: the scheme's own ``plan``,
  with the shock cord and the rocket swinging under the canopy, in small steps.
* ``FastRECO`` (``fast_reco``) drops the details that do not move the landing
  point, to fly thousands of descents for a Monte Carlo run.

New versions are written as a ``RECO`` subclass and added with ``register``;
``get_reco`` finds one by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from flight_sim.descent import DescentResult
from flight_sim.flight_computer import ApogeeDetection, RecoveryPlan, Sample
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_systems import RecoveryScheme
from flight_sim.vehicle.mass_properties import MassPropertiesSI
from flight_sim.vehicle.rocket_properties import RocketProperties

_G0 = 9.80665


@dataclass(frozen=True)
class RecoveryRequest:
    """What a RECO version needs to fly a descent.

    Attributes:
        flight (list[Sample]): The ascent, time and state, ending at the true
            apogee. A version that only needs the apogee state and how long the
            climb took may be given a short trace (launch and apogee).
        config (IntegrationConfiguration): Atmosphere, wind and gravity.
        apogee_mass (MassPropertiesSI): The rocket's mass after the burn.
        properties (RocketProperties | None): When given, the rocket is flown
            with the 6-DOF model from apogee to the first line stretch, so it
            can tumble.
    """

    flight: list[Sample]
    config: IntegrationConfiguration
    apogee_mass: MassPropertiesSI
    properties: RocketProperties | None = None


@dataclass(frozen=True)
class RecoveryOutcome:
    """What a RECO version reports; the same for every version.

    Attributes:
        version (str): Name of the version that flew it.
        descent (DescentResult): Samples from apogee to the ground, with the
            canopy openings and their loads.
        apogee (ApogeeDetection): When the flight computer's votes agreed.
        fire_s (float): Time the apogee charge fires.
        line_stretch_s (float): Time the first canopy starts to open.
        main_command_s (float | None): Time the main event was commanded.
        main_true_altitude_m (float | None): True height at that moment.
        separated (bool): Whether the apogee charge broke its shear pins.
        plan (RecoveryPlan | None): The full plan, for the version that has one.
    """

    version: str
    descent: DescentResult
    apogee: ApogeeDetection
    fire_s: float
    line_stretch_s: float
    main_command_s: float | None
    main_true_altitude_m: float | None
    separated: bool = True
    plan: RecoveryPlan | None = None

    @property
    def landing_position_m(self) -> np.ndarray:
        """Where the rocket came down: the pad frame, X up, in metres."""
        return np.asarray(self.descent.states[-1].position.m_as("m"), dtype=float)

    @property
    def landing_velocity_m_s(self) -> np.ndarray:
        """Velocity at the last sample, in m/s."""
        return np.asarray(self.descent.states[-1].velocity.m_as("m/s"), dtype=float)

    @property
    def drift_m(self) -> float:
        """Horizontal distance from the pad to where the rocket came down."""
        position = self.landing_position_m
        return float(np.hypot(position[1], position[2]))

    def drogue_rate_m_s(self) -> float | None:
        """Vertical speed just before the last canopy opens (None with one canopy).

        The drogue has settled by then. With a single canopy event there is no
        drogue stage.
        """
        events = self.descent.deployments
        if len(events) < 2:
            return None
        before = [
            abs(float(s.velocity.m_as("m/s")[0]))
            for t, s in zip(self.descent.times_s, self.descent.states, strict=True)
            if t < events[-1].time_s - 0.5
        ]
        return before[-1] if before else None


@dataclass(frozen=True)
class RECO(ABC):
    """One way of flying the descent of a recovery scheme.

    Attributes:
        scheme (RecoveryScheme): The canopies, charges and flight computer.
    """

    name: ClassVar[str] = ""

    scheme: RecoveryScheme

    @abstractmethod
    def descend(self, request: RecoveryRequest) -> RecoveryOutcome:
        """Fly the descent from the apogee that ends ``request.flight``."""


@dataclass(frozen=True)
class FullRECO(RECO):
    """The model as it was: the scheme's own plan, in small steps.

    Includes the shock cord, the rocket swinging under its canopy (single
    separation schemes) and the descent integrated through every opening.
    """

    name: ClassVar[str] = "full"

    def descend(self, request: RecoveryRequest) -> RecoveryOutcome:
        """Fly the descent with ``scheme.plan``."""
        plan = self.scheme.plan(
            request.flight, request.config, request.apogee_mass, request.properties
        )
        return RecoveryOutcome(
            version=self.name,
            descent=plan.descent,
            apogee=plan.apogee,
            fire_s=plan.fire_s,
            line_stretch_s=plan.line_stretch_s,
            main_command_s=plan.main_command_s,
            main_true_altitude_m=plan.main_true_altitude_m,
            separated=plan.separation.separated,
            plan=plan,
        )


_VERSIONS: dict[str, type[RECO]] = {FullRECO.name: FullRECO}


def register(version: type[RECO]) -> type[RECO]:
    """Make a RECO version available to ``get_reco`` under its name."""
    if not version.name:
        raise ValueError("A RECO version needs a name")
    _VERSIONS[version.name] = version
    return version


def versions() -> list[str]:
    """The names of the registered RECO versions."""
    return sorted(_VERSIONS)


def get_reco(name: str, scheme: RecoveryScheme) -> RECO:
    """Build the RECO version called ``name`` for a recovery scheme."""
    if name not in _VERSIONS:
        raise KeyError(f"No RECO version '{name}'; there are {', '.join(versions())}")
    return _VERSIONS[name](scheme)
