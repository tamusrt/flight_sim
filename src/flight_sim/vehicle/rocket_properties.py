"""Data structures defining physical vehicle properties"""

import math
from dataclasses import dataclass

import numpy as np

from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine
from flight_sim.vehicle.mass_properties import (
    MassProperties,
    MassPropertiesSI,
    combine,
)


@dataclass(frozen=True)
class TrapezoidFinSet:
    """Planform of a set of identical trapezoidal fins, for the roll loads.

    Lengths are in metres. ``sweep_length_m`` is the axial distance from the
    root leading edge to the tip leading edge, as OpenRocket defines it.
    ``misalignment_rad`` is a small manufacturing twist of every fin in the
    same sense, which spins the rocket up; zero means a perfectly built set.
    """

    fin_count: int
    root_chord_m: float
    tip_chord_m: float
    span_m: float
    sweep_length_m: float
    body_radius_m: float
    misalignment_rad: float = 0.0

    @property
    def planform_area_m2(self) -> float:
        """Area of one fin in m**2."""
        return 0.5 * (self.root_chord_m + self.tip_chord_m) * self.span_m

    @property
    def aspect_ratio(self) -> float:
        """Span squared over the area of one fin."""
        return self.span_m**2 / self.planform_area_m2

    @property
    def midchord_sweep_rad(self) -> float:
        """Sweep angle of the line joining the root and tip mid-chords."""
        offset = self.sweep_length_m + 0.5 * (self.tip_chord_m - self.root_chord_m)
        return math.atan2(offset, self.span_m)

    def _chord_moment(self, power: int) -> float:
        """Return fin count times the integral of chord * y**power over the span.

        ``y`` is the distance from the roll axis, from the body surface to the
        fin tip; the chord varies linearly from root to tip, so the integral
        is evaluated exactly.
        """
        inner = self.body_radius_m
        outer = inner + self.span_m
        taper = (self.tip_chord_m - self.root_chord_m) / self.span_m
        constant = self.root_chord_m - taper * inner

        def antiderivative(y: float) -> float:
            return constant * y ** (power + 1) / (power + 1) + taper * y ** (
                power + 2
            ) / (power + 2)

        return self.fin_count * (antiderivative(outer) - antiderivative(inner))

    @property
    def roll_forcing_integral(self) -> float:
        """Fin count times the integral of chord * y over the span, in m**3."""
        return self._chord_moment(1)

    @property
    def roll_damping_integral(self) -> float:
        """Fin count times the integral of chord * y**2 over the span, in m**4."""
        return self._chord_moment(2)

    def normal_force_slope(self, mach: float) -> float:
        """Return one fin's normal force slope per radian, on its own area.

        Subsonic: Barrowman's form of the Diederich finite-wing formula,
        which tends to the slender-wing value pi * AR at Mach 1. Supersonic:
        the smaller of slender-wing theory and Ackeret's 4 / sqrt(M**2 - 1),
        so the slope is continuous through Mach 1. Ackeret ignores tip
        losses, so above about Mach 2 this overstates the slope of low
        aspect ratio fins (by about a third at Mach 2.5 for Sol Invictus).
        That only changes how quickly the roll rate settles, not the rate it
        settles at.

        Args:
            mach (float): Mach number.

        Returns:
            float: d(C_N)/d(alpha) per radian.
        """
        aspect = self.aspect_ratio
        slender = math.pi * aspect
        if mach >= 1.0:
            if mach == 1.0:
                return slender
            return min(slender, 4.0 / math.sqrt(mach**2 - 1.0))
        beta = math.sqrt(1.0 - mach**2)
        compressible = beta * aspect / math.cos(self.midchord_sweep_rad)
        return 2.0 * math.pi * aspect / (1.0 + math.sqrt(1.0 + compressible**2))


_CROSSFLOW_CD = 1.2  # A cylinder or a flat fin across the flow, subsonic


@dataclass(frozen=True)
class PitchDamping:
    """The air resisting the rocket turning end over end (pitch and yaw).

    A rocket turning at rate ``w`` about its centre of gravity moves each part of
    it sideways through the air at ``w`` times its distance from the CG, and the
    air pushes back. The aero table has no rates in it, so this adds that torque:

    * fins at distance ``l`` aft of the CG meet the air at an extra angle
      ``w l / V`` (the usual pitch damping, linear in ``w``), and when the rocket
      is slow, as near apogee, they are pushed flat through the air like plates,
      ``0.5 rho Cd (w l)^2`` per unit area;
    * the body, a cylinder of diameter ``D`` whose ends are ``a`` and ``b`` from
      the CG, gives ``0.5 rho Cd D w^2 (a^4 + b^4) / 4`` as it sweeps through the air.

    The fins count as half their number (on average half of them face the
    turn). This is what slows a tumble after apogee and keeps the climb from
    oscillating without end; it costs one atmosphere lookup a step.

    Attributes:
        body_length_m (float): Nose tip to the aft end.
        body_diameter_m (float): Body diameter.
        fin_station_m (float): Centre of the fins' area, from the nose tip.
        crossflow_cd (float): Drag coefficient of the body and fins across the flow.
    """

    body_length_m: float
    body_diameter_m: float
    fin_station_m: float
    crossflow_cd: float = _CROSSFLOW_CD

    @classmethod
    def estimate(cls, table: "AeroTable", diameter_m: float) -> "PitchDamping":
        """A rough damping from the aero table alone, for a rocket with no geometry.

        The fins are put at the table's small-angle centre of pressure (Mach 0.3,
        2 degrees) and the body made 15 percent longer than that.
        """
        c = table(0.3, 2.0, 0.0)
        xcp = (
            c.cmy * table.reference_length_m / c.cz
            if abs(c.cz) > 1e-9
            else 20 * diameter_m
        )
        xcp = abs(xcp)
        return cls(
            body_length_m=xcp / 0.85, body_diameter_m=diameter_m, fin_station_m=xcp
        )

    def torque(  # pylint: disable=too-many-arguments
        self,
        angular_velocity: np.ndarray,
        *,
        airspeed_m_s: float,
        mach: float,
        air_density: float,
        cg_m: float,
        fins: "TrapezoidFinSet | None",
    ) -> np.ndarray:
        """The damping torque in body axes, N m, opposing the pitch and yaw rate.

        Args:
            angular_velocity (np.ndarray): Body rates in rad/s; the roll part is
                left to the fins' own roll damping.
            airspeed_m_s (float): Speed through the air.
            mach (float): Mach number.
            air_density (float): kg/m**3.
            cg_m (float): Centre of gravity, metres aft of the nose tip.
            fins (TrapezoidFinSet | None): The fins, for their area and lift slope.
        """
        turn = np.array([0.0, angular_velocity[1], angular_velocity[2]])
        rate = float(np.linalg.norm(turn))
        if rate < 1e-9:
            return np.zeros(3)
        half_rho = 0.5 * air_density
        aft = max(self.body_length_m - cg_m, 0.0)
        fore = max(cg_m, 0.0)
        body = (
            half_rho
            * self.crossflow_cd
            * self.body_diameter_m
            * rate**2
            * (fore**4 + aft**4)
            / 4.0
        )
        fin = 0.0
        if fins is not None:
            arm = self.fin_station_m - cg_m
            area = fins.planform_area_m2 * fins.fin_count / 2.0
            sideways = rate * abs(arm)
            fin = (
                half_rho
                * area
                * (
                    fins.normal_force_slope(mach) * airspeed_m_s * sideways
                    + self.crossflow_cd * sideways**2
                )
                * abs(arm)
            )
        result: np.ndarray = -(body + fin) * turn / rate
        return result


@dataclass
class RocketProperties:
    """Aerodynamics, motor and dry mass properties of the rocket."""

    aero_table: AeroTable
    engine: Engine

    # Everything but the motor
    dry_mass_properties: MassProperties

    # Fins for the roll damping and misalignment torques. None leaves roll
    # to the aerodynamic table alone.
    fins: TrapezoidFinSet | None = None

    # The air resisting pitch and yaw rates. None leaves them undamped.
    damping: PitchDamping | None = None

    def mass_properties(self, time: float) -> MassPropertiesSI:
        """Return the whole rocket's mass properties at a simulation time in s."""
        return combine(self.dry_mass_properties.si, self.engine.mass_properties(time))
