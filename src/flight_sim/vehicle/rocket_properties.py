"""Data structures defining physical vehicle properties"""

import math
from dataclasses import dataclass, field

from flight_sim.utilities.data_loader import AeroTable
from flight_sim.vehicle.engine import Engine, solid_engine_from_csv


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


@dataclass
class RocketProperties:
    """Aerodynamic and motor properties of the rocket"""

    aero_table: AeroTable
    motor_file_path: str
    propellant_mass: float  # Total weight of solid fuel in kg

    # Fins for the roll damping and misalignment torques. None leaves roll
    # to the aerodynamic table alone.
    fins: TrapezoidFinSet | None = None

    engine: Engine = field(init=False)

    def __post_init__(self) -> None:
        self.engine = solid_engine_from_csv(self.motor_file_path, self.propellant_mass)
