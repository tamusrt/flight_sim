"""Motion of the recovery hardware: nose ejection and swing under the canopy.

Ejection: the separation charge pressurises the bay behind the nose. Once
the force on the bulkhead beats the shear pins, the gas pushes the nose out
over the length of its shoulder, and the work it does sets the relative
speed of nose and body. The nose then runs out the shock cord, slowed by its
extra drag, until the cord comes tight (line stretch) and the canopy starts
to inflate. The cord stretching like a spring gives the snatch load
``v * sqrt(k * mu)``, with ``mu`` the reduced mass of the two parts.

Nose swing: the body's swing is part of the descent itself (see
``swing_descent``). The light nose hangs on its own leg of the shock cord
from the canopy's attachment point, like a spherical pendulum. It settles
along the apparent gravity in the canopy's frame, ``g - a`` with ``a`` the
canopy's acceleration, so it trails behind right after the opening and
hangs straight down in a steady descent. The air damps it.

Charge pressure, shear pins, cord stiffness and damping are
assumptions, set to values typical of high-power rockets.
"""

import math
from dataclasses import dataclass

import numpy as np

_G = 9.80665

# The two legs of the shock cord, from the separation joint to each part's
# harness, in metres (SOL_4_30.ork: 182 and 364 inches)
CORD_TO_NOSE_M = 182 * 0.0254
CORD_TO_BODY_M = 364 * 0.0254


@dataclass(frozen=True)
class Separation:
    """Outcome of the nose ejection.

    Attributes:
        separated (bool): Whether the charge broke the shear pins.
        speed_m_s (float): Relative speed of nose and body as the nose
            leaves the shoulder.
        line_stretch_s (float): Time from the charge to line stretch.
        stretch_speed_m_s (float): Relative speed when the cord comes tight.
        snatch_force_n (float): Peak cord load at line stretch.
        times_s (list[float]): Times after the charge, for the curve below.
        distances_m (list[float]): Nose-to-body distance along the cord.
    """

    separated: bool
    speed_m_s: float
    line_stretch_s: float
    stretch_speed_m_s: float
    snatch_force_n: float
    times_s: list[float]
    distances_m: list[float]


@dataclass(frozen=True)
class EjectionCharge:
    """Separation charge, shear pins and shock cord.

    Attributes:
        pressure_pa (float): Design bay pressure from the charge (15 psi).
        bore_radius_m (float): Radius the pressure acts on (the shoulder).
        stroke_m (float): Shoulder length the gas pushes the nose along.
        shear_pins (int): Number of shear pins holding the nose.
        pin_strength_n (float): Shear strength of one pin (2-56 nylon).
        nose_mass_kg (float): Mass of the separating nose section.
        nose_drag_area_m2 (float): Its drag coefficient times area.
        body_drag_area_m2 (float): The body's drag coefficient times area.
        cord_length_m (float): Total shock cord, nose to body.
        cord_stiffness_n_m (float): Spring rate of the whole cord.
    """

    pressure_pa: float = 15.0 * 6894.757
    bore_radius_m: float = 0.0742
    stroke_m: float = 0.1524
    shear_pins: int = 3
    pin_strength_n: float = 35.0 * 4.448222
    nose_mass_kg: float = 4.0
    nose_drag_area_m2: float = 0.5 * 0.0182414692
    body_drag_area_m2: float = 0.55 * 0.0182414692
    cord_length_m: float = CORD_TO_NOSE_M + CORD_TO_BODY_M
    cord_stiffness_n_m: float = 5100.0

    def separate(
        self, total_mass_kg: float, airspeed_m_s: float, air_density: float
    ) -> Separation:
        """Eject the nose and run out the cord.

        Args:
            total_mass_kg (float): Mass of the whole rocket at the charge.
            airspeed_m_s (float): Speed through the air at the charge.
            air_density (float): Air density there, kg/m**3.

        Returns:
            Separation: Speeds, line stretch time, snatch load and the
                nose-to-body distance over time.
        """
        body_mass = total_mass_kg - self.nose_mass_kg
        reduced = self.nose_mass_kg * body_mass / total_mass_kg
        area = math.pi * self.bore_radius_m**2
        net = self.pressure_pa * area - self.shear_pins * self.pin_strength_n
        if net <= 0.0:
            return Separation(False, 0.0, math.inf, 0.0, 0.0, [0.0], [0.0])
        speed = math.sqrt(2.0 * net * self.stroke_m / reduced)
        # The nose's extra drag per unit mass slows it relative to the body
        dynamic = 0.5 * air_density * airspeed_m_s**2
        decel = max(
            dynamic
            * (
                self.nose_drag_area_m2 / self.nose_mass_kg
                - self.body_drag_area_m2 / body_mass
            ),
            0.0,
        )
        reach = self.cord_length_m - self.stroke_m
        if decel > 0.0 and speed**2 < 2.0 * decel * reach:
            time = speed / decel  # Stops short: the cord is pulled tight by drift
            stretch_speed = 0.0
        elif decel > 0.0:
            time = (speed - math.sqrt(speed**2 - 2.0 * decel * reach)) / decel
            stretch_speed = speed - decel * time
        else:
            time = reach / speed
            stretch_speed = speed
        snatch = stretch_speed * math.sqrt(self.cord_stiffness_n_m * reduced)
        steps = np.linspace(0.0, time, 40)
        distances = self.stroke_m + speed * steps - 0.5 * decel * steps**2
        return Separation(
            separated=True,
            speed_m_s=speed,
            line_stretch_s=time,
            stretch_speed_m_s=stretch_speed,
            snatch_force_n=snatch,
            times_s=[float(t) for t in steps],
            distances_m=[float(min(d, self.cord_length_m)) for d in distances],
        )


@dataclass(frozen=True)
class NoseSwing:
    """Pendulum swing of the nose on its leg of the cord under the canopy.

    Attributes:
        length_m (float): Canopy attachment to the nose's centre of mass.
        damping (float): Damping ratio of the swing; the light nose is
            damped strongly by the air.
        spread_deg (float): The nose's leg hangs this far to the side of the
            body's, as the two legs spread from the attachment.
    """

    length_m: float
    damping: float = 0.35
    spread_deg: float = 12.0

    def swing(
        self,
        times_s: list[float],
        canopy_velocities: np.ndarray,
        *,
        start_s: float,
        start: np.ndarray,
    ) -> np.ndarray:
        """Integrate the swing over descent samples.

        Args:
            times_s (list[float]): Sample times, increasing.
            canopy_velocities (np.ndarray): Velocity of the attachment at
                each sample, N x 3, world frame with X up.
            start_s (float): Line stretch; the swing starts here.
            start (np.ndarray): Unit vector from the attachment to the nose
                at line stretch.

        Returns:
            np.ndarray: Unit vector from the attachment to the nose at each
                sample, N x 3; before line stretch it is the start vector.
        """
        times = np.asarray(times_s, dtype=float)
        nose = np.tile(_unit(start), (times.size, 1))
        if times.size < 2:
            return nose
        accel = np.gradient(canopy_velocities, times, axis=0)
        pendulum = _Pendulum(
            _unit(start), self.length_m, self.damping, math.radians(self.spread_deg)
        )
        for i in range(1, times.size):
            if times[i] <= start_s:
                continue
            begin = max(times[i - 1], start_s)
            substeps = max(math.ceil((times[i] - begin) / _SWING_STEP_S), 1)
            h = (times[i] - begin) / substeps
            for _ in range(substeps):
                pendulum.step(_DOWN - accel[i], h)
            nose[i] = pendulum.direction
        return nose


_SWING_STEP_S = 0.005
_DOWN = np.array([-_G, 0.0, 0.0])


class _Pendulum:
    """A spherical pendulum hanging from the canopy attachment."""

    def __init__(
        self, direction: np.ndarray, length: float, damping: float, spread: float
    ):
        self.direction, self.length, self.damping, self.spread = (
            direction,
            length,
            damping,
            spread,
        )
        self.rate = np.zeros(3)

    def step(self, apparent: np.ndarray, h: float) -> None:
        """Advance by h seconds under the apparent gravity ``g - a``."""
        magnitude = float(np.linalg.norm(apparent))
        settle = apparent / magnitude
        if self.spread:
            side = np.cross(settle, np.array([0.0, 0.0, 1.0]))
            if float(np.linalg.norm(side)) > 1e-9:
                settle = settle * math.cos(self.spread) + _unit(side) * math.sin(
                    self.spread
                )
        omega = math.sqrt(magnitude / self.length)
        u = self.direction
        # Restoring pull toward the settled direction, damping, and the
        # centripetal term that keeps u a unit vector
        pull = omega**2 * (settle - float(settle @ u) * u)
        if float(np.linalg.norm(pull)) < 1e-12 and float(settle @ u) < 0.0:
            pull = omega**2 * _unit(np.cross(u, np.array([0.0, 1.0, 0.0])))
        accel = (
            pull
            - 2.0 * self.damping * omega * self.rate
            - float(self.rate @ self.rate) * u
        )
        rate = self.rate + h * accel
        self.rate = rate - float(rate @ u) * u
        self.direction = _unit(u + h * self.rate)


def _unit(vector: np.ndarray) -> np.ndarray:
    """The vector scaled to unit length."""
    vector = np.asarray(vector, dtype=float)
    result: np.ndarray = vector / float(np.linalg.norm(vector))
    return result
