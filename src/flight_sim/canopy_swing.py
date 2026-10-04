"""The rocket and its canopy as two bodies on a line, flown with the descent.

Under a canopy the rocket is not a point that hangs where the drag says: the
canopy and the rocket are two bodies joined by the suspension lines and the
shock cord, and the rocket swings about the canopy like a pendulum. Published
parachute-payload models (a two-body system with the canopy's apparent mass,
for example Guglieri's trajectory simulation and the NASA CPAS pendulum
analysis) treat it that way, and so does this module. Here the line is a
rigid, massless rod of length ``L`` from the canopy to the rocket's centre of
gravity, and ``d`` is the unit vector along it, canopy to rocket.

The rocket has mass ``M`` and feels its weight, its own drag and the line.
The canopy has the inertia ``m`` of its own mass plus the air it drags along
(its apparent mass, taken as the air in a hemisphere of its diameter), feels
its drag and the line, and its weight. With ``F_c`` and ``F_b`` the outside
forces on canopy and rocket the constrained equations are

    lambda = (m F_b.d - M F_c.d + m M L |d'|^2) / (m + M)
    M b'' = F_b - lambda d
    d'' = (F_b - M F_c / m)_perp / (M L) - |d'|^2 d

with ``lambda`` the line tension and ``_perp`` the part across ``d``. Because
the canopy's drag acts on the canopy and not on the rocket, a sideways canopy
speed is resisted by drag (a damping ratio near 0.3 for this rocket) and the
swing period is about that of a pendulum of length ``L``, as in a real drop.

How much drag the canopy gives depends on the angle between the line and the
air flowing past the canopy, the canopy's angle of attack. In a steady
descent that is the swing angle from the vertical. Within ``full_drag_deg``
the canopy gives all its drag; between that and ``no_drag_deg`` the share
falls smoothly to a few percent, the canopy still upright but spilling its
air; beyond it the canopy is collapsed, streaming behind on its lines, and
gravity does the work. The angles, the collapsed share, the apparent mass
factor and the turbulence are assumptions.
"""

import math
from dataclasses import dataclass

import numpy as np

_HEMISPHERE_VOLUME_PER_D3 = math.pi / 12.0  # Volume of a hemisphere, in D**3


@dataclass(frozen=True)
class CanopySwing:
    """The two-body model of the rocket under its canopy.

    Attributes:
        line_length_m (float): From the canopy's skirt-line confluence to the
            rocket's centre of gravity: lines, shock cord and the rocket's
            harness to its centre of gravity.
        canopy_mass_kg (float): Dry mass of the canopy and lines; the rocket's
            mass already includes it, so the rocket body is lighter by it.
        apparent_mass_factor (float): Share of the air in a hemisphere of the
            canopy diameter that moves with it (1.0 is all of it).
        full_drag_deg (float): Largest angle at which the canopy gives all
            its drag.
        no_drag_deg (float): Angle at which the canopy gives practically no
            drag.
        collapsed_drag_fraction (float): Share of its drag a collapsed canopy
            still gives, streaming behind on its lines. With none at all the
            rocket and canopy would fall together and nothing would swing
            the canopy back into the flow.
        turbulence (float): Random gust standard deviation as a share of the
            wind. Off (0) by default: the wind's own changes with height
            (``LayeredWind``) swing the canopy instead, the same on every run.
            1.5 is a gusty day, swinging the rocket up to about 20 degrees
            under the open main, the size NASA's parachute tests report.
        gust_time_s (float): Correlation time of the gusts, a few swing
            periods so the canopy keeps being pushed.
        seed (int): Seed of the gusts, so runs repeat exactly.
    """

    line_length_m: float
    canopy_mass_kg: float = 0.71
    apparent_mass_factor: float = 1.0
    full_drag_deg: float = 20.0
    no_drag_deg: float = 75.0
    collapsed_drag_fraction: float = 0.03
    turbulence: float = 0.0
    gust_time_s: float = 8.0
    seed: int = 11

    def drag_fraction(self, angle_rad: float) -> float:
        """Share of its drag the canopy gives at an angle of attack.

        Args:
            angle_rad (float): Angle between the line and the airflow past
                the canopy.

        Returns:
            float: 1 up to ``full_drag_deg``, a smooth fall to the collapsed
                share at ``no_drag_deg``, and that share beyond.
        """
        low = math.radians(self.full_drag_deg)
        high = math.radians(self.no_drag_deg)
        floor = self.collapsed_drag_fraction
        if angle_rad <= low:
            return 1.0
        if angle_rad >= high:
            return floor
        taper = 0.5 * (1.0 + math.cos(math.pi * (angle_rad - low) / (high - low)))
        return floor + (1.0 - floor) * taper

    def canopy_inertia_kg(self, air_density: float, open_diameter_m: float) -> float:
        """Canopy mass plus the air it moves with, in kg."""
        enclosed = _HEMISPHERE_VOLUME_PER_D3 * open_diameter_m**3
        return self.canopy_mass_kg + self.apparent_mass_factor * air_density * enclosed

    def accelerations(
        self,
        body_mass_kg: float,
        canopy_inertia_kg: float,
        *,
        line: np.ndarray,
        line_rate: np.ndarray,
        canopy_force: np.ndarray,
        body_force: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Accelerations of the rocket and of the line direction.

        Args:
            body_mass_kg (float): Mass of the rocket.
            canopy_inertia_kg (float): Inertia of the canopy.
            line (np.ndarray): Unit vector along the line, canopy to rocket.
            line_rate (np.ndarray): Its time derivative.
            canopy_force (np.ndarray): Outside force on the canopy: its drag
                and weight.
            body_force (np.ndarray): Outside force on the rocket: its drag
                and weight.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: The rocket's acceleration,
                the second derivative of the line direction, and the line
                tension in N.
        """
        mass, inertia, length = body_mass_kg, canopy_inertia_kg, self.line_length_m
        spin_sq = float(line_rate @ line_rate)
        tension = (
            inertia * float(body_force @ line)
            - mass * float(canopy_force @ line)
            + inertia * mass * length * spin_sq
        ) / (inertia + mass)
        across = body_force / mass - canopy_force / inertia
        across = across - float(across @ line) * line
        line_accel = across / length - spin_sq * line
        body_accel: np.ndarray = (body_force - tension * line) / mass
        return body_accel, line_accel, tension


class Gusts:
    """Ornstein-Uhlenbeck gust velocity in the horizontal plane."""

    def __init__(self, swing: CanopySwing, wind_m_s: float):
        self.rng = np.random.default_rng(swing.seed)
        self.sigma = swing.turbulence * max(wind_m_s, 1.0)
        self.tau = swing.gust_time_s
        self.value = np.zeros(3)

    def advance(self, seconds: float) -> np.ndarray:
        """Step the gust by ``seconds`` and return it."""
        decay = math.exp(-seconds / self.tau)
        kick = self.rng.normal(0.0, self.sigma * math.sqrt(1.0 - decay**2), 3)
        self.value = self.value * decay + kick
        self.value[0] = 0.0
        return self.value
