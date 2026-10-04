"""What EDITH varies from run to run, and the bell curves it draws it from.

Every input of a run is drawn from a distribution through its inverse
cumulative distribution function, so any sampler that fills the unit cube
(random, Latin hypercube, scrambled Sobol) can drive it. The distributions
are the ``SiteConfig`` below; **the numbers in it are placeholders** (see
``ASSUMED``) to be replaced with the team's own site data and motor tests.
Nominal values (the day's wind, the rail, the pad's temperature and pressure)
come from the rocket's own profile, so the draws are spread around the
conditions the rest of the simulator already uses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from flight_sim.environment.wind import DIRECTION_LAYERS, SPEED_LAYERS, LayeredWind

# The order of the unit-cube coordinates of a run
INPUTS = (
    "thrust",
    "mass",
    "drag",
    "rail_elevation",
    "rail_azimuth",
    "temperature",
    "pressure",
    "wind_speed",
    "wind_direction",
    "speed_phase_1",
    "speed_phase_2",
    "speed_phase_3",
    "direction_phase_1",
    "direction_phase_2",
    "direction_phase_3",
    "layer_scale",
    "canopy_cd",
    "charge_pressure",
)
DIMENSIONS = len(INPUTS)

# Everything in the site file is a placeholder until the team replaces it
ASSUMED = (
    "Every number in this file is an assumed placeholder, not measured data: "
    "replace the wind and temperature spread with Spaceport America's climate "
    "for the launch week, the thrust and mass spread with the propulsion "
    "team's motor tests, and the ratings with the canopy's real strength."
)


@dataclass(frozen=True)
class SiteConfig:  # pylint: disable=too-many-instance-attributes
    """The launch site, the spread of each input and the ratings of the canopies.

    Attributes:
        name (str): Name of the site.
        wind_mean_m_s (float | None): Mean of the day's wind speed on the pad;
            the rocket profile's own wind when None.
        wind_weibull_shape (float): Shape of the Weibull curve of the wind
            speed (2 is a Rayleigh curve, typical of surface winds).
        wind_launch_limit_m_s (float): Wind speed above which the launch is
            called off, so the Weibull curve is cut there.
        wind_direction_sd_deg (float): Spread of the wind direction about the
            profile's, normal curve.
        temperature_sd_k (float): Spread of the pad temperature (normal).
        pressure_sd_pa (float): Spread of the pad pressure (normal).
        thrust_sd (float): Spread of the thrust as a share of nominal (normal).
        mass_sd (float): Spread of the dry mass as a share of nominal.
        drag_sd (float): Spread of the axial drag coefficient (aerodynamic
            model error), share of nominal.
        rail_elevation_sd_deg (float): Spread of the rail's tilt (normal).
        rail_azimuth_sd_deg (float): Spread of the rail's heading (normal).
        layer_scale_sd (float): Spread of how strong the wind's layers are,
            share of the nominal layers (normal about 1).
        canopy_cd_sd (float): Spread of the canopies' drag coefficient.
        charge_pressure_sd (float): Spread of the ejection charge's pressure.
        rated_load_g (float): The opening load (in g of the rocket's weight) a
            canopy and its harness are assumed to carry. Over it the canopy
            tears.
        target_apogee_m (float): Height above the pad the flight is aimed at
            (30,000 ft, the high IREC category).
        target_apogee_tolerance (float): How close to the target counts as
            "within range", as a share of the target (0.05 is plus or minus 5%).
    """

    name: str = "Spaceport America (IREC)"
    wind_mean_m_s: float | None = None
    wind_weibull_shape: float = 2.0
    wind_launch_limit_m_s: float = 11.0
    wind_direction_sd_deg: float = 30.0
    temperature_sd_k: float = 5.0
    pressure_sd_pa: float = 300.0
    thrust_sd: float = 0.05
    mass_sd: float = 0.02
    drag_sd: float = 0.05
    rail_elevation_sd_deg: float = 0.5
    rail_azimuth_sd_deg: float = 1.0
    layer_scale_sd: float = 0.25
    canopy_cd_sd: float = 0.07
    charge_pressure_sd: float = 0.10
    rated_load_g: float = 15.0
    target_apogee_m: float = 9144.0
    target_apogee_tolerance: float = 0.05
    note: str = field(default=ASSUMED)

    def save(self, path: str | Path) -> None:
        """Write the settings as JSON, to edit."""
        Path(path).write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SiteConfig:
        """Read settings written by ``save``; missing keys keep their defaults."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown settings in {path}: {', '.join(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class Nominal:
    """The conditions the draws are spread around, from the rocket's profile."""

    wind_speed_m_s: float
    wind_azimuth_deg: float
    rail_elevation_deg: float
    rail_azimuth_deg: float
    pad_temperature_k: float
    pad_pressure_pa: float
    pad_elevation_m: float


@dataclass(frozen=True)
class Variation:  # pylint: disable=too-many-instance-attributes
    """One run's inputs."""

    thrust: float
    mass: float
    drag: float
    rail_elevation_deg: float
    rail_azimuth_deg: float
    temperature_k: float
    pressure_pa: float
    wind_speed_m_s: float
    wind_azimuth_deg: float
    speed_phases: tuple[float, float, float]
    direction_phases: tuple[float, float, float]
    layer_scale: float
    canopy_cd: float
    charge_pressure: float
    seed: int

    def wind(self, ground_m: float) -> LayeredWind:
        """The layered wind of this run."""
        from flight_sim.units import scalar  # pylint: disable=import-outside-toplevel

        return LayeredWind(
            speed=scalar(self.wind_speed_m_s, "m/s"),
            from_azimuth=scalar(self.wind_azimuth_deg, "deg"),
            ground_m=ground_m,
            speed_layers=tuple(
                (length, share * self.layer_scale, phase)
                for (length, share, _), phase in zip(
                    SPEED_LAYERS, self.speed_phases, strict=True
                )
            ),
            direction_layers=tuple(
                (length, degrees * self.layer_scale, phase)
                for (length, degrees, _), phase in zip(
                    DIRECTION_LAYERS, self.direction_phases, strict=True
                )
            ),
        )

    def mean_wind(self, ground_m: float, top_m: float, points: int = 24) -> np.ndarray:
        """Mean horizontal wind (east, north) over the climb (a surrogate feature)."""
        wind = self.wind(ground_m)
        heights = np.linspace(0.0, top_m, points)
        samples = np.array([wind.velocity(ground_m + h)[1:3] for h in heights])
        result: np.ndarray = samples.mean(axis=0)
        return result


def _normal(u: float, mean: float, sd: float) -> float:
    return mean + sd * float(stats.norm.ppf(min(max(u, 1e-9), 1 - 1e-9)))


def draw(unit: np.ndarray, site: SiteConfig, nominal: Nominal, seed: int) -> Variation:
    """Turn one point of the unit cube into the inputs of a run.

    Args:
        unit (np.ndarray): ``DIMENSIONS`` numbers in (0, 1), in the order of
            ``INPUTS``.
        site (SiteConfig): The distributions.
        nominal (Nominal): Where they are centred.
        seed (int): Seed of the flight computer's sensor noise in this run.
    """
    u = dict(zip(INPUTS, (float(x) for x in unit), strict=True))
    mean = (
        site.wind_mean_m_s if site.wind_mean_m_s is not None else nominal.wind_speed_m_s
    )
    scale = max(mean, 1e-6) / math.gamma(1.0 + 1.0 / site.wind_weibull_shape)
    curve = stats.weibull_min(site.wind_weibull_shape, scale=scale)
    cut = float(
        curve.cdf(site.wind_launch_limit_m_s)
    )  # winds above the limit are not flown
    speed = float(curve.ppf(min(max(u["wind_speed"], 1e-9), 1 - 1e-9) * cut))
    two_pi = 2.0 * math.pi
    return Variation(
        thrust=max(_normal(u["thrust"], 1.0, site.thrust_sd), 0.5),
        mass=max(_normal(u["mass"], 1.0, site.mass_sd), 0.5),
        drag=max(_normal(u["drag"], 1.0, site.drag_sd), 0.5),
        rail_elevation_deg=min(
            _normal(
                u["rail_elevation"],
                nominal.rail_elevation_deg,
                site.rail_elevation_sd_deg,
            ),
            90.0,
        ),
        rail_azimuth_deg=_normal(
            u["rail_azimuth"], nominal.rail_azimuth_deg, site.rail_azimuth_sd_deg
        ),
        temperature_k=_normal(
            u["temperature"], nominal.pad_temperature_k, site.temperature_sd_k
        ),
        pressure_pa=_normal(
            u["pressure"], nominal.pad_pressure_pa, site.pressure_sd_pa
        ),
        wind_speed_m_s=speed,
        wind_azimuth_deg=_normal(
            u["wind_direction"], nominal.wind_azimuth_deg, site.wind_direction_sd_deg
        ),
        speed_phases=(
            two_pi * u["speed_phase_1"],
            two_pi * u["speed_phase_2"],
            two_pi * u["speed_phase_3"],
        ),
        direction_phases=(
            two_pi * u["direction_phase_1"],
            two_pi * u["direction_phase_2"],
            two_pi * u["direction_phase_3"],
        ),
        layer_scale=max(_normal(u["layer_scale"], 1.0, site.layer_scale_sd), 0.0),
        canopy_cd=max(_normal(u["canopy_cd"], 1.0, site.canopy_cd_sd), 0.5),
        charge_pressure=max(
            _normal(u["charge_pressure"], 1.0, site.charge_pressure_sd), 0.1
        ),
        seed=seed,
    )


def describe(site: SiteConfig, nominal: Nominal) -> list[dict[str, Any]]:
    """The distributions as rows for the report: input, curve, its numbers."""
    wind = (
        site.wind_mean_m_s if site.wind_mean_m_s is not None else nominal.wind_speed_m_s
    )
    n, s = nominal, site
    rows = [
        (
            "wind speed on the pad (m/s)",
            "Weibull, cut at the launch limit",
            f"mean {wind:.2f}, shape {s.wind_weibull_shape:g}, "
            f"limit {s.wind_launch_limit_m_s:g}",
        ),
        (
            "wind direction (deg)",
            "normal",
            f"mean {n.wind_azimuth_deg:.0f}, sd {s.wind_direction_sd_deg:g}",
        ),
        (
            "wind layers",
            "random phases, normal strength",
            f"strength 1.0, sd {s.layer_scale_sd:g}",
        ),
        (
            "pad temperature (K)",
            "normal",
            f"mean {n.pad_temperature_k:.1f}, sd {s.temperature_sd_k:g}",
        ),
        (
            "pad pressure (Pa)",
            "normal",
            f"mean {n.pad_pressure_pa:.0f}, sd {s.pressure_sd_pa:g}",
        ),
        ("thrust (share of nominal)", "normal", f"1.0, sd {s.thrust_sd:g}"),
        ("dry mass (share of nominal)", "normal", f"1.0, sd {s.mass_sd:g}"),
        ("axial drag (share of nominal)", "normal", f"1.0, sd {s.drag_sd:g}"),
        (
            "rail tilt from vertical (deg)",
            "normal",
            f"mean {90.0 - n.rail_elevation_deg:.1f}, sd {s.rail_elevation_sd_deg:g}",
        ),
        (
            "rail heading (deg)",
            "normal",
            f"mean {n.rail_azimuth_deg:.0f}, sd {s.rail_azimuth_sd_deg:g}",
        ),
        ("canopy drag coefficient (share)", "normal", f"1.0, sd {s.canopy_cd_sd:g}"),
        (
            "ejection charge pressure (share)",
            "normal",
            f"1.0, sd {s.charge_pressure_sd:g}",
        ),
        ("canopy rated opening load (g)", "fixed", f"{s.rated_load_g:g}"),
    ]
    return [{"input": a, "curve": b, "numbers": c} for a, b, c in rows]
