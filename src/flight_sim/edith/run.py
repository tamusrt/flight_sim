"""One EDITH run: the rocket with one draw of the inputs, flown to the ground."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from flight_sim.__main__ import (
    INVICTUS,
    MORPHEUS,
    RocketProfile,
    get_default_config,
    get_default_state,
    get_profile_properties,
)
from flight_sim.descent import Parachute, ReefedParachute, air_at
from flight_sim.edith.ascent import Ascent, fly_ascent
from flight_sim.edith.inputs import Nominal, SiteConfig, Variation
from flight_sim.environment.atmosphere import LaunchSiteAtmosphere
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.environment.wind import LayeredWind
from flight_sim.fast_reco import FastRECO
from flight_sim.integration import IntegrationConfiguration
from flight_sim.ork_profile import profile_from_ork
from flight_sim.reco import RecoveryRequest, get_reco
from flight_sim.recovery_systems import RecoveryScheme
from flight_sim.units import scalar, vector
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.engine import SolidEngine
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

_G0 = 9.80665


@dataclass(frozen=True)
class RocketSpec:
    """How to build a rocket again in another process.

    Attributes:
        kind (str): "invictus", "morpheus" or "ork".
        ork (str | None): The OpenRocket file, for "ork".
        aero (str | None): RASAero CSV, for "ork".
        motor (str | None): Motor file, for "ork".
        sim (str | None): Saved OpenRocket simulation whose launch to use.
        name (str | None): Name for the report.
    """

    kind: str = "invictus"
    ork: str | None = None
    aero: str | None = None
    motor: str | None = None
    sim: str | None = None
    name: str | None = None

    def build(self) -> RocketProfile:
        """The rocket profile."""
        if self.kind == "invictus":
            return INVICTUS
        if self.kind == "morpheus":
            return MORPHEUS
        if self.kind == "ork" and self.ork and self.aero and self.motor:
            return profile_from_ork(
                self.ork, self.aero, self.motor, sim=self.sim, name=self.name
            )
        raise ValueError("An ork rocket needs --ork, --aero and --motor")


class Rocket:
    """A rocket ready to fly with different inputs; one per process."""

    def __init__(self, spec: RocketSpec, site: SiteConfig) -> None:
        self.spec = spec
        self.site = site
        self.profile = spec.build()
        if not isinstance(self.profile.wind, LayeredWind):
            raise ValueError("EDITH needs a rocket whose wind is a LayeredWind")
        if self.profile.scheme is None:
            raise ValueError("EDITH needs a rocket with a recovery scheme")
        self.properties = get_profile_properties(self.profile)
        self.config = get_default_config(self.profile)
        rail, wind = self.profile.rail, self.profile.wind
        self.nominal = Nominal(
            wind_speed_m_s=float(wind.speed.m_as("m/s")),
            wind_azimuth_deg=float(wind.from_azimuth.m_as("deg")),
            rail_elevation_deg=float(rail.elevation.m_as("deg")),
            rail_azimuth_deg=float(rail.azimuth.m_as("deg")),
            pad_temperature_k=self.profile.pad_temperature_k,
            pad_pressure_pa=self.profile.pad_pressure_pa,
            pad_elevation_m=self.profile.pad_elevation_m,
        )

    def geometry(self) -> dict[str, float] | None:
        """Body length and diameter (metres), to show stability as % of length."""
        damping = self.properties.damping
        if damping is None:
            return None
        return {
            "length_m": float(damping.body_length_m),
            "diameter_m": float(damping.body_diameter_m),
        }

    # ----- one draw applied to the rocket --------------------------------------

    def vary(
        self, v: Variation
    ) -> tuple[RocketProperties, IntegrationConfiguration, RocketState, RecoveryScheme]:
        """The properties, truth, launch state and recovery of one run."""
        base = self.properties
        motor = base.engine
        if not isinstance(motor, SolidEngine):
            raise TypeError("EDITH varies the thrust of a solid motor")
        engine = replace(motor, thrusts=motor.thrusts * v.thrust)  # pylint: disable=no-member
        table = base.aero_table
        values = table.values.copy()
        values[..., 0] *= v.drag
        dry = base.dry_mass_properties
        dry = replace(dry, mass=dry.mass * v.mass, inertia=dry.inertia * v.mass)
        properties = replace(
            base,
            engine=engine,
            aero_table=replace(table, values=values),
            dry_mass_properties=dry,
        )
        profile = self.profile
        rail = LaunchRail(
            length=profile.rail.length,
            elevation=scalar(v.rail_elevation_deg, "deg"),
            azimuth=scalar(v.rail_azimuth_deg, "deg"),
            friction_coefficient=profile.rail.friction_coefficient,
        )
        truth = replace(
            self.config.truth,
            atmosphere=LaunchSiteAtmosphere(
                pad_elevation_m=profile.pad_elevation_m,
                pad_temperature_k=v.temperature_k,
                pad_pressure_pa=v.pressure_pa,
            ),
            wind=v.wind(profile.pad_elevation_m),
            launch_rail=rail,
        )
        config = replace(self.config, truth=truth)
        state = rail.mount(get_default_state())
        return properties, config, state, self._scheme(v)

    def _scheme(self, v: Variation) -> RecoveryScheme:
        scheme = self.profile.scheme
        assert scheme is not None
        canopies = []
        for canopy in scheme.recovery.parachutes:
            if isinstance(canopy, ReefedParachute):
                canopy = replace(
                    canopy,
                    drag_coefficient=canopy.drag_coefficient * v.canopy_cd,
                    reefed_drag_coefficient=canopy.reefed_drag_coefficient
                    * v.canopy_cd,
                )
            elif isinstance(canopy, Parachute):
                canopy = replace(
                    canopy, drag_coefficient=canopy.drag_coefficient * v.canopy_cd
                )
            canopies.append(canopy)

        def charged(charge: Any) -> Any:
            return (
                None
                if charge is None
                else replace(charge, pressure_pa=charge.pressure_pa * v.charge_pressure)
            )

        return replace(
            scheme,
            recovery=replace(scheme.recovery, parachutes=tuple(canopies)),
            apogee_charge=charged(scheme.apogee_charge),
            main_charge=charged(scheme.main_charge),
            computer=replace(scheme.computer, seed=v.seed),
        )

    # ----- flying it -----------------------------------------------------------

    def fly(
        self,
        v: Variation,
        *,
        predicted: dict[str, Any] | None = None,
        reco: str = "fast",
        calm: bool = False,
    ) -> dict[str, Any]:
        """Fly one run and report its numbers.

        Args:
            v (Variation): The run's inputs.
            predicted (dict | None): The surrogate's guess of the climb; the
                6-DOF climb is skipped and this is used in its place.
            reco (str): Name of the RECO version for the descent.
            calm (bool): Switch the gusts under the canopy off (used to compare two
                RECO versions without their different random gusts).

        Any error inside the flight (not only a maths error) is returned as a run
        with ``sim_ok`` false and the error's text, so a batch can carry on without it.
        """
        try:
            properties, config, state, scheme = self.vary(v)
            if predicted is None:
                ascent = fly_ascent(properties, state, config)
                if not ascent.reached_apogee:
                    return _failed(ascent, "no apogee")
            else:
                ascent = _from_prediction(predicted, state)
            result = self._descend(properties, config, scheme, ascent, reco, calm)
            if predicted is None:
                result["payload"] = apogee_payload(ascent)
            return result
        except Exception as error:  # pylint: disable=broad-exception-caught
            return {"sim_ok": False, "error": f"{type(error).__name__}: {error}"}

    def _descend(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
        self,
        properties: RocketProperties,
        config: IntegrationConfiguration,
        scheme: RecoveryScheme,
        ascent: Ascent,
        reco: str,
        calm: bool = False,
    ) -> dict[str, Any]:
        mass = properties.mass_properties(ascent.apogee_time_s)
        outcome = _reco_version(reco, scheme, calm).descend(
            RecoveryRequest(ascent.samples, config, mass, properties)
        )
        descent = outcome.descent
        last = descent.states[-1]
        position = last.position.m_as("m")
        velocity = last.velocity.m_as("m/s")
        deployments = descent.deployments
        stages = len(scheme.recovery.stages)
        apogee = ascent.samples[-1][1].position.m_as("m")
        result: dict[str, Any] = {
            "sim_ok": True,
            "apogee_ok": True,
            "rail_v": ascent.rail_speed_m_s,
            "rail_margin": ascent.rail_margin_cal,
            "margin_lo": ascent.lowest_margin_cal,
            "margin_hi": ascent.highest_margin_cal,
            "apogee_m": float(apogee[0]),
            "apogee_t": ascent.apogee_time_s,
            "apogee_east": float(apogee[1]),
            "apogee_north": float(apogee[2]),
            "land_east": float(position[1]),
            "land_north": float(position[2]),
            "drift": float(np.hypot(position[1], position[2])),
            "land_v_vert": abs(float(velocity[0])),
            "land_speed": float(np.linalg.norm(velocity)),
            "descent_s": float(descent.times_s[-1] - ascent.apogee_time_s),
            "landed": bool(descent.landed),
            "separated": bool(outcome.separated),
            "stages": stages,
            "deployed": len(deployments),
            "all_open": len(deployments) >= stages,
            "drogue_v": outcome.drogue_rate_m_s(),
            "main_alt": deployments[-1].altitude_m if deployments else None,
            "peak_force_n": max((d.peak_force_n for d in deployments), default=0.0),
            "load_ratio": max((d.peak_load_g for d in deployments), default=0.0)
            / self.site.rated_load_g,
            "detect_delay_s": (
                None
                if outcome.apogee.detected_s is None
                else outcome.apogee.detected_s - ascent.apogee_time_s
            ),
        }
        if len(ascent.samples) > 2:  # a flown climb, not the surrogate's two points
            result["history"] = flight_history(config, ascent, descent)
        return result


def _reco_version(name: str, scheme: RecoveryScheme, calm: bool) -> Any:
    """The RECO version called ``name``; with ``calm``, one flown without gusts."""
    if not calm:
        return get_reco(name, scheme)
    if name == "fast":
        return FastRECO(scheme, turbulence=0.0)
    kind: Any = type(scheme)
    if not hasattr(scheme, "swing"):  # this scheme has no gusting canopy swing
        return get_reco(name, scheme)

    def no_gusts(self: Any, centre_of_gravity_m: float) -> Any:
        return replace(kind.swing(self, centre_of_gravity_m), turbulence=0.0)

    # the same scheme, whose swing model has its gusts turned off
    calm_kind = type(kind.__name__, (kind,), {"swing": no_gusts})
    calm_scheme = calm_kind(**{f.name: getattr(scheme, f.name) for f in fields(scheme)})
    return get_reco(name, calm_scheme)


_PATH_STEP_S = 1.0  # spacing of the climb's points in the drawn path
_DESCENT_POINTS = 30  # points of the descent in the drawn path


def flight_history(
    config: IntegrationConfiguration, ascent: Ascent, descent: Any
) -> dict[str, Any]:
    """The flight over time, for the EDITH page's charts and picture.

    ``t`` and ``alt`` are the climb every 0.1 s (metres above the pad); ``mt`` and
    ``m`` the stability margin (calibres) while faster than 30 m/s; ``mach`` the Mach
    number (speed through the air over the speed of sound) at the climb's times;
    ``path`` the whole
    flight as [time, east, north, height] in seconds and metres, coarser.
    """
    times = [t for t, _ in ascent.samples]
    heights = [float(state.position.m_as("m")[0]) for _, state in ascent.samples]
    mach = []
    for (_, state), height in zip(ascent.samples, heights, strict=True):
        air, wind = air_at(config, height)
        through_air = float(np.linalg.norm(state.velocity.m_as("m/s") - wind))
        mach.append(round(through_air / air.speed_of_sound, 4))
    path: list[list[float]] = []
    next_t = 0.0
    for t, state in ascent.samples:
        if t >= next_t or t == times[-1]:
            pos = state.position.m_as("m")
            path.append(
                [
                    round(t, 2),
                    round(float(pos[1])),
                    round(float(pos[2])),
                    round(float(pos[0])),
                ]
            )
            next_t = t + _PATH_STEP_S
    down_t = np.asarray(descent.times_s, dtype=float)
    if len(down_t) > 1:
        picks = np.unique(
            np.linspace(0, len(down_t) - 1, _DESCENT_POINTS + 1).round().astype(int)
        )[1:]
        for i in picks:
            pos = descent.states[int(i)].position.m_as("m")
            path.append(
                [
                    round(float(down_t[i]), 2),
                    round(float(pos[1])),
                    round(float(pos[2])),
                    round(max(float(pos[0]), 0.0)),
                ]
            )
    return {
        "t": [round(t, 2) for t in times],
        "alt": [round(h, 1) for h in heights],
        "mt": [round(t, 2) for t, _ in ascent.margin_track],
        "m": [round(m, 3) for _, m in ascent.margin_track],
        "mach": mach,
        "path": path,
    }


def _failed(ascent: Ascent, why: str) -> dict[str, Any]:
    """The numbers of a run that never reached apogee."""
    return {
        "sim_ok": True,
        "apogee_ok": False,
        "rail_v": ascent.rail_speed_m_s,
        "rail_margin": ascent.rail_margin_cal,
        "margin_lo": ascent.lowest_margin_cal,
        "margin_hi": ascent.highest_margin_cal,
        "error": why,
    }


def apogee_payload(ascent: Ascent) -> dict[str, Any]:
    """The climb's end as plain numbers, to train the surrogate."""
    state = ascent.apogee_state
    q = state.orientation
    return {
        "time": ascent.apogee_time_s,
        "position": [float(x) for x in state.position.m_as("m")],
        "velocity": [float(x) for x in state.velocity.m_as("m/s")],
        "quaternion": [q.q_w, q.q_x, q.q_y, q.q_z],
        "rates": [float(x) for x in state.angular_velocity.m_as("rad/s")],
        "rail_v": ascent.rail_speed_m_s,
        "rail_margin": ascent.rail_margin_cal,
        "margin_lo": ascent.lowest_margin_cal,
        "margin_hi": ascent.highest_margin_cal,
    }


def _from_prediction(predicted: dict[str, Any], launch: RocketState) -> Ascent:
    """A short climb (launch and apogee) from the surrogate's guess."""
    q = predicted["quaternion"]
    apogee = replace(
        launch,
        position=vector(predicted["position"], "m"),
        velocity=vector(predicted["velocity"], "m/s"),
        angular_velocity=vector(predicted["rates"], "rad/s"),
        orientation=Quaternion(q_w=q[0], q_x=q[1], q_y=q[2], q_z=q[3]).normalized(),
        rail_start=None,
    )
    return Ascent(
        samples=[(0.0, launch), (float(predicted["time"]), apogee)],
        reached_apogee=True,
        rail_speed_m_s=predicted["rail_v"],
        rail_margin_cal=predicted["rail_margin"],
        lowest_margin_cal=predicted["margin_lo"],
        highest_margin_cal=predicted["margin_hi"],
    )


def finite(value: Any) -> bool:
    """Whether a number is a usable float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))
