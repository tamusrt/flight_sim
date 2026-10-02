"""When the flight computer would actually fire the recovery events.

The descent model on its own releases the canopy a fixed time after the true
apogee and disreefs at the true altitude. A real flight computer only sees a
barometer and an accelerometer, so it detects apogee late and judges altitude
with the standard atmosphere. This module models that, following the voting
logic the recovery team describes for the Blue Raven: apogee is declared when
at least two of three votes agree,

* pressure: the filtered pressure has been rising, so the rocket is falling;
* velocity: the integrated vertical velocity has reached zero;
* attitude: the rocket has turned more than 90 degrees from vertical.

The charge then fires after a programmed delay, the nose is ejected and the
canopy starts to inflate at line stretch (see ``recovery_motion``). The main
event (the reef cut) is commanded when the barometric altitude falls through
its setting.

Sensor noise, filter time constant and accelerometer bias are assumptions,
chosen to be typical of hobby flight computers.
"""

from dataclasses import dataclass, replace

import numpy as np

from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import (
    DescentResult,
    Parachute,
    RecoverySystem,
    ReefedParachute,
    air_at,
    simulate_descent,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import EjectionCharge, Separation
from flight_sim.swing_descent import SwingTrace, simulate_swing_descent
from flight_sim.utilities.dcm import body_to_world
from flight_sim.vehicle.rocket_state import RocketState

Sample = tuple[float, RocketState]

# International standard atmosphere constants for the barometric altitude
_ISA_LAPSE_K_PER_M = 0.0065
_ISA_SEA_LEVEL_K = 288.15
_ISA_EXPONENT = 0.190263  # R * L / (g * M)


@dataclass(frozen=True)
class FlightComputer:
    """Barometric and inertial event logic of the avionics.

    Attributes:
        sample_rate_hz (float): Rate the logic runs at.
        baro_noise_pa (float): Standard deviation of the pressure noise.
        baro_filter_s (float): Time constant of the pressure low-pass filter.
        accel_bias_m_s2 (float): Vertical accelerometer bias; integrated, it
            makes the inertial velocity read high, which delays its vote.
        pressure_window_s (float): The filtered pressure must have risen over
            this window for the pressure vote.
        lockout_s (float): No apogee vote is counted before this time, so the
            boost cannot trigger the charge.
        apogee_delay_s (float): Programmed delay from detecting apogee to
            firing the separation charge.
        main_altitude_m (float): Barometric altitude above the pad at which
            the main event (reef cut) is commanded.
        seed (int): Seed of the sensor noise, so runs repeat exactly.
    """

    sample_rate_hz: float = 100.0
    baro_noise_pa: float = 3.0
    baro_filter_s: float = 0.25
    accel_bias_m_s2: float = 0.05
    pressure_window_s: float = 0.5
    lockout_s: float = 10.0
    apogee_delay_s: float = 0.0
    main_altitude_m: float = 609.6
    seed: int = 7

    def sense(
        self, samples: list[Sample], config: IntegrationConfiguration
    ) -> "SensorTrace":
        """Resample the flight and produce what the sensors would read.

        Args:
            samples (list[Sample]): Time and state, in time order.
            config (IntegrationConfiguration): Atmosphere for the pressure.

        Returns:
            SensorTrace: Times, true altitude, filtered pressure, barometric
                altitude, inertial velocity and tilt on a uniform grid.
        """
        times = np.array([t for t, _ in samples])
        states = [s for _, s in samples]
        altitude = np.array([float(s.position.m_as("m")[0]) for s in states])
        vertical = np.array([float(s.velocity.m_as("m/s")[0]) for s in states])
        # Angle of the nose from vertical: the world X (up) part of body x
        up = np.array([float(body_to_world(s.orientation)[0, 0]) for s in states])
        tilt = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
        dt = 1.0 / self.sample_rate_hz
        grid = np.arange(times[0], times[-1] + 1e-9, dt)
        alt = np.interp(grid, times, altitude)
        pressure = np.array([air_at(config, h)[0].pressure for h in alt])
        noise = np.random.default_rng(self.seed).normal(
            0.0, self.baro_noise_pa, grid.size
        )
        filtered = np.empty_like(pressure)
        filtered[0] = pressure[0]
        gain = dt / (self.baro_filter_s + dt)
        for i in range(1, grid.size):
            filtered[i] = filtered[i - 1] + gain * (
                pressure[i] + noise[i] - filtered[i - 1]
            )
        pad = filtered[0]
        baro_altitude = (
            _ISA_SEA_LEVEL_K
            / _ISA_LAPSE_K_PER_M
            * (1.0 - (filtered / pad) ** _ISA_EXPONENT)
        )
        inertial = np.interp(grid, times, vertical) + self.accel_bias_m_s2 * (
            grid - grid[0]
        )
        return SensorTrace(
            times=grid,
            altitude=alt,
            pressure=filtered,
            baro_altitude=baro_altitude,
            inertial_velocity=inertial,
            tilt_deg=np.interp(grid, times, tilt),
        )

    def detect_apogee(self, trace: "SensorTrace") -> "ApogeeDetection":
        """Run the two-of-three apogee vote over a sensor trace.

        Args:
            trace (SensorTrace): What the sensors read.

        Returns:
            ApogeeDetection: When each vote first agreed and when apogee was
                declared, or None where it never happened.
        """
        window = max(round(self.pressure_window_s * self.sample_rate_hz), 1)
        armed = trace.times >= trace.times[0] + self.lockout_s
        rising = np.zeros(trace.times.size, dtype=bool)
        rising[window:] = trace.pressure[window:] > trace.pressure[:-window]
        votes = {
            "pressure": armed & rising,
            "velocity": armed & (trace.inertial_velocity <= 0.0),
            "attitude": armed & (trace.tilt_deg > 90.0),
        }
        count = sum(v.astype(int) for v in votes.values())

        def first(mask: np.ndarray) -> float | None:
            index = np.flatnonzero(mask)
            return float(trace.times[index[0]]) if index.size else None

        return ApogeeDetection(
            pressure_vote_s=first(votes["pressure"]),
            velocity_vote_s=first(votes["velocity"]),
            attitude_vote_s=first(votes["attitude"]),
            detected_s=first(count >= 2),
        )

    def main_command(self, trace: "SensorTrace", after_s: float) -> float | None:
        """Time the barometric altitude falls through the main setting."""
        mask = (trace.times > after_s) & (trace.baro_altitude <= self.main_altitude_m)
        index = np.flatnonzero(mask)
        return float(trace.times[index[0]]) if index.size else None


@dataclass(frozen=True)
class SensorTrace:
    """Sensor readings on the flight computer's uniform time grid."""

    times: np.ndarray
    altitude: np.ndarray  # True altitude above the pad, m
    pressure: np.ndarray  # Filtered static pressure, Pa
    baro_altitude: np.ndarray  # Standard-atmosphere altitude above the pad, m
    inertial_velocity: np.ndarray  # Integrated vertical velocity, m/s
    tilt_deg: np.ndarray  # Angle of the body axis from vertical


@dataclass(frozen=True)
class ApogeeDetection:
    """When each apogee vote first agreed, and when apogee was declared."""

    pressure_vote_s: float | None
    velocity_vote_s: float | None
    attitude_vote_s: float | None
    detected_s: float | None


@dataclass(frozen=True)
class RecoveryPlan:
    """The recovery timeline as the flight computer would fly it.

    Attributes:
        recovery (RecoverySystem): The recovery system with its release
            delay and disreef height set from the flight computer.
        descent (DescentResult): The descent flown with that system.
        swing (SwingTrace | None): The line and canopy along the descent, when
            it was flown with the rocket swinging under its canopy.
        apogee (ApogeeDetection): The apogee votes.
        fire_s (float): Time the separation charge fires.
        separation (Separation): Nose ejection and line stretch.
        line_stretch_s (float): Time the canopy starts to inflate.
        main_command_s (float | None): Time the main event is commanded.
        main_true_altitude_m (float | None): True altitude above the pad at
            that moment; it differs from the setting through filter lag and
            the standard-atmosphere assumption.
    """

    recovery: RecoverySystem
    descent: DescentResult
    swing: SwingTrace | None
    apogee: ApogeeDetection
    fire_s: float
    separation: Separation
    line_stretch_s: float
    main_command_s: float | None
    main_true_altitude_m: float | None


def plan_recovery(
    flight: list[Sample],
    config: IntegrationConfiguration,
    recovery: RecoverySystem,
    computer: FlightComputer,
    charge: EjectionCharge,
    *,
    swing: CanopySwing | None = None,
) -> RecoveryPlan:
    """Fly the descent with the events where the flight computer puts them.

    The rocket first coasts past apogee with no canopy so the computer can
    vote; the charge fires after its delay, the canopy is released at line
    stretch, and the descent is flown once to find when the barometric main
    command comes and then again with the disreef at that true altitude.

    Args:
        flight (list[Sample]): The ascent, ending at the true apogee.
        config (IntegrationConfiguration): Atmosphere and gravity.
        recovery (RecoverySystem): The canopies, with nominal settings.
        computer (FlightComputer): The avionics.
        charge (EjectionCharge): The separation charge and shock cord.
        swing (CanopySwing | None): When given, the descents are flown with
            the rocket swinging under its canopy instead of as a point mass.

    Returns:
        RecoveryPlan: The timeline and the descent flown with it.
    """
    apogee_time, apogee_state = flight[-1]
    coast = simulate_descent(
        apogee_time,
        apogee_state,
        config,
        replace(recovery, parachutes=()),
        max_time_s=computer.apogee_delay_s + 30.0,
    )
    coasting = flight + list(zip(coast.times_s, coast.states, strict=True))
    detection = computer.detect_apogee(computer.sense(coasting, config))
    detected = detection.detected_s if detection.detected_s is not None else apogee_time
    fire = detected + computer.apogee_delay_s

    separation = _eject(_state_at(coasting, fire), config, charge)
    line_stretch = fire + separation.line_stretch_s

    planned = _retime(recovery, line_stretch - apogee_time, None)
    descent, trace = _fly(apogee_time, apogee_state, config, planned, swing)
    command = computer.main_command(
        computer.sense(
            flight + list(zip(descent.times_s, descent.states, strict=True)), config
        ),
        after_s=detected,
    )
    true_altitude = None
    if command is not None:
        true_altitude = float(
            np.interp(
                command,
                descent.times_s,
                [float(s.position.m_as("m")[0]) for s in descent.states],
            )
        )
        planned = _retime(recovery, line_stretch - apogee_time, true_altitude)
        descent, trace = _fly(apogee_time, apogee_state, config, planned, swing)
    return RecoveryPlan(
        recovery=planned,
        descent=descent,
        swing=trace,
        apogee=detection,
        fire_s=fire,
        separation=separation,
        line_stretch_s=line_stretch,
        main_command_s=command,
        main_true_altitude_m=true_altitude,
    )


def _fly(
    apogee_time_s: float,
    apogee_state: RocketState,
    config: IntegrationConfiguration,
    recovery: RecoverySystem,
    swing: CanopySwing | None,
) -> tuple[DescentResult, SwingTrace | None]:
    """Fly the descent as a point mass, or with the swing when it is given."""
    if swing is None:
        return simulate_descent(apogee_time_s, apogee_state, config, recovery), None
    result = simulate_swing_descent(
        apogee_time_s, apogee_state, config, recovery, swing=swing
    )
    return result.descent, result.swing


def _retime(
    recovery: RecoverySystem, release_delay_s: float, main_altitude_m: float | None
) -> RecoverySystem:
    """Set every timed release to the delay and every height to the main."""
    parachutes: list[Parachute | ReefedParachute] = []
    for canopy in recovery.parachutes:
        if isinstance(canopy, ReefedParachute):
            canopy = replace(canopy, deploy_delay_s=release_delay_s)
            if main_altitude_m is not None:
                canopy = replace(canopy, disreef_altitude_m=main_altitude_m)
        elif canopy.deploy_altitude_m is None:
            canopy = replace(canopy, deploy_delay_s=release_delay_s)
        elif main_altitude_m is not None:
            canopy = replace(canopy, deploy_altitude_m=main_altitude_m)
        parachutes.append(canopy)
    return replace(recovery, parachutes=tuple(parachutes))


def _eject(
    state: RocketState, config: IntegrationConfiguration, charge: EjectionCharge
) -> Separation:
    """Fire the charge in the air the rocket is flying through."""
    air, wind = air_at(config, float(state.position.m_as("m")[0]))
    airspeed = float(np.linalg.norm(state.velocity.m_as("m/s") - wind))
    return charge.separate(
        float(state.current_mass.m_as("kg")), airspeed, air.air_density
    )


def _state_at(samples: list[Sample], time_s: float) -> RocketState:
    """The sample nearest a time."""
    times = np.array([t for t, _ in samples])
    return samples[int(np.argmin(np.abs(times - time_s)))][1]
