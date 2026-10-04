"""FastRECO: the descent without the details that do not move the landing point.

It follows the same story as the full model (apogee votes, separation charge,
tumble to line stretch, drogue, barometric main command), and differs in four
ways, each chosen so that the landing point and the loads stay close:

1. **A canopy opens as an event, not an integration.** The opening is solved
   along the line of flight with the same inflation law as the full model
   (``descent.inflation_fractions``): the speed through the air falls by
   ``dv/ds`` from drag and gravity over the fill distance, in closed form for
   each short stretch. The result is the speed after the canopy has settled,
   the time it took, the distance covered and the peak load, with no small
   steps and no shock cord spring.
2. **After the last canopy is open, the descent is vertical** with the wind
   carrying the rocket sideways: the fall speed is the one drag and gravity
   give at the local density, and the sideways velocity relaxes to the wind
   with the time constant drag gives (about half a second under a main). The
   rocket does not swing. The stage before that (the tumble, the drogue) is
   flown as a 3D point mass, as in the full model.
3. **The step grows once the fall has settled** (terminal velocity), but
   never past a guard: a step may not cross more than ``guard_m`` of height,
   so the layers of the wind are not skipped, and it is cut short to land
   exactly on every event (a canopy opening, a charge, the barometer's main
   altitude, the ground). After any event the steps start small again.
4. **The barometer is followed along the way down** (the filter's lag and the
   standard-atmosphere altitude), so the main command comes from the same
   reading the flight computer would use, without flying the descent twice.
   The filtered pressure's noise (under a pascal) is left out.

Under the last canopy the wind gusts, as in the full model: a random gust
(``CanopySwing``'s strength and correlation time) is added to the wind, stepped
exactly however long the step, one draw per step. Each descent gets its own
gusts, seeded from where and when the last canopy opened, so a batch's flights
differ and a repeated flight is the same. ``turbulence=0`` turns them off.

What is not simulated is the pendulum swing of the rocket under the canopy and
the bounce of the shock cord, and the nose section leaving the rocket. See
``tests/test_fast_reco.py`` for how close the landing point and loads are to
``FullRECO``.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, replace
from typing import ClassVar, NamedTuple

import numpy as np

from flight_sim import tumble
from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import (
    LOAD_WINDOW_S,
    SETTLING_SPAN,
    Deployment,
    DescentResult,
    DragStage,
    air_at,
    inflation_fractions,
    launch_elevation_m,
    simulate_descent,
)
from flight_sim.flight_computer import (
    _ISA_EXPONENT,
    _ISA_LAPSE_K_PER_M,
    _ISA_SEA_LEVEL_K,
    ApogeeDetection,
    Sample,
    SensorTrace,
    _eject,
    _state_at,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.reco import RECO, RecoveryOutcome, RecoveryRequest, register
from flight_sim.recovery_motion import Separation
from flight_sim.units import vector
from flight_sim.vehicle.rocket_state import RocketState

_G0 = 9.80665
_NO_SEPARATION = Separation(False, 0.0, math.inf, 0.0, 0.0, [0.0], [0.0])
_OPENING_POINTS = 49  # Points of the opening's solution along its fill distance
_MAX_STEPS = 200_000


@dataclass(frozen=True)
class FastRECO(RECO):
    """The quick descent; see the module docstring for what it leaves out.

    Attributes:
        guard_m (float): The most height a step may cover once settled, so the
            wind's layers (the shortest about 300 m long) are all seen.
        settled_step_s (float): Longest step in the 3D stage once settled.
        moving_step_s (float): Longest step in the 3D stage before that.
        main_step_s (float): Longest step of the vertical stage once settled.
        main_moving_step_s (float): Longest step of the vertical stage while
            the fall speed is still changing.
        settled_tolerance (float): The fall is settled when the net
            acceleration is below this share of gravity (3D stage), or the
            speed is within this share of terminal velocity (vertical stage).
        coast_s (float): How long past the charge the apogee votes are looked
            for before the search is widened.
        tumble_output_s (float): Spacing of the tumble's samples.
        turbulence (float): Gust strength under the last canopy, as a share of
            the ground wind speed (as ``CanopySwing.turbulence``); 0 for none.
        gust_time_s (float): Correlation time of the gusts.
    """

    name: ClassVar[str] = "fast"

    guard_m: float = 25.0
    settled_step_s: float = 1.0
    moving_step_s: float = 0.1
    main_step_s: float = 2.0
    main_moving_step_s: float = 0.25
    settled_tolerance: float = 0.05
    coast_s: float = 10.0
    tumble_output_s: float = 0.25
    turbulence: float = CanopySwing.turbulence
    gust_time_s: float = CanopySwing.gust_time_s

    def descend(self, request: RecoveryRequest) -> RecoveryOutcome:
        """Fly the descent; same request and outcome as every RECO version."""
        return _Fast(self, request).fly()


register(FastRECO)


class _Solved(NamedTuple):
    """The solution of one canopy opening."""

    elapsed_s: float  # until the canopy has settled
    filled_s: float  # until it first reaches its full size
    velocity: np.ndarray  # after it has settled
    move_m: np.ndarray  # displacement over the opening
    peak_force_n: float
    airspeed_m_s: float  # at the start


def _opening(  # pylint: disable=too-many-locals,too-many-positional-arguments
    stage: DragStage,
    cda_before: float,
    velocity: np.ndarray,
    wind: np.ndarray,
    density: float,
    mass_kg: float,
) -> _Solved:
    """Solve one canopy opening along the line of flight.

    The speed through the air ``v`` changes over the path length ``s`` as
    ``d(v^2)/ds = 2 g_s - (rho CdA(s) / m) v^2``, with ``g_s`` gravity along the
    path and ``CdA(s)`` the drag area, growing with the inflation law of the
    full model. Each short stretch has the exact solution for a drag area
    held at its middle value.

    Returns:
        The times to fill and to settle, the velocity after it, the
        displacement over it (path through the air plus the wind's carry), the
        peak drag force on the rocket in N and the speed through the air at the
        start.
    """
    relative = velocity - wind
    v0 = float(np.linalg.norm(relative))
    direction = relative / v0 if v0 > 1e-6 else np.array([-1.0, 0.0, 0.0])
    fill = stage.fill_distance_m
    if fill <= 0.0:  # opens at once: nothing to solve, the drag area just rises
        force = 0.5 * density * v0**2 * (cda_before + stage.drag_area_m2)
        return _Solved(0.0, 0.0, velocity, np.zeros(3), force, v0)
    span = SETTLING_SPAN * fill
    s = np.linspace(0.0, span, _OPENING_POINTS)
    area = cda_before + stage.drag_area_m2 * inflation_fractions(s / fill)
    along = max(-float(direction[0]), -1.0) * _G0  # gravity along the path, X is up
    w = np.empty_like(s)  # v squared
    w[0] = v0 * v0
    ds = span / (_OPENING_POINTS - 1)
    for i in range(_OPENING_POINTS - 1):
        k = density * 0.5 * (area[i] + area[i + 1]) / mass_kg
        if k > 1e-12:
            decay = math.exp(-k * ds)
            w[i + 1] = w[i] * decay + (2.0 * along / k) * (1.0 - decay)
        else:
            w[i + 1] = w[i] + 2.0 * along * ds
    w = np.maximum(w, 1e-4)
    speed = np.sqrt(w)
    clock = np.concatenate(
        ([0.0], np.cumsum(0.5 * (1.0 / speed[1:] + 1.0 / speed[:-1]) * ds))
    )
    elapsed = float(clock[-1])
    filled = float(np.interp(fill, s, clock))
    new_relative = direction * float(speed[-1])
    # The full model counts the load until a second after the canopy is full.
    # A canopy too small to slow the fall keeps loading harder after it settles.
    force = float((0.5 * density * w * area).max())
    window_end = filled + LOAD_WINDOW_S
    v, area_full = float(speed[-1]), float(area[-1])
    t = elapsed
    while t < window_end:
        step = min(0.05, window_end - t)
        k = density * area_full / mass_kg
        decay = math.exp(-k * v * step)  # exact for a constant speed over the step
        w_next = (
            v * v * decay + (2.0 * along / k) * (1.0 - decay) if k > 1e-12 else v * v
        )
        v = math.sqrt(max(w_next, 1e-4))
        force = max(force, 0.5 * density * v * v * area_full)
        t += step
    return _Solved(
        elapsed,
        filled,
        wind + new_relative,
        direction * span + wind * elapsed,
        force,
        v0,
    )


class _Fast:  # pylint: disable=too-many-instance-attributes
    """One FastRECO descent: its state, schedule and samples."""

    def __init__(self, reco: FastRECO, request: RecoveryRequest) -> None:
        self.reco = reco
        self.request = request
        self.config: IntegrationConfiguration = request.config
        self.scheme = reco.scheme
        self.computer = reco.scheme.computer
        self.mass = request.apogee_mass.mass
        self.apogee_time, self.apogee_state = request.flight[-1]
        truth = request.config.truth
        self.latitude = float(truth.launch_latitude.m_as("rad"))
        self.elevation = launch_elevation_m(truth)
        self.times: list[float] = []
        self.states: list[RocketState] = []
        self.deployments: list[Deployment] = []
        self.opened: set[str] = set()
        self.cda = reco.scheme.recovery.body_drag_area_m2
        self.template = self.apogee_state
        # the schedule
        self.stretch = math.inf  # first canopy opens
        self.main_open_s: float | None = None
        self.main_fire_s: float | None = None
        self.main_command_s: float | None = None
        self.main_true_altitude_m: float | None = None
        self.main_separation: Separation | None = None
        self.detected_s = self.apogee_time
        self.detection = ApogeeDetection(None, None, None, None)
        self.coasting: list[Sample] = []
        self.trace: SensorTrace | None = None
        # the barometer
        self.pad_pressure = 0.0
        self.baro_pressure = 0.0  # filtered, Pa
        self.command_pressure = 0.0

    # ----- the apogee and the tumble -------------------------------------------

    def _detect(self) -> tuple[list[Sample], ApogeeDetection, SensorTrace]:
        """Coast past apogee with no canopy and let the computer vote."""
        flight, config, computer = self.request.flight, self.config, self.computer
        recovery = replace(self.scheme.recovery, parachutes=())
        horizon = computer.apogee_delay_s + self.reco.coast_s
        for window in (horizon, computer.apogee_delay_s + 30.0):
            coast = simulate_descent(
                self.apogee_time,
                self.apogee_state,
                config,
                recovery,
                mass_kg=self.mass,
                max_step_s=0.1,
                max_time_s=window,
            )
            coasting = flight + list(zip(coast.times_s, coast.states, strict=True))
            trace = computer.sense(coasting, config)
            detection = computer.detect_apogee(trace)
            if detection.detected_s is not None:
                break
        return coasting, detection, trace

    def _start(
        self,
    ) -> tuple[float, np.ndarray, list[Sample], Separation, float, float]:
        """Apogee votes, charge and tumble; returns where the 3D stage starts."""
        coasting, detection, self.trace = self._detect()
        self.detection = detection
        self.coasting = coasting
        scheme, config = self.scheme, self.config
        self.detected_s = (
            detection.detected_s
            if detection.detected_s is not None
            else self.apogee_time
        )
        fire = self.detected_s + self.computer.apogee_delay_s
        charge = scheme.apogee_charge
        separation = (
            _NO_SEPARATION
            if charge is None
            else _eject(_state_at(coasting, fire), config, charge, self.mass)
        )
        self.stretch = (
            fire + separation.line_stretch_s if separation.separated else math.inf
        )
        until = min(self.stretch, self.apogee_time + tumble.MAX_TUMBLE_S)
        tumbled: list[Sample] = [(self.apogee_time, self.apogee_state)]
        if self.request.properties is not None:
            tumbled = tumble.resample(
                tumble.fly(
                    self.apogee_time,
                    self.apogee_state,
                    self.request.properties,
                    config,
                    until_s=until,
                    fire_s=fire if separation.separated else None,
                    separation_speed_m_s=separation.speed_m_s,
                    nose_share=(charge.nose_mass_kg / self.mass) if charge else 0.0,
                ),
                self.reco.tumble_output_s,
            )
        start_time, start_state = tumbled[-1]
        self.template = start_state
        y = np.concatenate(
            (start_state.position.m_as("m"), start_state.velocity.m_as("m/s"))
        )
        return start_time, y, tumbled, separation, fire, until

    # ----- the barometer -------------------------------------------------------

    def _baro_setup(self, start_s: float) -> None:
        """Pad pressure, the filter's state at the start and the command pressure."""
        flight, config, computer = self.request.flight, self.config, self.computer
        self.pad_pressure = air_at(config, float(flight[0][1].position.m_as("m")[0]))[
            0
        ].pressure
        trace = self.trace
        assert trace is not None
        self.baro_pressure = float(np.interp(start_s, trace.times, trace.pressure))
        ratio = 1.0 - _ISA_LAPSE_K_PER_M * computer.main_altitude_m / _ISA_SEA_LEVEL_K
        self.command_pressure = self.pad_pressure * ratio ** (1.0 / _ISA_EXPONENT)

    def _filter(
        self, pressure_old: float, pressure_new: float, h: float, filtered: float
    ) -> float:
        """The pressure filter after a step, for a pressure that changes linearly."""
        tau = self.computer.baro_filter_s
        slope = (pressure_new - pressure_old) / h if h > 0 else 0.0
        decay = math.exp(-h / tau)
        return (
            pressure_new - tau * slope + (filtered - pressure_old + tau * slope) * decay
        )

    # ----- the 3D stage --------------------------------------------------------

    def _gravity(self, height: float) -> float:
        return float(
            self.config.truth.gravity.magnitude(self.latitude, self.elevation + height)
        )

    def _acceleration(self, y: np.ndarray) -> np.ndarray:
        air, wind = air_at(self.config, float(y[0]))
        relative = y[3:6] - wind
        speed = float(np.linalg.norm(relative))
        result: np.ndarray = (
            -0.5 * air.air_density * speed * self.cda / self.mass * relative
        )
        result[0] -= self._gravity(float(y[0]))
        return result

    def _derivative(self, y: np.ndarray) -> np.ndarray:
        return np.concatenate((y[3:6], self._acceleration(y)))

    def _rk4(self, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self._derivative(y)
        k2 = self._derivative(y + h / 2 * k1)
        k3 = self._derivative(y + h / 2 * k2)
        k4 = self._derivative(y + h * k3)
        result: np.ndarray = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return result

    def _step_limit(self, y: np.ndarray) -> float:
        """Longest step now: short while the fall changes, long once settled."""
        reco = self.reco
        air, wind = air_at(self.config, float(y[0]))
        speed = float(np.linalg.norm(y[3:6] - wind))
        rate = air.air_density * speed * self.cda / self.mass
        stable = 0.1 / rate if rate > 0.0 else math.inf
        net = float(np.linalg.norm(self._acceleration(y)))
        if net < reco.settled_tolerance * _G0:
            vertical = max(abs(float(y[3])), 0.5)
            return min(reco.settled_step_s, reco.guard_m / vertical, stable * 4.0)
        return min(reco.moving_step_s, stable)

    def _near_command(self, y: np.ndarray, limit: float) -> float:
        """Shorten steps as the barometer nears the main altitude."""
        if self.main_command_s is not None or not self._has_altitude_stage():
            return limit
        # distance in pressure to the command, in terms of height at this air
        air, _ = air_at(self.config, float(y[0]))
        gap = (self.command_pressure - self.baro_pressure) / max(
            air.air_density * _G0, 1.0
        )
        speed = max(abs(float(y[3])), 1.0)
        if gap < 4.0 * speed * limit + 20.0:
            return min(limit, 0.1)
        return limit

    def _has_altitude_stage(self) -> bool:
        return any(
            s.deploy_altitude_m is not None and s.name not in self.opened
            for s in self.scheme.recovery.stages
        )

    # ----- recording and events ------------------------------------------------

    def _record(self, t: float, y: np.ndarray) -> None:
        self.times.append(t)
        self.states.append(
            replace(
                self.template,
                position=vector(y[:3], "m"),
                velocity=vector(y[3:6], "m/s"),
                angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
            )
        )

    def _open_due(self, t: float, y: np.ndarray) -> tuple[float, np.ndarray, bool]:
        """Open every canopy whose time has come; returns the new time and state."""
        opened_any = False
        while True:
            stage = self._next_due(t)
            if stage is None:
                return t, y, opened_any
            air, wind = air_at(self.config, float(y[0]))
            solved = _opening(stage, self.cda, y[3:6], wind, air.air_density, self.mass)
            self.deployments.append(
                Deployment(
                    name=stage.name,
                    time_s=t,
                    altitude_m=float(y[0]),
                    airspeed_m_s=solved.airspeed_m_s,
                    inflation_time_s=solved.filled_s,
                    peak_load_g=solved.peak_force_n / (_G0 * self.mass),
                    peak_force_n=solved.peak_force_n,
                )
            )
            y = np.concatenate((y[:3] + solved.move_m, solved.velocity))
            self.cda += stage.drag_area_m2
            self.opened.add(stage.name)
            t += solved.elapsed_s
            y[0] = max(float(y[0]), 0.0)
            self._record(t, y)
            opened_any = True

    def _next_due(self, t: float) -> DragStage | None:
        for stage in self.scheme.recovery.stages:
            if stage.name in self.opened:
                continue
            if stage.after is not None and stage.after not in self.opened:
                continue
            if stage.deploy_altitude_m is None:
                if t >= self.stretch - 1e-9:
                    return stage
            elif self.main_open_s is not None and t >= self.main_open_s - 1e-9:
                return stage
        return None

    def _events(self) -> list[float]:
        """The times the stepping must land on."""
        times = []
        if math.isfinite(self.stretch) and any(
            s.deploy_altitude_m is None and s.name not in self.opened
            for s in self.scheme.recovery.stages
        ):
            times.append(self.stretch)
        for value in (self.main_fire_s, self.main_open_s):
            if value is not None:
                times.append(value)
        return times

    def _schedule_main(self, command_s: float, t: float, y: np.ndarray) -> None:
        """The barometer reached the main altitude: set the charge and the opening."""
        scheme = self.scheme
        self.main_command_s = command_s
        if scheme.main_charge is None:
            self.main_open_s = command_s
        else:
            self.main_fire_s = command_s + scheme.main_delay_s
            if self.main_fire_s <= t + 1e-9:
                self._fire_main(t, y)

    def _fire_main(self, t: float, y: np.ndarray) -> None:
        scheme = self.scheme
        assert scheme.main_charge is not None
        state = replace(
            self.template, position=vector(y[:3], "m"), velocity=vector(y[3:6], "m/s")
        )
        self.main_separation = _eject(state, self.config, scheme.main_charge, self.mass)
        fired = float(self.main_fire_s if self.main_fire_s is not None else t)
        self.main_fire_s = None
        self.main_open_s = (
            fired + self.main_separation.line_stretch_s
            if self.main_separation.separated
            else None
        )
        if self.main_open_s is None:  # the charge failed: it never opens
            self.main_fire_s = None

    # ----- the flight ----------------------------------------------------------

    def fly(self) -> RecoveryOutcome:
        """Run the whole descent."""
        start_s, y, tumbled, separation, fire, _ = self._start()
        self._baro_setup(start_s)
        t = start_s
        self._record(t, y)
        landed = False
        steps = 0
        t, y, _ = self._open_due(t, y)
        while steps < _MAX_STEPS:
            steps += 1
            if self._all_open():
                t, y, landed = self._vertical(t, y)
                break
            limit = self._near_command(y, self._step_limit(y))
            for event in self._events():
                if event > t + 1e-9:
                    limit = min(limit, event - t)
            h = max(limit, 1e-6)
            new = self._rk4(y, h)
            if new[0] <= 0.0 < y[0]:
                h *= float(y[0] / (y[0] - new[0]))
                new = self._rk4(y, h)
                new[0] = 0.0
                t += h
                self._record(t, new)
                y = new
                landed = True
                break
            t_new, y_new, h = self._baro_step(t, y, new, h)
            t, y = t_new, y_new
            if self.main_fire_s is not None and t >= self.main_fire_s - 1e-9:
                self._fire_main(t, y)
            t, y, _ = self._open_due(t, y)
            if self.times[-1] < t - 1e-9:
                self._record(t, y)
            if y[0] <= 0.0:
                landed = True
                break
        descent = DescentResult(
            list(self.times),
            self.states,
            self.deployments,
            landed,
        )
        joined = tumble.join(tumbled, descent) if len(tumbled) > 0 else descent
        return RecoveryOutcome(
            version=FastRECO.name,
            descent=joined,
            apogee=self.detection,
            fire_s=fire,
            line_stretch_s=self.stretch,
            main_command_s=self.main_command_s,
            main_true_altitude_m=self.main_true_altitude_m,
            separated=separation.separated,
        )

    def _all_open(self) -> bool:
        return all(s.name in self.opened for s in self.scheme.recovery.stages) and bool(
            self.scheme.recovery.stages
        )

    def _baro_step(
        self, t: float, y: np.ndarray, new: np.ndarray, h: float
    ) -> tuple[float, np.ndarray, float]:
        """Advance the barometer over a step; find the main command inside it."""
        config = self.config
        p_old = air_at(config, float(y[0]))[0].pressure
        p_new = air_at(config, float(new[0]))[0].pressure
        filtered_new = self._filter(p_old, p_new, h, self.baro_pressure)
        if (
            self.main_command_s is None
            and self._has_altitude_stage()
            and t + h > self.detected_s
        ):
            command: float | None = None
            if self.baro_pressure >= self.command_pressure:  # already below the setting
                command = max(t, self.detected_s + 1e-6)
            elif filtered_new >= self.command_pressure:
                fraction = (self.command_pressure - self.baro_pressure) / max(
                    filtered_new - self.baro_pressure, 1e-12
                )
                command = max(
                    t + min(max(fraction, 0.0), 1.0) * h, self.detected_s + 1e-6
                )
            if command is not None:
                # the true height at the command, for the report
                part = self._rk4(y, max(command - t, 1e-9)) if command > t else y
                self.main_true_altitude_m = float(part[0])
                self._schedule_main(command, t, y)
                due = (
                    self.main_open_s
                    if self.main_open_s is not None
                    else self.main_fire_s
                )
                if due is not None and due <= t + h + 1e-9:
                    # something is due inside this step: stop the step there instead
                    stop = max(min(due, t + h) - t, 1e-6)
                    new = self._rk4(y, stop)
                    p_new = air_at(config, float(new[0]))[0].pressure
                    filtered_new = self._filter(p_old, p_new, stop, self.baro_pressure)
                    h = stop
        self.baro_pressure = filtered_new
        return t + h, new, h

    # ----- the vertical stage --------------------------------------------------

    def _vertical(  # pylint: disable=too-many-locals
        self, t: float, y: np.ndarray
    ) -> tuple[float, np.ndarray, bool]:
        """After the last canopy: fall at the local drag law, drift with the wind."""
        reco = self.reco
        y = y.copy()
        gust = self._gusts(t, y)
        while len(self.times) < _MAX_STEPS:
            height = float(y[0])
            air, _ = air_at(self.config, height)
            gravity = self._gravity(height)
            down = max(-float(y[3]), 0.0)
            terminal = math.sqrt(
                2.0 * self.mass * gravity / (air.air_density * self.cda)
            )
            settled = abs(down - terminal) < reco.settled_tolerance * terminal
            cap = reco.main_step_s if settled else reco.main_moving_step_s
            h = min(cap, reco.guard_m / max(down, 0.5))
            # density a little below the start of the step (the fall in the step)
            mid = max(height - 0.5 * down * h, 0.0)
            air_mid, wind_mid = air_at(self.config, mid)
            terminal = math.sqrt(
                2.0 * self.mass * gravity / (air_mid.air_density * self.cda)
            )
            drop = _fall(down, terminal, gravity, h)
            landed = drop[1] >= height
            if landed:
                h = _time_to_fall(down, terminal, gravity, height, h)
                drop = _fall(down, terminal, gravity, h)
                mid = max(height - 0.5 * drop[1], 0.0)
                air_mid, wind_mid = air_at(self.config, mid)
            wind_mid = wind_mid + gust(h)
            speed_through = math.hypot(
                0.5 * (down + drop[0]), *(y[4:6] - wind_mid[1:3])
            )
            tau = (
                2.0
                * self.mass
                / (air_mid.air_density * self.cda * max(speed_through, 1e-3))
            )
            decay = math.exp(-h / tau)
            slip = y[4:6] - wind_mid[1:3]
            sideways = wind_mid[1:3] * h + slip * tau * (1.0 - decay)
            y[1:3] += sideways
            y[4:6] = wind_mid[1:3] + slip * decay
            y[3] = -drop[0]
            y[0] = 0.0 if landed else height - drop[1]
            t += h
            self._record(t, y)
            if landed:
                return t, y, True
        return t, y, False

    def _gusts(self, t: float, y: np.ndarray):  # type: ignore[no-untyped-def]
        """A function giving the gust over the next step of ``h`` seconds.

        Ornstein-Uhlenbeck, as ``canopy_swing.Gusts``, stepped exactly: one
        normal draw per step, right for any step length.
        """
        reco = self.reco
        ground = float(np.linalg.norm(air_at(self.config, 0.0)[1][1:3]))
        sigma = reco.turbulence * max(ground, 1.0)
        value = np.zeros(3)
        if sigma <= 0.0 or reco.gust_time_s <= 0.0:
            return lambda h: value
        seed = zlib.crc32(np.round(np.append(y, t), 6).tobytes())
        rng = np.random.default_rng(seed)
        # start from a gust of the right size, as the wind is already gusting
        value[1:3] = rng.normal(0.0, sigma, 2)

        def advance(h: float) -> np.ndarray:
            decay = math.exp(-h / reco.gust_time_s)
            value[1:3] = value[1:3] * decay + rng.normal(
                0.0, sigma * math.sqrt(1.0 - decay * decay), 2
            )
            return value

        return advance


def _fall(
    down: float, terminal: float, gravity: float, h: float
) -> tuple[float, float]:
    """Fall speed and distance after ``h`` seconds, for a given terminal speed.

    Exact for a constant air density: ``du/dt = g (1 - u^2 / ut^2)``.
    """
    rate = gravity / terminal
    ratio = down / terminal
    if ratio < 1.0:
        c = math.atanh(min(ratio, 1.0 - 1e-9))
        u = terminal * math.tanh(rate * h + c)
        dist = (
            terminal
            / rate
            * (math.log(math.cosh(rate * h + c)) - math.log(math.cosh(c)))
        )
    else:
        c = math.atanh(1.0 / max(ratio, 1.0 + 1e-9))
        u = terminal / math.tanh(rate * h + c)
        dist = (
            terminal
            / rate
            * (math.log(math.sinh(rate * h + c)) - math.log(math.sinh(c)))
        )
    return u, dist


def _time_to_fall(
    down: float, terminal: float, gravity: float, height: float, h: float
) -> float:
    """The time within ``h`` at which the fall reaches exactly ``height``."""
    low, high = 0.0, h
    for _ in range(40):
        mid = 0.5 * (low + high)
        if _fall(down, terminal, gravity, mid)[1] < height:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
