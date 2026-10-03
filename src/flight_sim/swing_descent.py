"""The descent flown with the rocket and its canopy as two bodies on a line.

``descent.simulate_descent`` flies the rocket as one point mass. This module
flies the same descent, with the same atmosphere, drag stages and canopy
openings, but from the first release as two bodies joined by a line (see
``canopy_swing`` for the model). It builds on the integrator in ``descent``
and leaves that module as it is, so the point-mass descent is unchanged; it
is only used when the extended recovery model is asked for.

The state is the point-mass state, [position, velocity, distance through the
air] of the rocket, followed by the line direction and its rate. The first
seven entries keep their meaning, so the releases by height, the loads and
the landing work as they do for the point mass.
"""

# The loop mirrors descent.simulate_descent, which this module must not edit
# pylint: disable=duplicate-code

import math
from dataclasses import dataclass, field, replace

import numpy as np

from flight_sim.canopy_swing import CanopySwing, Gusts
from flight_sim.descent import (
    _STANDARD_GRAVITY,
    Deployment,
    DescentResult,
    RecoverySystem,
    ReefedParachute,
    _Descent,
    _nose_along,
    air_at,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.units import vector
from flight_sim.utilities.dcm import body_to_world
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_state import RocketState

_POINT_STATE = 7  # Position, velocity and distance through the air
_MIN_AIRSPEED_M_S = 0.5  # Below this the airflow has no usable direction
_MAX_STEP_S = 0.02
_OUTPUT_INTERVAL_S = 0.1
# Display only: the rocket turning onto the line at line stretch, a damped
# oscillation of about 2.5 s period (the cord's torque on its inertia, assumed)
_ATTITUDE_OMEGA_RAD_S = 2.5
_ATTITUDE_DAMPING = 0.3
_ATTITUDE_STEP_S = 0.01


@dataclass
class SwingTrace:
    """The swing under the canopy at each sample of a descent.

    Before the first release there is no line: the direction is zero and the
    canopy gives no drag.

    Attributes:
        line_directions (list[np.ndarray]): Unit vector along the line from
            the canopy to the rocket, world frame.
        canopy_velocities (list[np.ndarray]): Canopy velocity, world frame.
        angles_rad (list[float]): Angle between the line and the airflow
            past the canopy.
        drag_fractions (list[float]): Share of its drag the canopy gives.
    """

    line_directions: list[np.ndarray] = field(default_factory=list)
    canopy_velocities: list[np.ndarray] = field(default_factory=list)
    angles_rad: list[float] = field(default_factory=list)
    drag_fractions: list[float] = field(default_factory=list)


@dataclass
class SwingDescentResult:
    """A descent with the swing under the canopy.

    Attributes:
        descent (DescentResult): The rocket's states, as for the point mass.
        swing (SwingTrace): The line and the canopy at the same samples.
    """

    descent: DescentResult
    swing: SwingTrace


class _Attitude:
    """The rocket's nose direction, turning onto the line as a damped oscillator.

    Display only: it follows the line and never feeds back into the descent.
    """

    def __init__(self, nose: np.ndarray, start_s: float):
        self.nose = np.asarray(nose, dtype=float)
        self.rate = np.zeros(3)
        self.time_s = start_s

    def advance(self, time_s: float, target: np.ndarray) -> np.ndarray:
        """Turn toward the target direction up to a time; return the nose."""
        steps = max(math.ceil((time_s - self.time_s) / _ATTITUDE_STEP_S), 1)
        h = max(time_s - self.time_s, 0.0) / steps
        omega = _ATTITUDE_OMEGA_RAD_S
        for _ in range(steps):
            pull = omega**2 * (target - float(target @ self.nose) * self.nose)
            if float(np.linalg.norm(pull)) < 1e-9 and float(target @ self.nose) < 0:
                # Exactly opposite: turn about any axis across the nose
                pull = omega**2 * np.cross(self.nose, [0.0, 0.0, 1.0])
            accel = (
                pull
                - 2.0 * _ATTITUDE_DAMPING * omega * self.rate
                - float(self.rate @ self.rate) * self.nose
            )
            rate = self.rate + h * accel
            self.rate = rate - float(rate @ self.nose) * self.nose
            nose = self.nose + h * self.rate
            self.nose = nose / float(np.linalg.norm(nose))
        self.time_s = time_s
        result: np.ndarray = self.nose.copy()
        return result


@dataclass(kw_only=True)
class _SwingDescent(_Descent):
    """Integrates a descent whose canopy and rocket are two bodies."""

    swing: CanopySwing
    gusts: Gusts
    diameters: dict[str, float]
    gust: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: _Attitude | None = None

    def canopy_area(self, air_distance_m: float) -> float:
        """Drag coefficient times area of the open canopies, in m**2."""
        return sum(
            opening.open_fraction(air_distance_m) * opening.stage.drag_area_m2
            for opening in self.openings
        )

    def drag_area(self, air_distance_m: float) -> float:
        """Total drag coefficient times area at a distance through the air."""
        return self.recovery.body_drag_area_m2 + self.canopy_area(air_distance_m)

    def open_diameter(self, air_distance_m: float) -> float:
        """Diameter of the canopy as far as it is open, in metres.

        The area opens with the square of the progress, so the diameter
        opens in proportion to the progress.
        """
        return max(
            (
                self.diameters[opening.stage.name]
                * math.sqrt(opening.open_fraction(air_distance_m))
                for opening in self.openings
            ),
            default=0.0,
        )

    @staticmethod
    def swinging(y: np.ndarray) -> bool:
        """Whether the state is the two-body one (rocket, canopy and line)."""
        return y.size > _POINT_STATE

    def canopy_velocity(self, y: np.ndarray) -> np.ndarray:
        """Velocity of the canopy end of the line, world frame."""
        velocity: np.ndarray = y[3:6] - self.swing.line_length_m * y[10:13]
        return velocity

    def attack(self, y: np.ndarray, wind: np.ndarray) -> tuple[float, float]:
        """Angle of the line to the airflow past the canopy, and its drag share."""
        relative = self.canopy_velocity(y) - wind
        speed = float(np.linalg.norm(relative))
        if speed < _MIN_AIRSPEED_M_S:
            return 0.0, 1.0
        angle = math.acos(min(max(float(y[7:10] @ relative) / speed, -1.0), 1.0))
        return angle, self.swing.drag_fraction(angle)

    def canopy_drag(
        self, y: np.ndarray, wind: np.ndarray, density: float
    ) -> np.ndarray:
        """Drag on the canopy alone, in N."""
        relative = self.canopy_velocity(y) - wind
        area = self.attack(y, wind)[1] * self.canopy_area(float(y[6]))
        drag: np.ndarray = (
            -0.5 * density * float(np.linalg.norm(relative)) * area * relative
        )
        return drag

    def swing_derivative(self, y: np.ndarray) -> np.ndarray:
        """Rate of change of [rocket, speed, air distance, line, line rate]."""
        height = float(y[0])
        air, ground_wind = air_at(self.config, height)
        wind = ground_wind + self.gust
        pull = self.config.truth.gravity.magnitude(
            self.latitude_rad,
            float(self.config.truth.launch_elevation.m_as("m")) + height,
        )
        gravity = np.array([-pull, 0.0, 0.0])
        relative = y[3:6] - wind
        body_drag = (
            -0.5
            * air.air_density
            * float(np.linalg.norm(relative))
            * self.recovery.body_drag_area_m2
            * relative
        )
        # The canopy's mass is part of the mass the rocket flew with
        body_mass = self.mass_kg - self.swing.canopy_mass_kg
        inertia = self.swing.canopy_inertia_kg(
            air.air_density, self.open_diameter(float(y[6]))
        )
        body_accel, line_accel, _ = self.swing.accelerations(
            body_mass,
            inertia,
            line=y[7:10],
            line_rate=y[10:13],
            canopy_force=self.canopy_drag(y, wind, air.air_density)
            + self.swing.canopy_mass_kg * gravity,
            body_force=body_drag + body_mass * gravity,
        )
        canopy_speed = float(np.linalg.norm(self.canopy_velocity(y) - wind))
        return np.concatenate(
            (y[3:6], body_accel, [canopy_speed], y[10:13], line_accel)
        )

    def drag_acceleration(self, y: np.ndarray) -> np.ndarray:
        """Drag over mass in the world frame, in m/s**2.

        In the two-body state this is the drag on the canopy, which is what
        the harness carries.
        """
        if not self.swinging(y):
            return super().drag_acceleration(y)
        air, wind = air_at(self.config, float(y[0]))
        drag = self.canopy_drag(y, wind + self.gust, air.air_density)
        result: np.ndarray = drag / self.mass_kg
        return result

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """Rate of change of the state, point-mass or two-body."""
        if self.swinging(y):
            return self.swing_derivative(y)
        return super().derivative(y)

    def stable_step(self, y: np.ndarray, h: float) -> float:
        """Shorten h to a tenth of the time drag takes to change the speed.

        The light canopy reacts faster than the rocket, so its inertia sets
        the time once the canopy is open.
        """
        step = super().stable_step(y, h)
        if not self.swinging(y):
            return step
        air, wind = air_at(self.config, float(y[0]))
        speed = float(np.linalg.norm(y[3:6] - wind))
        distance = float(y[6]) + speed * h
        inertia = self.swing.canopy_inertia_kg(
            air.air_density, self.open_diameter(distance)
        )
        rate = air.air_density * speed * self.canopy_area(distance) / inertia
        return step if rate <= 0.0 else min(step, 0.1 / rate)

    def couple(self, y: np.ndarray) -> np.ndarray:
        """Switch to the two-body state at the first release.

        By line stretch the nose and bag, slowed by their drag, have been
        carried back into the rocket's wake, so the line comes taut along the
        rocket's motion through the air with the canopy trailing behind it.
        The canopy moves with the rocket at that moment.
        """
        if not self.openings or self.swinging(y):
            return y
        relative = y[3:6] - air_at(self.config, float(y[0]))[1]
        speed = float(np.linalg.norm(relative))
        down = np.array([-1.0, 0.0, 0.0])
        line = relative / speed if speed > _MIN_AIRSPEED_M_S else down
        return np.concatenate((y, line, np.zeros(3)))

    def tidy(self, y: np.ndarray) -> np.ndarray:
        """Keep the line a unit vector and its rate across it."""
        if self.swinging(y):
            y[7:10] /= float(np.linalg.norm(y[7:10]))
            y[10:13] -= float(y[10:13] @ y[7:10]) * y[7:10]
        return y

    def display_orientation(self, elapsed_s: float, y: np.ndarray) -> Quaternion:
        """Attitude shown for the sample: the rocket settling onto the line.

        At line stretch the cord pulls the rocket's forward end, so the
        rocket turns about its centre of gravity toward hanging nose-up along
        the line. It does so as a damped oscillation, overshooting and
        wobbling before it settles, rather than turning straight onto it.
        """
        if not self.swinging(y) or not self.openings:
            return super().display_orientation(elapsed_s, y)
        start = self.apogee_state.orientation
        if self.attitude is None:
            self.attitude = _Attitude(
                body_to_world(start)[:, 0], self.openings[0].time_s
            )
        return _nose_along(start, self.attitude.advance(elapsed_s, -y[7:10]))

    def trace_into(self, trace: SwingTrace, y: np.ndarray) -> None:
        """Add the swing at a sample; before the first release there is none."""
        if not self.swinging(y):
            trace.line_directions.append(np.zeros(3))
            trace.canopy_velocities.append(y[3:6].copy())
            trace.angles_rad.append(0.0)
            trace.drag_fractions.append(0.0)
            return
        _, wind = air_at(self.config, float(y[0]))
        angle, fraction = self.attack(y, wind + self.gust)
        trace.line_directions.append(y[7:10].copy())
        trace.canopy_velocities.append(self.canopy_velocity(y))
        trace.angles_rad.append(angle)
        trace.drag_fractions.append(fraction)

    def fly(self, apogee_time_s: float, max_time_s: float) -> SwingDescentResult:
        """Fall from apogee to the ground, or until the time limit."""
        state = self.apogee_state
        y = np.concatenate(
            (state.position.m_as("m"), state.velocity.m_as("m/s"), [0.0])
        )
        elapsed, next_output, landed = 0.0, _OUTPUT_INTERVAL_S, False
        times: list[float] = []
        states: list[RocketState] = []
        trace = SwingTrace()

        def sample(at_s: float, values: np.ndarray) -> None:
            times.append(apogee_time_s + at_s)
            self.trace_into(trace, values)
            states.append(
                replace(
                    state,
                    position=vector(values[:3], "m"),
                    velocity=vector(values[3:6], "m/s"),
                    angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
                    orientation=self.display_orientation(at_s, values),
                )
            )

        while elapsed < max_time_s:
            self.release_due(elapsed, y)
            y = self.couple(y)
            limit = min(_MAX_STEP_S, next_output - elapsed, max_time_s - elapsed)
            h, new = self.advance(y, self.step_length(elapsed, y, limit))
            elapsed += h
            y = self.tidy(new)
            self.gust = self.gusts.advance(h)
            self.after_step(elapsed, y)
            if y[0] <= 1e-6:
                y[0] = 0.0
                sample(elapsed, y)
                landed = True
                break
            if elapsed >= next_output - 1e-9:
                sample(elapsed, y)
                next_output += _OUTPUT_INTERVAL_S
        if not landed and (not times or times[-1] < apogee_time_s + elapsed):
            sample(elapsed, y)
        deployments = [
            Deployment(
                name=opening.stage.name,
                time_s=apogee_time_s + opening.time_s,
                altitude_m=opening.altitude_m,
                airspeed_m_s=opening.airspeed_m_s,
                inflation_time_s=(
                    None
                    if opening.inflated_s is None
                    else opening.inflated_s - opening.time_s
                ),
                peak_load_g=opening.peak_load_g,
                peak_force_n=opening.peak_load_g * _STANDARD_GRAVITY * self.mass_kg,
            )
            for opening in self.openings
        ]
        return SwingDescentResult(
            DescentResult(times, states, deployments, landed), trace
        )


def simulate_swing_descent(
    apogee_time_s: float,
    apogee_state: RocketState,
    config: IntegrationConfiguration,
    recovery: RecoverySystem,
    *,
    mass_kg: float,
    swing: CanopySwing,
    max_time_s: float = 3600.0,
) -> SwingDescentResult:
    """Fall from apogee to the ground with the rocket swinging under its canopy.

    The same as ``simulate_descent`` until the first canopy is released; from
    then the rocket and the canopy are two bodies on a line of fixed length.
    The returned states are the rocket's, with its attitude hanging nose-up
    along the line.

    Args:
        apogee_time_s (float): Time since ignition at apogee, in seconds.
        apogee_state (RocketState): State at apogee.
        config (IntegrationConfiguration): Atmosphere, gravity and latitude.
        recovery (RecoverySystem): Canopies and body drag.
        mass_kg (float): Mass of the rocket, which stays the same, in kg.
        swing (CanopySwing): The two-body model.
        max_time_s (float): Longest descent to simulate, in seconds.

    Returns:
        SwingDescentResult: Samples after apogee through the impact, and the
            line and canopy at the same samples.
    """
    diameters: dict[str, float] = {}
    for canopy in recovery.parachutes:
        opened = (
            (canopy.reefed_opening_diameter_m, canopy.diameter_m)
            if isinstance(canopy, ReefedParachute)
            else (canopy.diameter_m,)
        )
        for stage, diameter in zip(canopy.stages(), opened, strict=True):
            diameters[stage.name] = diameter
    ground_wind = float(np.linalg.norm(air_at(config, 0.0)[1]))
    descent = _SwingDescent(
        apogee_state=apogee_state,
        config=config,
        recovery=recovery,
        mass_kg=mass_kg,
        latitude_rad=float(config.truth.launch_latitude.m_as("rad")),
        swing=swing,
        gusts=Gusts(swing, ground_wind),
        diameters=diameters,
    )
    return descent.fly(apogee_time_s, max_time_s)
