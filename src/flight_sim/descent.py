"""Point-mass (3-DOF) descent from apogee under gravity, drag and parachutes.

After apogee the rocket is modelled as a point mass: the 6-DOF aerodynamic
table only covers small angles of attack, and under a canopy the drag of the
canopy dominates. Each step integrates

    m * dv/dt = m * g + D,    D = -0.5 * rho * |v_air| * v_air * (Cd * A)

where ``v_air`` is the velocity relative to the wind and ``Cd * A`` is the
body's drag area plus that of every canopy released so far. Air density,
wind and gravity come from the same models as the ascent, so the descent
slows as the air thickens and drifts with the wind.

A canopy is released either a set time after apogee or when the rocket
falls through a set height above the pad. It then inflates while the rocket
travels ``fill_constant * diameter`` through the air (Knacke's
filling-distance rule), with its drag area growing with the square of the
share of that distance covered; it then overshoots its steady drag area by
10 percent and settles within half a fill distance (overinflation). A
canopy released at apogee, where the airspeed is low, therefore takes
longer to open than one released fast. The deceleration during the
opening is reported as the opening load.

The attitude is not simulated. For display, the rocket keeps its apogee
attitude until the first canopy is released, then turns over two seconds to
hang nose-up along the airflow under it. The angular velocity is reported
as zero.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import r3f

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.integration import IntegrationConfiguration
from flight_sim.units import vector
from flight_sim.utilities.dcm import world_to_body
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.rocket_state import RocketState

_STANDARD_GRAVITY = 9.80665  # m/s**2, for loads in g
_TURN_TO_HANG_S = 2.0  # Display only: time to swing under the first canopy
LOAD_WINDOW_S = 1.0  # Time after full inflation still counted as opening
_OPENING_STEP_S = 0.002  # Longest step while a canopy is opening
# Overinflation: right after it fills, a canopy overshoots its steady drag area
# by about 10 percent and settles within half a fill distance (Knacke,
# Parachute Recovery Systems Design Manual, ch. 5; an assumed typical value)
_OVERINFLATION = 0.10
_OVERSHOOT_SPAN = 0.5


@dataclass(frozen=True)
class DragStage:
    """One step up in drag: a canopy opening, or a reefed canopy disreefing.

    Attributes:
        name (str): Name used in the printed results.
        drag_area_m2 (float): Drag coefficient times area it adds once fully
            open, in m**2.
        fill_distance_m (float): Distance through the air over which it
            opens, in metres.
        deploy_delay_s (float): Seconds after apogee at which it starts.
            Ignored when ``deploy_altitude_m`` is set.
        deploy_altitude_m (float | None): Height above the pad at which it
            starts on the way down, or None to start on the delay.
        after (str | None): Name of a stage that must have started first.
    """

    name: str
    drag_area_m2: float
    fill_distance_m: float
    deploy_delay_s: float = 0.0
    deploy_altitude_m: float | None = None
    after: str | None = None


@dataclass(frozen=True)
class Parachute:
    """One canopy and when it is released.

    Attributes:
        name (str): Name used in the printed results.
        diameter_m (float): Nominal canopy diameter in metres.
        drag_coefficient (float): Drag coefficient on the canopy area, the
            nominal disc less any spill hole.
        deploy_delay_s (float): Seconds after apogee at which it is released.
            Ignored when ``deploy_altitude_m`` is set.
        deploy_altitude_m (float | None): Height above the pad at which it is
            released on the way down, or None to release on the delay.
        fill_constant (float): Canopy diameters of travel it takes to
            inflate; about 8 for solid flat or conical textile canopies.
        spill_hole_diameter_m (float): Diameter of the vent at the apex.
    """

    name: str
    diameter_m: float
    drag_coefficient: float
    deploy_delay_s: float = 0.0
    deploy_altitude_m: float | None = None
    fill_constant: float = 8.0
    spill_hole_diameter_m: float = 0.0

    @property
    def drag_area_m2(self) -> float:
        """Drag coefficient times canopy area when fully open, in m**2."""
        return self.drag_coefficient * _canopy_area(
            self.diameter_m, self.spill_hole_diameter_m
        )

    @property
    def fill_distance_m(self) -> float:
        """Distance through the air over which it inflates, in metres."""
        return self.fill_constant * self.diameter_m

    def stages(self) -> tuple[DragStage, ...]:
        """The single opening of this canopy."""
        return (
            DragStage(
                self.name,
                self.drag_area_m2,
                self.fill_distance_m,
                self.deploy_delay_s,
                self.deploy_altitude_m,
            ),
        )


@dataclass(frozen=True)
class ReefedParachute:
    """One canopy flown reefed as a drogue, then disreefed as the main.

    This is single separation, dual deploy with a reefed main: the canopy
    comes out reefed on the delay after apogee, and a line cutter frees the
    reefing line at ``disreef_altitude_m`` above the pad.

    The reefed drag area follows the recovery team's estimate: the canopy
    area plus the reefed mouth area is taken as the surface of a sphere,
    whose cross-section less the mouth area is the reefed area. The team
    notes that the reefed drag coefficient needs testing.

    Attributes:
        name (str): Name used in the printed results.
        diameter_m (float): Nominal canopy diameter in metres.
        drag_coefficient (float): Fully open drag coefficient on the canopy
            area, the nominal disc less the spill hole.
        reefed_opening_diameter_m (float): Diameter of the mouth the reefing
            line holds, in metres.
        reefed_drag_coefficient (float): Drag coefficient while reefed.
        disreef_altitude_m (float): Height above the pad of the reef cut.
        deploy_delay_s (float): Seconds after apogee at which it comes out.
        fill_constant (float): Diameters of travel to open, applied to the
            reefed mouth for the first opening and to the full canopy for
            the disreef.
        spill_hole_diameter_m (float): Diameter of the vent at the apex.
    """

    name: str
    diameter_m: float
    drag_coefficient: float
    reefed_opening_diameter_m: float
    reefed_drag_coefficient: float
    disreef_altitude_m: float
    deploy_delay_s: float = 0.0
    fill_constant: float = 8.0
    spill_hole_diameter_m: float = 0.0

    @property
    def drag_area_m2(self) -> float:
        """Drag coefficient times canopy area when fully open, in m**2."""
        return self.drag_coefficient * _canopy_area(
            self.diameter_m, self.spill_hole_diameter_m
        )

    @property
    def reefed_drag_area_m2(self) -> float:
        """Drag coefficient times area while reefed, in m**2."""
        mouth = math.pi * self.reefed_opening_diameter_m**2 / 4.0
        flat = math.pi * self.diameter_m**2 / 4.0
        # Sphere of surface flat + mouth has cross-section (flat + mouth) / 4
        return self.reefed_drag_coefficient * ((flat + mouth) / 4.0 - mouth)

    def stages(self) -> tuple[DragStage, ...]:
        """The reefed opening, then the disreef to the full canopy."""
        reefed = DragStage(
            f"{self.name} reefed",
            self.reefed_drag_area_m2,
            self.fill_constant * self.reefed_opening_diameter_m,
            self.deploy_delay_s,
        )
        disreef = DragStage(
            f"{self.name} reef cut",
            self.drag_area_m2 - self.reefed_drag_area_m2,
            self.fill_constant * self.diameter_m,
            deploy_altitude_m=self.disreef_altitude_m,
            after=reefed.name,
        )
        return reefed, disreef


@dataclass(frozen=True)
class RecoverySystem:
    """The canopies and the drag of the rocket body falling with them.

    Attributes:
        parachutes (tuple[Parachute | ReefedParachute, ...]): Canopies.
        body_drag_area_m2 (float): Drag coefficient times area of the body,
            acting through the whole descent, in m**2.
    """

    parachutes: tuple[Parachute | ReefedParachute, ...]
    body_drag_area_m2: float = 0.0

    @property
    def stages(self) -> tuple[DragStage, ...]:
        """Every drag stage of every canopy, in order."""
        return tuple(stage for p in self.parachutes for stage in p.stages())


def _canopy_area(diameter_m: float, spill_hole_diameter_m: float) -> float:
    """Return the canopy area in m**2: the nominal disc less the spill hole."""
    return math.pi * (diameter_m**2 - spill_hole_diameter_m**2) / 4.0


@dataclass(frozen=True)
class Deployment:
    """What happened when one canopy was released.

    Attributes:
        name (str): Canopy name.
        time_s (float): Time since ignition of the release, in seconds.
        altitude_m (float): Height above the pad at release, in metres.
        airspeed_m_s (float): Speed relative to the air at release, in m/s.
        inflation_time_s (float | None): Time it took to inflate fully, in
            seconds, or None if it was still inflating at the end.
        peak_load_g (float): Largest drag deceleration, in g, from release
            until a second after full inflation.
        peak_force_n (float): Largest total drag force over the same window,
            in N; the load the harness and shock cord carry.
    """

    name: str
    time_s: float
    altitude_m: float
    airspeed_m_s: float
    inflation_time_s: float | None
    peak_load_g: float
    peak_force_n: float


@dataclass(frozen=True)
class DescentResult:
    """Outcome of a descent.

    Attributes:
        times_s (list[float]): Time since ignition of each sample, in seconds.
        states (list[RocketState]): State at each sample.
        deployments (list[Deployment]): Canopy releases, in time order.
        landed (bool): Whether the last sample is on the ground; False if the
            time limit ran out first.
    """

    times_s: list[float]
    states: list[RocketState]
    deployments: list[Deployment]
    landed: bool


def inflation_fractions(progress: np.ndarray) -> np.ndarray:
    """Share of full drag area at each progress through the fill distance.

    The same law as ``_Opening.open_fraction`` (squared growth over the fill
    distance, then the overinflation bulge), for many points at once. A
    progress of 1 is the end of the fill distance.
    """
    p = np.asarray(progress, dtype=float)
    grow = np.maximum(p, 0.0) ** 2
    past = np.clip((p - 1.0) / _OVERSHOOT_SPAN, 0.0, 1.0)
    bulge = 1.0 + _OVERINFLATION * np.sin(np.pi * past)
    settled = np.where(past >= 1.0, 1.0, bulge)
    result: np.ndarray = np.where(p <= 1.0, grow, settled)
    return result


# How far past the fill distance a canopy is still settling (in fill distances)
SETTLING_SPAN = 1.0 + _OVERSHOOT_SPAN


@dataclass
class _Opening:
    """A released canopy while the descent is integrated."""

    stage: DragStage
    time_s: float  # Since apogee
    air_distance_m: float  # Distance through the air at release
    altitude_m: float
    airspeed_m_s: float
    inflated_s: float | None = None  # Since apogee
    peak_load_g: float = 0.0

    def open_fraction(self, air_distance_m: float) -> float:
        """Share of the full drag area it gives at a distance through the air."""
        fill = self.stage.fill_distance_m
        if fill <= 0.0:
            return 1.0
        progress = (air_distance_m - self.air_distance_m) / fill
        if progress <= 1.0:
            return max(progress, 0.0) ** 2
        # just after it fills the canopy bulges past its steady size and settles
        past = (progress - 1.0) / _OVERSHOOT_SPAN
        if past >= 1.0:
            return 1.0
        return 1.0 + _OVERINFLATION * math.sin(math.pi * past)

    def loading(self, elapsed_s: float) -> bool:
        """Whether a time since apogee is in its opening-load window."""
        if elapsed_s < self.time_s:
            return False
        return self.inflated_s is None or elapsed_s <= self.inflated_s + LOAD_WINDOW_S


def launch_elevation_m(truth: Any) -> float:
    """The pad's height above sea level in metres, converted once per quantity.

    The conversion through the unit library is slow next to the rest of a
    descent step, and a descent asks for it many thousands of times, so the
    plain number is kept beside the quantity it came from.
    """
    quantity = truth.launch_elevation
    cached = truth.__dict__.get("_elevation_cache")
    if cached is None or cached[0] is not quantity:
        cached = (quantity, float(quantity.m_as("m")))
        truth.__dict__["_elevation_cache"] = cached
    return float(cached[1])


def air_at(
    config: IntegrationConfiguration, height_m: float
) -> tuple[AtmosphereConditions, np.ndarray]:
    """Air conditions and wind at a height above the pad, in m."""
    truth = config.truth
    altitude = launch_elevation_m(truth) + height_m
    return truth.atmosphere.conditions(altitude), truth.wind.velocity(altitude)


@dataclass
class _Descent:
    """Integrates one descent; ``simulate_descent`` is the public entry."""

    apogee_state: RocketState
    config: IntegrationConfiguration
    recovery: RecoverySystem
    mass_kg: float
    latitude_rad: float
    openings: list[_Opening] = field(default_factory=list)

    def air(self, height_m: float) -> tuple[AtmosphereConditions, np.ndarray]:
        """Air conditions and wind at a height above the pad, in m."""
        return air_at(self.config, height_m)

    def drag_area(self, air_distance_m: float) -> float:
        """Total drag coefficient times area at a distance through the air."""
        return self.recovery.body_drag_area_m2 + sum(
            opening.open_fraction(air_distance_m) * opening.stage.drag_area_m2
            for opening in self.openings
        )

    def drag_acceleration(self, y: np.ndarray) -> np.ndarray:
        """Drag force over mass in the world frame, in m/s**2."""
        air, wind = self.air(float(y[0]))
        relative = y[3:6] - wind
        speed = float(np.linalg.norm(relative))
        scale = 0.5 * air.air_density * speed * self.drag_area(float(y[6]))
        result: np.ndarray = -scale * relative / self.mass_kg
        return result

    def derivative(self, y: np.ndarray) -> np.ndarray:
        """Rate of change of [position, velocity, distance through the air]."""
        acceleration = self.drag_acceleration(y)
        truth = self.config.truth
        acceleration[0] -= truth.gravity.magnitude(
            self.latitude_rad, launch_elevation_m(truth) + float(y[0])
        )
        _, wind = self.air(float(y[0]))
        airspeed = float(np.linalg.norm(y[3:6] - wind))
        return np.concatenate((y[3:6], acceleration, [airspeed]))

    def rk4(self, y: np.ndarray, h: float) -> np.ndarray:
        """One classical Runge-Kutta step of length h seconds."""
        k1 = self.derivative(y)
        k2 = self.derivative(y + h / 2 * k1)
        k3 = self.derivative(y + h / 2 * k2)
        k4 = self.derivative(y + h * k3)
        result: np.ndarray = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return result

    def stable_step(self, y: np.ndarray, h: float) -> float:
        """Shorten h to a tenth of the time drag takes to change the speed.

        That time is ``m / (rho * |v_air| * Cd * A)``; RK4 is accurate well
        inside it, which keeps the opening of a large canopy resolved.
        """
        air, wind = self.air(float(y[0]))
        speed = float(np.linalg.norm(y[3:6] - wind))
        area = self.drag_area(float(y[6]) + speed * h)
        rate = air.air_density * speed * area / self.mass_kg
        return h if rate <= 0.0 else min(h, 0.1 / rate)

    def release(self, stage: DragStage, elapsed_s: float, y: np.ndarray) -> None:
        """Start a canopy opening at a time since apogee and a state."""
        _, wind = self.air(float(y[0]))
        self.openings.append(
            _Opening(
                stage=stage,
                time_s=elapsed_s,
                air_distance_m=float(y[6]),
                altitude_m=float(y[0]),
                airspeed_m_s=float(np.linalg.norm(y[3:6] - wind)),
            )
        )
        self.after_step(elapsed_s, y)

    def after_step(self, elapsed_s: float, y: np.ndarray) -> None:
        """Note full inflations and track each opening's peak load, in g."""
        load = float(np.linalg.norm(self.drag_acceleration(y))) / _STANDARD_GRAVITY
        _, wind = self.air(float(y[0]))
        airspeed = float(np.linalg.norm(y[3:6] - wind))
        for opening in self.openings:
            if opening.inflated_s is None and opening.open_fraction(y[6]) >= 1.0:
                # Back off the distance travelled past full inflation
                past = y[6] - opening.air_distance_m - opening.stage.fill_distance_m
                opening.inflated_s = elapsed_s - past / max(airspeed, 1e-9)
            if opening.loading(elapsed_s):
                opening.peak_load_g = max(opening.peak_load_g, load)

    def pending(self) -> list[DragStage]:
        """Drag stages not yet started."""
        started = {opening.stage.name for opening in self.openings}
        return [s for s in self.recovery.stages if s.name not in started]

    def release_due(self, elapsed_s: float, y: np.ndarray) -> None:
        """Release every canopy whose delay has passed or height is reached."""
        for stage in self.pending():
            started = {opening.stage.name for opening in self.openings}
            if stage.after is not None and stage.after not in started:
                continue
            if stage.deploy_altitude_m is None:
                due = elapsed_s >= stage.deploy_delay_s - 1e-9
            else:
                due = float(y[0]) <= stage.deploy_altitude_m
            if due:
                self.release(stage, elapsed_s, y)

    def step_length(self, elapsed_s: float, y: np.ndarray, limit_s: float) -> float:
        """Choose a stable step that lands on the next timed release.

        While a canopy is opening the step is held short, so the peak load
        between steps is not missed.
        """
        h = limit_s
        for stage in self.pending():
            if stage.deploy_altitude_m is None:
                h = min(h, stage.deploy_delay_s - elapsed_s)
        if any(opening.loading(elapsed_s) for opening in self.openings):
            h = min(h, _OPENING_STEP_S)
        return max(self.stable_step(y, h), 1e-6)

    def advance(self, y: np.ndarray, h: float) -> tuple[float, np.ndarray]:
        """Take a step, shortened to end on the ground or a release height.

        Returns:
            tuple[float, np.ndarray]: The step length taken and the new state.
        """
        new = self.rk4(y, h)
        heights = [
            p.deploy_altitude_m
            for p in self.pending()
            if p.deploy_altitude_m is not None
        ]
        floor = max([0.0, *heights])
        if new[0] < floor < y[0]:
            h *= (y[0] - floor) / (y[0] - new[0])
            new = self.rk4(y, h)
            if floor > 0.0:
                new[0] = max(new[0], floor)
        return h, new

    def display_orientation(self, elapsed_s: float, y: np.ndarray) -> Quaternion:
        """Attitude shown for the sample; see the module docstring."""
        start = self.apogee_state.orientation
        if not self.openings:
            return start
        _, wind = self.air(float(y[0]))
        airflow = wind - y[3:6]  # Air moving past the rocket
        speed = float(np.linalg.norm(airflow))
        nose = airflow / speed if speed > 1e-6 else np.array([1.0, 0.0, 0.0])
        progress = (elapsed_s - self.openings[0].time_s) / _TURN_TO_HANG_S
        return _slerp(start, _nose_along(start, nose), min(max(progress, 0.0), 1.0))


def simulate_descent(  # pylint: disable=too-many-arguments
    apogee_time_s: float,
    apogee_state: RocketState,
    config: IntegrationConfiguration,
    recovery: RecoverySystem,
    *,
    mass_kg: float,
    max_step_s: float = 0.02,
    output_interval_s: float = 0.1,
    max_time_s: float = 3600.0,
) -> DescentResult:
    """Fall from apogee to the ground as a point mass.

    Steps are RK4, at most ``max_step_s`` long and shorter while drag is
    changing the speed quickly. Steps land exactly on every output time and
    every timed release; a release by height and the impact are found by
    shortening the step that crosses them.

    Args:
        apogee_time_s (float): Time since ignition at apogee, in seconds.
        apogee_state (RocketState): State at apogee.
        config (IntegrationConfiguration): Atmosphere, gravity and latitude.
        recovery (RecoverySystem): Canopies and body drag.
        mass_kg (float): Mass of the rocket, which stays the same, in kg.
        max_step_s (float): Longest integration step in seconds.
        output_interval_s (float): Spacing of the returned samples, seconds.
        max_time_s (float): Longest descent to simulate, in seconds.

    Returns:
        DescentResult: Samples after apogee through the impact.
    """
    descent = _Descent(
        apogee_state=apogee_state,
        config=config,
        recovery=recovery,
        mass_kg=mass_kg,
        latitude_rad=float(config.truth.launch_latitude.m_as("rad")),
    )
    # Position and velocity in the world frame, then distance through the air
    y = np.concatenate(
        (apogee_state.position.m_as("m"), apogee_state.velocity.m_as("m/s"), [0.0])
    )
    elapsed = 0.0
    next_output = output_interval_s
    times: list[float] = []
    states: list[RocketState] = []
    landed = False

    def sample(at_s: float, values: np.ndarray) -> None:
        times.append(apogee_time_s + at_s)
        states.append(
            replace(
                apogee_state,
                position=vector(values[:3], "m"),
                velocity=vector(values[3:6], "m/s"),
                angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
                orientation=descent.display_orientation(at_s, values),
            )
        )

    while elapsed < max_time_s:
        descent.release_due(elapsed, y)
        limit = min(max_step_s, next_output - elapsed, max_time_s - elapsed)
        h, new = descent.advance(y, descent.step_length(elapsed, y, limit))
        elapsed += h
        y = new
        descent.after_step(elapsed, y)

        if y[0] <= 1e-6:
            y[0] = 0.0
            sample(elapsed, y)
            landed = True
            break
        if elapsed >= next_output - 1e-9:
            sample(elapsed, y)
            next_output += output_interval_s

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
            peak_force_n=opening.peak_load_g * _STANDARD_GRAVITY * descent.mass_kg,
        )
        for opening in descent.openings
    ]
    return DescentResult(times, states, deployments, landed)


def _nose_along(start: Quaternion, nose: np.ndarray) -> Quaternion:
    """Return the attitude with the nose along a world direction.

    The body Y axis is kept as close as possible to its direction at
    ``start`` so the rocket does not appear to roll.
    """
    start_y = world_to_body(start)[1]
    side = start_y - float(start_y @ nose) * nose
    if float(np.linalg.norm(side)) < 1e-6:
        helper = np.array([0.0, 1.0, 0.0]) if abs(nose[1]) < 0.9 else np.eye(3)[2]
        side = helper - float(helper @ nose) * nose
    side /= float(np.linalg.norm(side))
    rows = np.vstack((nose, side, np.cross(nose, side)))
    q_w, q_x, q_y, q_z = (float(c) for c in r3f.dcm_to_quat(rows))
    return Quaternion(q_w=q_w, q_x=q_x, q_y=q_y, q_z=q_z)


def _slerp(start: Quaternion, end: Quaternion, fraction: float) -> Quaternion:
    """Spherical interpolation between two attitudes, by the shorter arc."""
    a = np.array([start.q_w, start.q_x, start.q_y, start.q_z])
    b = np.array([end.q_w, end.q_x, end.q_y, end.q_z])
    a /= float(np.linalg.norm(a))
    b /= float(np.linalg.norm(b))
    dot = float(a @ b)
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        mixed = a + fraction * (b - a)
    else:
        angle = math.acos(dot)
        mixed = (
            math.sin((1.0 - fraction) * angle) * a + math.sin(fraction * angle) * b
        ) / math.sin(angle)
    mixed /= float(np.linalg.norm(mixed))
    return Quaternion(
        q_w=float(mixed[0]),
        q_x=float(mixed[1]),
        q_y=float(mixed[2]),
        q_z=float(mixed[3]),
    )
