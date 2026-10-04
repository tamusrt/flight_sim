"""The rocket tumbling after apogee, flown with the full 6-DOF model.

At apogee the rocket is slow, so the air barely holds it nose-first, and the
separation charge then kicks it: the body is pushed back from the nose and set
turning. Until the canopy's line comes tight (line stretch) nothing else holds
it, so it can turn over and tumble. This module flies that stretch with the
same 6-DOF integrator as the climb, with the aero table carried to 180 degrees
of angle of attack (``aero_extend``) and the pitch damping of
``PitchDamping``; from line stretch on the canopy holds the rocket and the
cheaper descent models take over from the state this one ends in.

Only a few seconds are flown this way (from apogee to line stretch, at most
``MAX_TUMBLE_S``), so it adds little run time. The rocket keeps the mass and
shape of the whole vehicle until line stretch; the nose section that has left
it is small next to the body, and is drawn flying out on its cord.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from flight_sim.descent import DescentResult
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import scalar, vector
from flight_sim.utilities.dcm import body_to_world
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState

# A fall with no canopy at apogee (drogueless) is flown with the 6-DOF model for
# this long at most, then as a point mass
MAX_TUMBLE_S = 15.0
_FIRST_STEP_S = 0.01


@dataclass(frozen=True)
class SeparationKick:
    """The turn the separation charge gives the body.

    The charge pushes the nose out and the body back along their axis; it is
    never perfectly straight, so the body also starts turning. That tip-off
    rate is not measured for our rockets, so it is an assumption, about the
    size reported for high-power rocket separations. It acts about a fixed
    body axis, so every run is the same.

    Attributes:
        tipoff_deg_s (float): Pitch or yaw rate the charge adds, degrees/s.
        axis_body (tuple[float, float, float]): Body axis it turns about.
    """

    tipoff_deg_s: float = 30.0
    axis_body: tuple[float, float, float] = (0.0, 1.0, 0.0)


KICK = SeparationKick()


def apply_kick(
    state: RocketState,
    *,
    separation_speed_m_s: float,
    nose_share: float,
    kick: SeparationKick = KICK,
) -> RocketState:
    """The state just after the charge fires.

    The nose and body fly apart at ``separation_speed_m_s``; by momentum the
    body (with the canopy still packed against it) moves back by the nose's
    share of that speed. It also starts turning at the tip-off rate.
    """
    nose_dir = body_to_world(state.orientation) @ np.array([1.0, 0.0, 0.0])
    velocity = state.velocity.m_as("m/s") - separation_speed_m_s * nose_share * nose_dir
    axis = np.array(kick.axis_body, dtype=float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    spin = state.angular_velocity.m_as("rad/s") + math.radians(kick.tipoff_deg_s) * axis
    return replace(
        state, velocity=vector(velocity, "m/s"), angular_velocity=vector(spin, "rad/s")
    )


def fly(  # pylint: disable=too-many-arguments
    start_s: float,
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
    *,
    until_s: float,
    fire_s: float | None = None,
    separation_speed_m_s: float = 0.0,
    nose_share: float = 0.0,
    kick: SeparationKick = KICK,
) -> list[tuple[float, RocketState]]:
    """Fly the 6-DOF model from ``start_s`` to ``until_s``, kicked at ``fire_s``.

    Returns:
        Every step's time and state, starting with the given one and ending
        exactly at ``until_s`` (or on the ground, if it comes first).
    """
    samples = [(start_s, state)]
    time, dt = start_s, _FIRST_STEP_S
    kicked = fire_s is None
    if fire_s is not None and fire_s <= start_s:
        state = apply_kick(
            state,
            separation_speed_m_s=separation_speed_m_s,
            nose_share=nose_share,
            kick=kick,
        )
        samples[0] = (start_s, state)
        kicked = True
    while time < until_s - 1e-9 and float(state.position.m_as("m")[0]) > 0.0:
        stop = until_s if kicked else min(until_s, float(fire_s or until_s))
        step = min(dt, stop - time)
        state, taken, dt_next, _ = adaptive_step(
            time, state, properties, config, scalar(step, "s"), events=()
        )
        time += float(taken.m_as("s"))
        dt = float(dt_next.m_as("s"))
        if not kicked and fire_s is not None and time >= fire_s - 1e-9:
            state = apply_kick(
                state,
                separation_speed_m_s=separation_speed_m_s,
                nose_share=nose_share,
                kick=kick,
            )
            kicked = True
        samples.append((time, state))
    return samples


def join(
    tumble: list[tuple[float, RocketState]], descent: DescentResult
) -> DescentResult:
    """The tumble followed by the descent flown from its last state.

    The descent's first sample is after the tumble's last, which it started
    from, so nothing is repeated.
    """
    times = [t for t, _ in tumble] + [
        t for t in descent.times_s if t > tumble[-1][0] + 1e-9
    ]
    states = [s for _, s in tumble] + [
        s
        for t, s in zip(descent.times_s, descent.states, strict=True)
        if t > tumble[-1][0] + 1e-9
    ]
    return DescentResult(times, states, descent.deployments, descent.landed)


def resample(
    tumble: list[tuple[float, RocketState]], interval_s: float = 0.05
) -> list[tuple[float, RocketState]]:
    """The tumble's steps thinned to about one per ``interval_s``, keeping the last."""
    out = [tumble[0]]
    for sample in tumble[1:-1]:
        if sample[0] - out[-1][0] >= interval_s - 1e-9:
            out.append(sample)
    if len(tumble) > 1:
        out.append(tumble[-1])
    return out
