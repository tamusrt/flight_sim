"""The extended recovery model that ``visual_run`` flies after apogee.

``python -m flight_sim`` flies the descent as a point mass under the canopy
set in ``__main__`` and does not use this module. Around that same recovery
system, and without changing it, this adds:

* the flight computer's apogee vote and the barometric main command
  (``flight_computer``),
* the nose ejection and the shock cord running out (``recovery_motion``),
* the rocket swinging under its canopy as a two-body system, with the
  canopy's drag falling away as it tips away from the airflow
  (``canopy_swing`` and ``swing_descent``).

The settings below are the recovery team's, or assumptions noted at each one.
"""

from collections.abc import Iterator

import numpy as np

from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import RecoverySystem
from flight_sim.flight_computer import (
    FlightComputer,
    RecoveryPlan,
    Sample,
    plan_recovery,
)
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import (
    CORD_TO_BODY_M,
    CORD_TO_NOSE_M,
    EjectionCharge,
    NoseSwing,
)
from flight_sim.visualize import RecoveryFrame, TelemetryLog

_FT_TO_M = 0.3048

# Apogee is declared by the flight computer, the charge fires 3 s later (as
# the standard launch assumes for the canopy), the reef is cut at 2000 ft
FLIGHT_COMPUTER = FlightComputer(apogee_delay_s=3.0, main_altitude_m=2000.0 * _FT_TO_M)

# The nose section's mass, from SOL_4_30.ork
EJECTION_CHARGE = EjectionCharge(nose_mass_kg=4.004)

# Suspension lines from the canopy's skirt to the cord, as a share of its
# diameter (assumed)
_LINE_SHARE = 0.9

# The nose section's centre of mass from its harness at the shoulder, about
# half the section (assumed)
NOSE_HARNESS_TO_CG_M = 0.45

# Nose tip to the separation joint, where both harnesses are (SOL_4_30.ork)
_SEPARATION_STATION_M = 0.9144


def lines_m(recovery: RecoverySystem) -> float:
    """Length of the suspension lines of the biggest canopy, in metres."""
    return _LINE_SHARE * max((p.diameter_m for p in recovery.parachutes), default=0.0)


def canopy_swing_for(
    recovery: RecoverySystem, centre_of_gravity_m: float
) -> CanopySwing:
    """The two-body model for a recovery system and a rocket.

    The line runs from the canopy's skirt through the suspension lines and
    the body's leg of the cord to the harness at the separation joint, and
    on to the rocket's centre of gravity.

    Args:
        recovery (RecoverySystem): Its first canopy sets the line length.
        centre_of_gravity_m (float): Rocket centre of gravity, metres aft of
            the nose tip, at apogee.

    Returns:
        CanopySwing: The model, with its other settings at their defaults.
    """
    length = (
        CORD_TO_BODY_M + lines_m(recovery) + centre_of_gravity_m - _SEPARATION_STATION_M
    )
    return CanopySwing(line_length_m=length)


def plan_full_recovery(
    flight: list[Sample], config: IntegrationConfiguration, recovery: RecoverySystem
) -> RecoveryPlan:
    """Fly the descent with the flight computer, the ejection and the swing.

    Args:
        flight (list[Sample]): The ascent, ending at the true apogee.
        config (IntegrationConfiguration): Atmosphere and gravity.
        recovery (RecoverySystem): The canopies, with nominal settings.

    Returns:
        RecoveryPlan: The timeline and the descent flown with it.
    """
    apogee = flight[-1][1]
    swing = canopy_swing_for(recovery, -float(apogee.cg_location.m_as("m")[0]))
    return plan_recovery(
        flight, config, recovery, FLIGHT_COMPUTER, EJECTION_CHARGE, swing=swing
    )


def log_events(log: TelemetryLog, plan: RecoveryPlan) -> None:
    """Add the flight computer and recovery events to the timeline."""
    votes = plan.apogee
    for kind, name, time in (
        ("vote", "Velocity vote", votes.velocity_vote_s),
        ("vote", "Pressure vote", votes.pressure_vote_s),
        ("vote", "Attitude vote", votes.attitude_vote_s),
        ("fc", "Apogee detected", votes.detected_s),
    ):
        if time is not None:
            log.add_event(kind, name, time)
    log.add_event(
        "charge", "Separation charge", plan.fire_s, speed=plan.separation.speed_m_s
    )
    log.add_event(
        "stretch",
        "Line stretch",
        plan.line_stretch_s,
        snatch=plan.separation.snatch_force_n,
    )
    for stage, deployment in enumerate(plan.descent.deployments):
        log.add_event(
            "deploy" if stage == 0 else "disreef",
            deployment.name.capitalize(),
            deployment.time_s,
            inflation=deployment.inflation_time_s or 0.0,
            load=deployment.peak_force_n,
            altitude=deployment.altitude_m,
        )


def frames(plan: RecoveryPlan) -> Iterator[tuple[float, RecoveryFrame]]:
    """The recovery hardware at each sample of the descent.

    Yields the sample time and what the scene draws besides the rocket: the
    nose coming out of the body, then the line to the canopy, the canopy's
    angle of attack and drag share, and the nose swinging on its leg.
    """
    descent, swing = plan.descent, plan.swing
    if swing is None:
        raise ValueError("The plan was not flown with the swing model")
    times = descent.times_s
    stretch = int(np.searchsorted(times, plan.line_stretch_s))
    lines = np.array(swing.line_directions)
    nose = NoseSwing(
        length_m=CORD_TO_NOSE_M + lines_m(plan.recovery) + NOSE_HARNESS_TO_CG_M
    )
    nose_dirs = nose.swing(
        times,
        np.array(swing.canopy_velocities),
        start_s=plan.line_stretch_s,
        start=lines[min(stretch, len(times) - 1)],
    )
    curve = plan.separation
    for i, time in enumerate(times):
        if time >= plan.line_stretch_s and float(np.linalg.norm(lines[i])) > 0.0:
            yield (
                time,
                RecoveryFrame(
                    nose_dir=[float(x) for x in nose_dirs[i]],
                    line=[float(x) for x in lines[i]],
                    swing_deg=float(np.degrees(swing.angles_rad[i])),
                    drag_fraction=swing.drag_fractions[i],
                ),
            )
        elif time >= plan.fire_s:
            yield (
                time,
                RecoveryFrame(
                    nose_sep=float(
                        np.interp(time - plan.fire_s, curve.times_s, curve.distances_m)
                    )
                ),
            )
        else:
            yield time, RecoveryFrame()
