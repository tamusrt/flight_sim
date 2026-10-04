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

from flight_sim.canopy_swing import CanopySwing
from flight_sim.descent import RecoverySystem
from flight_sim.flight_computer import FlightComputer, RecoveryPlan, Sample
from flight_sim.integration import IntegrationConfiguration
from flight_sim.recovery_motion import EjectionCharge
from flight_sim.recovery_systems import RecoveryScheme, SingleSeparation
from flight_sim.vehicle.mass_properties import MassPropertiesSI
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.visualize import RecoveryFrame, TelemetryLog

_FT_TO_M = 0.3048

# Apogee is declared by the flight computer, the charge fires 3 s later (as
# the standard launch assumes for the canopy), the reef is cut at 2000 ft
FLIGHT_COMPUTER = FlightComputer(apogee_delay_s=3.0, main_altitude_m=2000.0 * _FT_TO_M)

# The nose section's mass, from SOL_4_30.ork. The suspension lines (0.9 of the
# canopy diameter), the nose's centre of mass (0.45 m from its harness) and
# the joint station (0.9144 m from the nose tip) are the defaults of
# ``SingleSeparation``.
EJECTION_CHARGE = EjectionCharge(nose_mass_kg=4.004)


def _sol_invictus_scheme(recovery: RecoverySystem) -> SingleSeparation:
    """Single separation with this module's settings, around a recovery system."""
    return SingleSeparation(recovery, FLIGHT_COMPUTER, EJECTION_CHARGE)


def lines_m(recovery: RecoverySystem) -> float:
    """Length of the suspension lines of the biggest canopy, in metres."""
    return _sol_invictus_scheme(recovery).lines_m


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
    return _sol_invictus_scheme(recovery).swing(centre_of_gravity_m)


def plan_full_recovery(
    flight: list[Sample],
    config: IntegrationConfiguration,
    recovery: RecoverySystem | RecoveryScheme,
    apogee_mass: MassPropertiesSI,
    properties: RocketProperties | None = None,
) -> RecoveryPlan:
    """Fly the descent with the flight computer, the ejection and the swing.

    Args:
        flight (list[Sample]): The ascent, ending at the true apogee.
        config (IntegrationConfiguration): Atmosphere and gravity.
        recovery (RecoverySystem | RecoveryScheme): A recovery architecture,
            which flies as it describes; a bare set of canopies flies as
            Sol Invictus's single separation, with this module's settings.
        apogee_mass (MassPropertiesSI): The rocket's mass properties after the
            burn.
        properties (RocketProperties | None): When given, the rocket tumbles
            from apogee to line stretch with the 6-DOF model (``tumble``).

    Returns:
        RecoveryPlan: The timeline and the descent flown with it.
    """
    scheme = (
        recovery
        if isinstance(recovery, RecoveryScheme)
        else _sol_invictus_scheme(recovery)
    )
    return scheme.plan(flight, config, apogee_mass, properties)


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
    timeline: list[tuple[float, str, str, dict[str, float]]] = []
    if plan.separation.separated:
        timeline.append(
            (
                plan.fire_s,
                "charge",
                "Separation charge",
                {"speed": plan.separation.speed_m_s},
            )
        )
        timeline.append(
            (
                plan.line_stretch_s,
                "stretch",
                "Line stretch",
                {"snatch": plan.separation.snatch_force_n},
            )
        )
    main = plan.main_separation
    if main is not None and plan.main_fire_s is not None and main.separated:
        timeline.append(
            (
                plan.main_fire_s,
                "charge",
                "Main separation charge",
                {"speed": main.speed_m_s},
            )
        )
        if plan.main_line_stretch_s is not None:
            timeline.append(
                (
                    plan.main_line_stretch_s,
                    "stretch",
                    "Main line stretch",
                    {"snatch": main.snatch_force_n},
                )
            )
    for deployment in plan.descent.deployments:
        timeline.append(
            (
                deployment.time_s,
                "disreef" if deployment.name.endswith("reef cut") else "deploy",
                deployment.name.capitalize(),
                {
                    "inflation": deployment.inflation_time_s or 0.0,
                    "load": deployment.peak_force_n,
                    "altitude": deployment.altitude_m,
                },
            )
        )
    for time, kind, name, extra in sorted(timeline, key=lambda e: e[0]):
        log.add_event(kind, name, time, **extra)


def frames(
    plan: RecoveryPlan, scheme: RecoveryScheme | None = None
) -> Iterator[tuple[float, RecoveryFrame]]:
    """The recovery hardware at each sample of the descent.

    Yields the sample time and what the scene draws besides the rocket: the
    nose coming out of the body, then the line to the canopy, the canopy's
    angle of attack and drag share, and the nose swinging on its leg.

    Args:
        plan (RecoveryPlan): The descent to draw.
        scheme (RecoveryScheme | None): The architecture it was flown with;
            by default, single separation with this module's settings.
    """
    return (scheme or _sol_invictus_scheme(plan.recovery)).frames(plan)
