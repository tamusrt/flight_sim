"""Run the standard flight and write the 3D viewer. Does not touch the sim.

Usage: uv run python -m flight_sim.visual_run [--max-time SECONDS] [--apogee]

The flight is the standard launch of ``python -m flight_sim``, flown by the
same code and only drawn. After apogee the rocket is flown with the 6-DOF model
until line stretch, so it can tumble (``tumble``), then the descent is flown with
the extended recovery model of ``recovery_extension`` (flight computer, nose
ejection, rocket swinging under its canopy); ``--apogee`` stops at apogee instead.
"""

# The flight loop deliberately mirrors __main__.main, which it must not change
# pylint: disable=duplicate-code

import argparse
import os

from flight_sim import recovery_extension
from flight_sim.__main__ import (
    RocketProfile,
    add_rocket_arguments,
    get_default_config,
    get_launch_state,
    get_profile_properties,
    select_rocket,
)
from flight_sim.events import APOGEE, IMPACT, peak_vertical_velocity
from flight_sim.flight_event import FlightEvent
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.real_flight import real_flight_telemetry
from flight_sim.units import scalar
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState
from flight_sim.visualize import TelemetryLog, write_viewer


def main() -> None:
    """Fly the default rocket, logging telemetry, then open the viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-time", type=float, default=None, help="stop after this many seconds"
    )
    parser.add_argument(
        "--apogee", action="store_true", help="stop at apogee, with no descent"
    )
    parser.add_argument(
        "--real",
        metavar="FILE",
        default=None,
        help="Blue Raven log to offer in the viewer (default: the rocket's own)",
    )
    parser.add_argument(
        "--output", default="flight.html", help="where to write the viewer page"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="do not open the page in a browser"
    )
    add_rocket_arguments(parser)
    args = parser.parse_args()
    max_time: float | None = args.max_time
    profile, aero_file = select_rocket(args)
    print(f"Flying {profile.name} with {aero_file}")

    properties = get_profile_properties(profile, aero_file)
    state = get_launch_state(profile)
    config = get_default_config(profile)
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)
    log = TelemetryLog(properties, config)
    log.describe_rail(profile.rail)
    log.describe_recovery(profile.recovery)

    dt = scalar(0.01, "s")
    current_time = 0.0
    hit: FlightEvent | None = None
    flight: list[tuple[float, RocketState]] = []
    while True:
        log.record(current_time, state)
        flight.append((current_time, state))
        if hit is not None:
            print(f"{hit.name} at {current_time:.2f} seconds")
        if (
            hit is APOGEE
            or hit is IMPACT
            or (max_time is not None and current_time >= max_time)
        ):
            break
        state, dt_taken, dt, hit = adaptive_step(
            current_time, state, properties, config, dt, events=events
        )
        current_time += float(dt_taken.m_as("s"))
        if hit is not None and hit.name == "rail exit":
            log.add_event("rail", "Rail exit", current_time)

    if hit is APOGEE and not args.apogee and profile.scheme is not None:
        _log_recovery(
            log, flight, properties, config, max_time=max_time, profile=profile
        )

    real = None
    real_path = args.real or profile.flight_data
    if real_path is not None and os.path.isfile(real_path):
        azimuth = float(profile.rail.azimuth.m_as("deg"))
        real = real_flight_telemetry(real_path, azimuth)
        print(f"Loaded real flight from {real_path}")
    elif args.real is not None:
        parser.error(f"flight log not found: {args.real}")
    # the OpenRocket run of the same simulation, which the dynamics site can show
    openrocket = (
        {"ork": os.path.basename(args.ork), "sim": args.ork_sim}
        if args.ork is not None and args.ork_sim is not None
        else None
    )
    written = write_viewer(
        log,
        args.output,
        open_browser=not args.no_open,
        real=real,
        openrocket=openrocket,
    )
    print(f"Viewer written to {written}")


def _log_recovery(
    log: TelemetryLog,
    flight: list[tuple[float, RocketState]],
    properties: RocketProperties,
    config: IntegrationConfiguration,
    *,
    max_time: float | None,
    profile: RocketProfile,
) -> None:
    """Fly and log the descent with the extended recovery model."""
    scheme = profile.scheme
    if scheme is None:
        return
    apogee_mass = properties.mass_properties(flight[-1][0])
    plan = recovery_extension.plan_full_recovery(
        flight, config, scheme, apogee_mass, properties
    )
    print(f"Recovery: {scheme.kind}")
    if plan.separation.separated:
        print(f"charge fires at {plan.fire_s:.2f} seconds")
        print(f"line stretch at {plan.line_stretch_s:.2f} seconds")
    if plan.main_fire_s is not None and plan.main_line_stretch_s is not None:
        print(f"main charge fires at {plan.main_fire_s:.2f} seconds")
        print(f"main line stretch at {plan.main_line_stretch_s:.2f} seconds")
    for deployment in plan.descent.deployments:
        print(f"{deployment.name} at {deployment.time_s:.2f} seconds")
    print(f"impact at {plan.descent.times_s[-1]:.2f} seconds")
    recovery_extension.log_events(log, plan)
    for (time, frame), sample in zip(
        recovery_extension.frames(plan, scheme),
        plan.descent.states,
        strict=True,
    ):
        if max_time is not None and time > max_time:
            break
        log.record(time, sample, frame)


if __name__ == "__main__":
    main()
