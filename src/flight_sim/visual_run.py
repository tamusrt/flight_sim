"""Run the standard flight and write the 3D viewer. Does not touch the sim.

Usage: uv run python -m flight_sim.visual_run [--max-time SECONDS] [--apogee]

The flight is the standard launch of ``python -m flight_sim``, flown by the
same code and only drawn. After apogee the descent is flown with the extended
recovery model of ``recovery_extension`` (flight computer, nose ejection,
rocket swinging under its canopy); ``--apogee`` stops at apogee instead.
"""

# The flight loop deliberately mirrors __main__.main, which it must not change
# pylint: disable=duplicate-code

import argparse

from flight_sim import recovery_extension
from flight_sim.__main__ import (
    LAUNCH_RAIL,
    RECOVERY,
    get_default_config,
    get_default_properties,
    get_launch_state,
    launch_constraints,
)
from flight_sim.events import APOGEE, IMPACT, FlightEvent, peak_vertical_velocity
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import scalar
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
    args = parser.parse_args()
    max_time: float | None = args.max_time

    properties = get_default_properties()
    state = get_launch_state()
    config = get_default_config()
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)
    log = TelemetryLog(properties, config)
    log.describe_rail(LAUNCH_RAIL)
    log.describe_recovery(RECOVERY)

    dt = scalar(0.01, "s")
    current_time = 0.0
    hit: FlightEvent | None = None
    on_rail = True
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
        was_on_rail = on_rail
        state, on_rail = launch_constraints(state, on_rail)
        if was_on_rail and not on_rail:
            log.add_event("rail", "Rail exit", current_time)

    if hit is APOGEE and not args.apogee:
        _log_recovery(log, flight, config, max_time)

    print(f"Viewer written to {write_viewer(log)}")


def _log_recovery(
    log: TelemetryLog,
    flight: list[tuple[float, RocketState]],
    config: IntegrationConfiguration,
    max_time: float | None,
) -> None:
    """Fly and log the descent with the extended recovery model."""
    plan = recovery_extension.plan_full_recovery(flight, config, RECOVERY)
    print(f"charge fires at {plan.fire_s:.2f} seconds")
    print(f"line stretch at {plan.line_stretch_s:.2f} seconds")
    for deployment in plan.descent.deployments:
        print(f"{deployment.name} at {deployment.time_s:.2f} seconds")
    print(f"impact at {plan.descent.times_s[-1]:.2f} seconds")
    recovery_extension.log_events(log, plan)
    for (time, frame), sample in zip(
        recovery_extension.frames(plan), plan.descent.states, strict=True
    ):
        if max_time is not None and time > max_time:
            break
        log.record(time, sample, frame)


if __name__ == "__main__":
    main()
