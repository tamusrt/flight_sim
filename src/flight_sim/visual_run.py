"""Run the standard flight and write the 3D viewer. Does not touch the sim.

Usage: uv run python -m flight_sim.visual_run [--max-time SECONDS]
"""

# The flight loop deliberately mirrors __main__.main, which it must not change
# pylint: disable=duplicate-code

import argparse

from flight_sim.__main__ import (
    descend,
    get_default_config,
    get_default_properties,
    get_launch_state,
    launch_constraints,
)
from flight_sim.events import APOGEE, IMPACT, FlightEvent, peak_vertical_velocity
from flight_sim.integration import adaptive_step
from flight_sim.units import scalar
from flight_sim.visualize import TelemetryLog, write_viewer


def main() -> None:
    """Fly the default rocket, logging telemetry, then open the viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-time", type=float, default=None, help="stop after this many seconds"
    )
    max_time = parser.parse_args().max_time

    properties = get_default_properties()
    state = get_launch_state()
    config = get_default_config()
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)
    log = TelemetryLog(properties, config)

    dt = scalar(0.01, "s")
    current_time = 0.0
    hit: FlightEvent | None = None
    on_rail = True
    while True:
        log.record(current_time, state)
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
        state, on_rail = launch_constraints(state, on_rail)

    if hit is APOGEE:
        descent = descend(current_time, state, config)
        for sample_time, sample in zip(descent.times_s, descent.states, strict=True):
            if max_time is not None and sample_time > max_time:
                break
            log.record(sample_time, sample)

    print(f"Viewer written to {write_viewer(log)}")


if __name__ == "__main__":
    main()
