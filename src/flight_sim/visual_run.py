"""Run the standard flight and write the 3D viewer.

Usage: uv run python -m flight_sim.visual_run [--max-time SECONDS]
"""

import argparse

from flight_sim.__main__ import get_default_config, get_default_properties, run_flight
from flight_sim.visualize import TelemetryLog, write_viewer


def main() -> None:
    """Fly the default rocket, logging telemetry, then open the viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-time", type=float, default=None, help="stop after this many seconds"
    )
    max_time = parser.parse_args().max_time

    properties = get_default_properties()
    config = get_default_config()
    log = TelemetryLog(properties, config)
    result = run_flight(properties, config, max_time_s=max_time, on_sample=log.record)
    log.events = result.events
    print(f"Viewer written to {write_viewer(log)}")


if __name__ == "__main__":
    main()
