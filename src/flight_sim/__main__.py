"""Core FS runner."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.events import APOGEE, IMPACT, peak_vertical_velocity
from flight_sim.flight_event import FlightEvent
from flight_sim.integration import (
    IntegrationConfiguration,
    TruthConfiguration,
    adaptive_step,
)
from flight_sim.units import matrix, scalar, vector
from flight_sim.utilities.data_loader import aero_table_from_csv
from flight_sim.utilities.dcm import aero_angles, body_to_world
from flight_sim.utilities.quaternion import Quaternion
from flight_sim.vehicle.engine import PropellantGrain, solid_engine_from_csv
from flight_sim.vehicle.mass_properties import MassProperties
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState


# ============================================================
# Sol Invictus known OpenRocket locations.
#
# flight_sim body coordinate convention:
#   Nose tip = X = 0.0 m
#   Aft of nose = negative X
# ============================================================

INCH_TO_M = 0.0254

DRY_CG_X_M = -(106.0 * INCH_TO_M)          # -2.6924 m
LOADED_CG_X_M = -(136.0 * INCH_TO_M)       # -3.4544 m
OPENROCKET_CP_X_M = -(166.0 * INCH_TO_M)   # -4.2164 m


def get_default_state() -> RocketState:
    """Generate the standard initial state."""
    return RocketState(
        position=vector((0.0, 0.0, 0.0), "m"),
        velocity=vector((0.0, 0.0, 0.0), "m/s"),
        angular_velocity=vector((0.0, 0.0, 0.0), "rad/s"),
        orientation=Quaternion(
            q_x=0.0,
            q_y=0.0,
            q_z=0.0,
            q_w=1.0,
        ),
    )


def get_default_properties() -> RocketProperties:
    """Build fallback/default rocket properties."""

    grain = PropellantGrain(
        mass=scalar(5.0, "kg"),
        length=scalar(0.75, "m"),
        outer_diameter=scalar(0.0762, "m"),
        core_diameter=scalar(0.03, "m"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
    )

    casing = MassProperties(
        mass=scalar(3.0, "kg"),
        cg_location=vector((-2.6, 0.0, 0.0), "m"),
        inertia=matrix(
            np.diag((0.0061, 0.21, 0.21)),
            "kg*m**2",
        ),
    )

    return RocketProperties(
        aero_table=aero_table_from_csv(
            "data/aero/estimated_aero.csv",
            reference_area=scalar(0.0182414692, "m**2"),
            reference_length=scalar(0.1524, "m"),
            reference_point=vector((0.0, 0.0, 0.0), "m"),
            frame="missile",
        ),
        engine=solid_engine_from_csv(
            "tests/test_data/standard_motor.csv",
            grain,
            casing,
        ),
        dry_mass_properties=MassProperties(
            mass=scalar(17.0, "kg"),
            cg_location=vector((-1.0, 0.0, 0.0), "m"),
            inertia=matrix(
                np.diag((2.4, 135.0, 135.0)),
                "kg*m**2",
            ),
        ),
    )


def load_rocket(
    name: str,
    properties: RocketProperties,
    rail: LaunchRail,
) -> tuple[RocketProperties, LaunchRail, dict[str, Any]]:
    """Load settings from rockets/<name>.py."""

    path = Path("rockets") / f"{name}.py"

    if not path.is_file():
        raise SystemExit(f"No settings file found at {path.resolve()}")

    spec = importlib.util.spec_from_file_location(
        f"rocket_{name}",
        path,
    )

    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_properties"):
        properties = module.get_properties()

    if hasattr(module, "get_rail"):
        rail = module.get_rail()

    truth_options: dict[str, Any] = {}

    if hasattr(module, "get_truth_options"):
        truth_options = module.get_truth_options()

    return properties, rail, truth_options


def get_engine_thrust_newtons(
    engine: Any,
    time_s: float,
    ambient_pressure_pa: float,
) -> float:
    """Return the actual thrust currently used by the simulator."""

    return float(
        engine.get_thrust(
            time_s,
            ambient_pressure_pa,
        )
    )


def get_aero_diagnostics(
    state: RocketState,
    properties: RocketProperties,
    config: IntegrationConfiguration,
) -> tuple[float, float]:
    """Return displayed drag coefficient and validated OpenRocket CP.

    CP is intentionally set to the known OpenRocket value, rather than being
    reconstructed from RASAero CMy/CMz. The converted RASAero moment-table
    reference point/sign convention has not yet been independently verified.
    """

    position_m = state.position.m_as("m")
    velocity_mps = state.velocity.m_as("m/s")

    altitude_m = (
        float(config.truth.launch_elevation.m_as("m"))
        + float(position_m[0])
    )

    conditions = config.truth.atmosphere.conditions(altitude_m)

    air_velocity_world = (
        velocity_mps
        - config.truth.wind.velocity(altitude_m)
    )

    to_world = body_to_world(state.orientation)
    air_velocity_body = to_world.T @ air_velocity_world

    speed_mps = float(np.linalg.norm(air_velocity_body))

    if speed_mps < 1e-9:
        return float("nan"), OPENROCKET_CP_X_M

    alpha_rad, phi_rad = aero_angles(air_velocity_body)

    mach = speed_mps / conditions.speed_of_sound

    coefficients = properties.aero_table(
        mach,
        np.degrees(alpha_rad),
        np.degrees(phi_rad),
    )

    # In this simulator, Cx acts along body X.
    # Body +X points toward the nose, so normal drag is normally negative Cx.
    drag_coefficient = -float(coefficients.cx)

    # Known OpenRocket CP location:
    # 166 in aft of nose = 4.2164 m aft = -4.2164 body X.
    cp_x_m = OPENROCKET_CP_X_M

    return drag_coefficient, cp_x_m


def print_mass_and_cg_check(properties: RocketProperties) -> None:
    """Print mass and CG checks in user-friendly units."""

    print("\n--- MASS / CG CHECK ---")

    for check_time in (0.0, 12.2, 15.85):
        mass_properties = properties.mass_properties(check_time)

        # MassPropertiesSI stores ordinary SI values.
        mass_kg = float(mass_properties.mass)
        cg_x_m = float(mass_properties.cg_location[0])

        print(
            f"t = {check_time:5.2f} s | "
            f"mass = {mass_kg:7.3f} kg | "
            f"CG = {-cg_x_m:6.3f} m aft of nose | "
            f"CG = {-cg_x_m / INCH_TO_M:6.1f} in aft of nose"
        )

    print(
        f"\nExpected loaded CG: {-LOADED_CG_X_M:.3f} m "
        f"({-LOADED_CG_X_M / INCH_TO_M:.1f} in) aft of nose"
    )

    print(
        f"Dry-airframe CG:     {-DRY_CG_X_M:.3f} m "
        f"({-DRY_CG_X_M / INCH_TO_M:.1f} in) aft of nose"
    )

    print(
        f"OpenRocket CP:       {-OPENROCKET_CP_X_M:.3f} m "
        f"({-OPENROCKET_CP_X_M / INCH_TO_M:.1f} in) aft of nose"
    )


def main(rocket: str | None = None) -> None:  # pylint: disable=too-many-statements
    """Run the complete flight simulation."""

    properties = get_default_properties()

    rail = LaunchRail(
        length=scalar(17.0, "ft"),
        elevation=scalar(85.0, "deg"),
        azimuth=scalar(0.0, "deg"),
    )

    truth_options: dict[str, Any] = {}

    if rocket is not None:
        properties, rail, truth_options = load_rocket(
            rocket,
            properties,
            rail,
        )

    print_mass_and_cg_check(properties)

    state = rail.mount(get_default_state())

    config = IntegrationConfiguration(
        truth=TruthConfiguration(
            launch_rail=rail,
            **truth_options,
        )
    )

    events = (
        peak_vertical_velocity(properties, config),
        APOGEE,
        IMPACT,
    )

    dt = scalar(0.01, "s")
    current_time = 0.0

    telemetry: list[dict[str, Any]] = []
    step_lengths: list[float] = []
    hit: FlightEvent | None = None

    while True:
        position_m = state.position.m_as("m")
        velocity_mps = state.velocity.m_as("m/s")
        angular_rate_rad_s = state.angular_velocity.m_as("rad/s")

        altitude_m = (
            float(config.truth.launch_elevation.m_as("m"))
            + float(position_m[0])
        )

        atmospheric_conditions = config.truth.atmosphere.conditions(
            altitude_m
        )

        mass_properties = properties.mass_properties(current_time)

        mass_kg = float(mass_properties.mass)
        cg_now_m = np.asarray(
            mass_properties.cg_location,
            dtype=float,
        )

        thrust_n = get_engine_thrust_newtons(
            properties.engine,
            current_time,
            atmospheric_conditions.pressure,
        )

        drag_coefficient, cp_x_m = get_aero_diagnostics(
            state,
            properties,
            config,
        )

        telemetry.append(
            {
                "Time (s)": current_time,
                "Inertial Position (m)": position_m,
                "Inertial Velocity (m/s)": velocity_mps,
                "Angular Rate (rad/s)": angular_rate_rad_s,
                "Mass (kg)": mass_kg,
                "CG (m)": cg_now_m,
                "CP X (m)": cp_x_m,
                "Thrust (N)": thrust_n,
                "Drag Coefficient (-)": drag_coefficient,
            }
        )

        if hit is not None:
            print(f"{hit.name} at {current_time:.2f} seconds")

        if hit is IMPACT:
            break

        state, dt_taken, dt, hit = adaptive_step(
            current_time,
            state,
            properties,
            config,
            dt,
            events=events,
        )

        current_time += float(dt_taken.m_as("s"))
        step_lengths.append(float(dt_taken.m_as("s")))

    df = pd.DataFrame(telemetry)

    time_s = df["Time (s)"].to_numpy(dtype=float)

    position_m = np.vstack(df["Inertial Position (m)"])
    velocity_mps = np.vstack(df["Inertial Velocity (m/s)"])
    angular_rate_rad_s = np.vstack(df["Angular Rate (rad/s)"])

    cg_x_m = np.vstack(df["CG (m)"])[:, 0]
    cp_x_m = df["CP X (m)"].to_numpy(dtype=float)

    print(f"\nApogee: {position_m[:, 0].max():.1f} m")
    print(
        f"Max Vertical Velocity: "
        f"{velocity_mps[:, 0].max():.1f} m/s"
    )
    print(f"Peak Thrust: {df['Thrust (N)'].max():.1f} N")

    if "PYTEST_CURRENT_TEST" not in os.environ:
        log_path = f"jarvis_log_{rocket or 'default'}.csv"

        pd.DataFrame(
            {
                "Time (s)": time_s,
                "Altitude (m)": position_m[:, 0],
                "Vertical Velocity (m/s)": velocity_mps[:, 0],
                "Mass (kg)": df["Mass (kg)"],
                "CG X (m)": cg_x_m,
                "CP X (m)": cp_x_m,
                "Thrust (N)": df["Thrust (N)"],
                "Drag Coefficient (-)": df["Drag Coefficient (-)"],
            }
        ).to_csv(log_path, index=False)

        print(f"Wrote {log_path}")

    world_axes = ["X (up)", "Y (east)", "Z (north)"]

    # Position, velocity, and body rate.
    plt.figure(figsize=(10, 12))

    plt.subplot(3, 1, 1)
    plt.plot(time_s, position_m, linewidth=2)
    plt.title("Flight Profile")
    plt.ylabel("Position (m)")
    plt.legend(world_axes)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_s, velocity_mps, linewidth=2)
    plt.ylabel("Velocity (m/s)")
    plt.legend(world_axes)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_s, angular_rate_rad_s, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Rate (rad/s)")
    plt.legend(["Body X (roll)", "Body Y", "Body Z"])
    plt.grid(True)

    plt.tight_layout()

    # Adaptive time-step plot.
    plt.figure(figsize=(10, 5))
    plt.plot(step_lengths, linewidth=2)
    plt.title("Adaptive Integration Step Length")
    plt.xlabel("Step Number")
    plt.ylabel("dt Taken (s)")
    plt.grid(True)
    plt.tight_layout()

    # Thrust plot.
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_s,
        df["Thrust (N)"],
        linewidth=2,
        color="tab:red",
        label="Motor Thrust",
    )
    plt.title("Motor Thrust vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Drag coefficient plot.
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_s,
        df["Drag Coefficient (-)"],
        linewidth=2,
        color="tab:purple",
        label="Displayed Cd",
    )
    plt.title("Drag Coefficient vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Drag Coefficient (-)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # CG and CP plot.
    plt.figure(figsize=(10, 5))

    # Negate body coordinates so the chart shows positive distance aft of nose.
    plt.plot(
        time_s,
        -cg_x_m,
        linewidth=2,
        color="tab:blue",
        label="CG (simulation)",
    )

    plt.plot(
        time_s,
        -cp_x_m,
        linewidth=2,
        color="tab:green",
        linestyle="--",
        label="CP (OpenRocket)",
    )

    plt.axhline(
        -LOADED_CG_X_M,
        color="tab:blue",
        linestyle=":",
        alpha=0.7,
        label="Expected Loaded CG: 136 in",
    )

    plt.axhline(
        -DRY_CG_X_M,
        color="tab:cyan",
        linestyle=":",
        alpha=0.7,
        label="Dry-Airframe CG: 106 in",
    )

    plt.axhline(
        -OPENROCKET_CP_X_M,
        color="tab:green",
        linestyle=":",
        alpha=0.7,
        label="OpenRocket CP: 166 in",
    )

    plt.title("Center of Gravity and Center of Pressure")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance Aft of Nose (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if "PYTEST_CURRENT_TEST" not in os.environ:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)