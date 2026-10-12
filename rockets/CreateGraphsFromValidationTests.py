from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. SETUP & FILE PATHS
# ==========================================

rockets_dir = Path(__file__).resolve().parent

ras_path = rockets_dir / "FlightTestRAS_SOL.CSV"
# Written by `uv run python -m flight_sim SOLINV` in the repository root
jarvis_path = rockets_dir.parent / "jarvis_log_SOLINV.csv"
or_path = rockets_dir / "OR-COmparsion-Sim-SOL-CORRECT-FULL.csv"


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def require_columns(dataframe, columns, dataframe_name):
    """Raise a helpful error if expected CSV columns are missing."""
    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise KeyError(
            f"\nMissing column(s) in {dataframe_name}:\n"
            f"{missing_columns}\n\n"
            f"Columns actually found:\n"
            f"{list(dataframe.columns)}"
        )


def convert_to_numeric(dataframe, columns):
    """Convert selected columns to numeric, replacing invalid values with NaN."""
    for column in columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")


# ==========================================
# 3. LOAD DATA
# ==========================================

# utf-8-sig handles possible Windows BOM characters.
# skipinitialspace handles spaces after commas.
ras_df = pd.read_csv(ras_path, encoding="utf-8-sig", skipinitialspace=True)
jarvis_df = pd.read_csv(jarvis_path, encoding="utf-8-sig", skipinitialspace=True)

# Strip accidental spaces from CSV column headers.
ras_df.columns = ras_df.columns.str.strip()
jarvis_df.columns = jarvis_df.columns.str.strip()

# OpenRocket sometimes stores its header as a commented line.
# Read the first seven data columns and provide consistent names manually.
or_cols = [
    "Time (s)",
    "Altitude (ft)",
    "Vertical velocity (ft/s)",
    "Mass (oz)",
    "CG location (in)",
    "Thrust (N)",
    "Mach number",
]

or_df = pd.read_csv(
    or_path,
    comment="#",
    names=or_cols,
    header=None,
    usecols=range(len(or_cols)),
    encoding="utf-8-sig",
    skipinitialspace=True,
)

# ==========================================
# 4. CHECK REQUIRED COLUMNS
# ==========================================

ras_required = [
    "Time (sec)",
    "Altitude (ft)",
    "Vel-V (ft/sec)",
    "Weight (lb)",
    "CG (in)",
]

jarvis_required = [
    "Time (s)",
    "Altitude (m)",
    "Vertical Velocity (m/s)",
    "Mass (kg)",
    "CG X (m)",
]

or_required = [
    "Time (s)",
    "Altitude (ft)",
    "Vertical velocity (ft/s)",
    "Mass (oz)",
    "CG location (in)",
]

require_columns(ras_df, ras_required, "RASAero CSV")
require_columns(jarvis_df, jarvis_required, "Jarvis CSV")
require_columns(or_df, or_required, "OpenRocket CSV")


# ==========================================
# 5. CONVERT DATA TO NUMBERS
# ==========================================

convert_to_numeric(ras_df, ras_required)
convert_to_numeric(jarvis_df, jarvis_required)
convert_to_numeric(or_df, or_required)


# ==========================================
# 6. PLOTTING
# ==========================================

# --- Graph 1: Altitude ---
plt.figure("Altitude Comparison", figsize=(8, 6))

plt.plot(
    ras_df["Time (sec)"],
    ras_df["Altitude (ft)"] * 0.3048,
    label="RASAero",
    linewidth=2,
)

plt.plot(
    jarvis_df["Time (s)"],
    jarvis_df["Altitude (m)"],
    label="Jarvis Sim",
    linewidth=2,
    linestyle="--",
)

plt.plot(
    or_df["Time (s)"],
    or_df["Altitude (ft)"] * 0.3048,
    label="OpenRocket",
    linewidth=2,
    linestyle=":",
)

plt.title("Flight Altitude over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Altitude (meters)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Graph 2: Vertical Velocity ---
plt.figure("Velocity Comparison", figsize=(8, 6))

plt.plot(
    ras_df["Time (sec)"],
    ras_df["Vel-V (ft/sec)"] * 0.3048,
    label="RASAero",
    linewidth=2,
)

plt.plot(
    jarvis_df["Time (s)"],
    jarvis_df["Vertical Velocity (m/s)"],
    label="Jarvis Sim",
    linewidth=2,
    linestyle="--",
)

plt.plot(
    or_df["Time (s)"],
    or_df["Vertical velocity (ft/s)"] * 0.3048,
    label="OpenRocket",
    linewidth=2,
    linestyle=":",
)

plt.title("Vertical Velocity over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Vertical Velocity (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Graph 3: Mass ---
plt.figure("Mass Comparison", figsize=(8, 6))

plt.plot(
    ras_df["Time (sec)"],
    ras_df["Weight (lb)"] * 0.45359237,
    label="RASAero",
    linewidth=2,
)

plt.plot(
    jarvis_df["Time (s)"],
    jarvis_df["Mass (kg)"],
    label="Jarvis Sim",
    linewidth=2,
    linestyle="--",
)

plt.plot(
    or_df["Time (s)"],
    or_df["Mass (oz)"] * 0.028349523125,
    label="OpenRocket",
    linewidth=2,
    linestyle=":",
)

plt.title("Vehicle Mass over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Mass (kg)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# --- Graph 4: Center of Gravity ---
plt.figure("CG Comparison", figsize=(8, 6))

plt.plot(
    ras_df["Time (sec)"],
    ras_df["CG (in)"] * 0.0254,
    label="RASAero",
    linewidth=2,
)

# Negative sign needs to be used to align 
plt.plot(
    jarvis_df["Time (s)"],
    -jarvis_df["CG X (m)"],
    label="Jarvis Sim",
    linewidth=2,
    linestyle="--",
)

plt.plot(
    or_df["Time (s)"],
    or_df["CG location (in)"] * 0.0254,
    label="OpenRocket",
    linewidth=2,
    linestyle=":",
)

plt.title("Center of Gravity Shift over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("CG Distance from Nose (meters)")
plt.grid(True)
plt.legend()
plt.tight_layout()


# ==========================================
# 7. DISPLAY ALL FIGURES
# ==========================================

plt.show()