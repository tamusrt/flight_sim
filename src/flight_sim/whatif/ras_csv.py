"""Turn RASAero II "Run Test" alpha files into a flight_sim aero CSV.

Put alpha0.txt ... alpha30.txt (RASAero's "Run Test" output, made by
tools/whatif/rasaero_sweep.py in the dynamics repo) in one folder, then run:

    python -m flight_sim.whatif.ras_csv --in DIR --out FILE

Above RASAero's last angle (30 degrees) the CSV carries on to 180 degrees in 10 degree
steps from a crossflow model fitted to RASAero's rows (flight_sim.aero_extend), for the
tumble after apogee.

Output columns match flight_sim's aero loader:
    Mach, Alpha, Phi, Cx, Cy, Cz, CMx, CMy, CMz

Conventions (they must match the sim):
  * Body axes: x out the nose, forces from RASAero's CN (normal) and CA (axial).
  * Cx = -CA; the normal force acts along -z_m, and phi is measured from body +Z toward
    body +Y to the crossflow (see flight_sim.utilities.dcm), so Cy = -CN*sin(phi) and
    Cz = -CN*cos(phi).
  * Moments are about the NOSE TIP (the default reference_point), made
    dimensionless by q * area * reference length, so --ref-length must equal
    reference_length in the sim (0.1524 m = the 6 in body diameter).
  * The centre of pressure sits xcp metres aft of the nose, i.e. at x = -xcp, so
    CMy = xcp*Cz/L, CMz = -xcp*Cy/L and CMx = 0 (RASAero gives no roll moment).
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flight_sim.aero_extend import extend

IN_TO_M = 0.0254


@dataclass(frozen=True)
class Options:
    """How the CSV is made; the defaults are the flight_sim conventions."""

    ref_length: float = 0.1524  # metres, the 6 in body diameter
    max_mach: float = 5.0
    phi_step: float = 15.0  # degrees
    cp: str = "alpha"  # or "header": the alpha-independent value the old MATLAB used


@dataclass(frozen=True)
class Summary:
    """What ``convert`` wrote."""

    alphas: list[int]
    mach: tuple[float, float]
    mach_points: int
    rows: int
    power_differs: bool


def read_alpha_file(path: Path) -> dict[int, dict[str, float]]:
    """Parse one alpha file into {Mach in hundredths: values}.

    Every Mach point is a block of 6 numeric lines (banner lines are skipped):
      1 flow summary | 2 Mach, x, x | 3 Mach, alpha, ... | 4 Mach, alpha, CN, xCP
      5 power-off: Mach, alpha, CL, CD, CN, CA | 6 power-on, same columns.
    """
    rows: list[tuple[int, list[float]]] = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            rows.append((number, [float(p) for p in parts]))
        except ValueError:
            continue  # "--> BODY TRANSITIONING ..." banner lines
    if len(rows) % 6:
        raise ValueError(
            f"{path.name}: {len(rows)} numeric lines is not a multiple of 6"
        )

    blocks: dict[int, dict[str, float]] = {}
    for start in range(0, len(rows), 6):
        block = rows[start : start + 6]
        first_line = block[0][0]
        lens = [len(values) for _, values in block]
        machs = {round(values[0], 4) for _, values in block}
        if len(machs) != 1 or lens[0] < 4 or lens[1:] != [3, 4, 4, 6, 6]:
            raise ValueError(
                f"{path.name} line {first_line}: unexpected block layout {lens}"
            )
        v = [values for _, values in block]
        if abs(v[3][2] - v[4][4]) > 2e-3:
            raise ValueError(f"{path.name} line {first_line}: CN columns disagree")
        mach_key = round(v[0][0] * 100)
        blocks[mach_key] = {
            "alpha": v[2][1],
            "cn": v[4][4],
            "ca": v[4][5],
            "xcp_in": v[3][3],  # centre of pressure at this alpha, inches from nose
            "xcp_header_in": v[1][2],  # constant-in-alpha value the old MATLAB used
            "power_diff": abs(v[4][4] - v[5][4]) + abs(v[4][5] - v[5][5]),
        }
    return blocks


def _parse_args(argv: list[str] | None, here: Path) -> argparse.Namespace:
    """Read the command line."""
    summary = (__doc__ or "").split("\n", maxsplit=1)[0]
    parser = argparse.ArgumentParser(description=summary)
    defaults = Options()
    parser.add_argument("--in", dest="source", type=Path, default=here)
    parser.add_argument("--out", type=Path, default=here / "aero.csv")
    parser.add_argument(
        "--ref-length", type=float, default=defaults.ref_length, help="metres"
    )
    parser.add_argument("--max-mach", type=float, default=defaults.max_mach)
    parser.add_argument(
        "--phi-step", type=float, default=defaults.phi_step, help="degrees"
    )
    parser.add_argument(
        "--cp",
        choices=("alpha", "header"),
        default=defaults.cp,
        help="alpha: xCP printed with CN at each alpha (default); "
        "header: the alpha-independent value the old MATLAB script used",
    )
    return parser.parse_args(argv)


def _load_tables(
    source: Path, max_mach: float
) -> tuple[dict[int, dict[int, dict[str, float]]], list[int]]:
    """Read every alpha file, check they agree, and pick the Mach points.

    Returns:
        The tables keyed by alpha in degrees, and the Mach keys (hundredths)
        to write: every 0.01 up to Mach 1.5, every 0.05 above, up to max_mach.
    """
    files = sorted(
        (int(m.group(1)), p)
        for p in source.glob("alpha*.txt")
        if (m := re.fullmatch(r"alpha(\d+)\.txt", p.name))
    )
    if not files:
        raise SystemExit(f"No alpha*.txt files found in {source}")

    tables = {alpha: read_alpha_file(path) for alpha, path in files}
    mach_sets = {tuple(sorted(t)) for t in tables.values()}
    if len(mach_sets) != 1:
        raise SystemExit("The alpha files do not share the same Mach points")
    for alpha, table in tables.items():
        found = {round(b["alpha"]) for b in table.values()}
        if found != {alpha}:
            raise SystemExit(f"alpha{alpha}.txt contains alpha values {sorted(found)}")
    mach_keys = [
        k
        for k in mach_sets.pop()
        if k <= round(max_mach * 100) and (k <= 150 or k % 5 == 0)
    ]
    return tables, mach_keys


def _rows(
    block: dict[str, float], mach_key: int, alpha: int, args: Options
) -> list[list[float | int]]:
    """Return the CSV rows of one Mach and alpha, one per phi."""
    xcp_in = block["xcp_in"] if args.cp == "alpha" else block["xcp_header_in"]
    xcp = xcp_in * IN_TO_M
    rows: list[list[float | int]] = []
    steps = round(360 / args.phi_step)
    for index in range(steps + 1):
        phi = index * args.phi_step
        sin_phi, cos_phi = math.sin(math.radians(phi)), math.cos(math.radians(phi))
        cy, cz = -block["cn"] * sin_phi, -block["cn"] * cos_phi
        cmy, cmz = xcp * cz / args.ref_length, -xcp * cy / args.ref_length
        rows.append([mach_key / 100, alpha, phi, -block["ca"], cy, cz, 0.0, cmy, cmz])
    return rows


def _with_tumble_rows(
    tables: dict[int, dict[int, dict[str, float]]],
    alphas: list[int],
    mach_key: int,
    options: Options,
) -> list[tuple[int, dict[str, float]]]:
    """RASAero's rows for one Mach, then 40 to 180 degrees from the crossflow model.

    RASAero stops at 30 degrees; the tumble after apogee needs every angle, so the
    rows above are made by ``flight_sim.aero_extend`` from RASAero's own rows.
    """
    key = "xcp_in" if options.cp == "alpha" else "xcp_header_in"
    blocks = [tables[a][mach_key] for a in alphas]
    added, cn, ca, xcp = extend(
        np.array(alphas, dtype=float),
        mach_key / 100,
        np.array([b["cn"] for b in blocks]),
        np.array([b["ca"] for b in blocks]),
        np.array([b[key] for b in blocks]),
        options.ref_length / IN_TO_M,
    )
    rows = list(zip(alphas, blocks, strict=True))
    for deg, n, a, x in zip(added, cn, ca, xcp, strict=True):
        rows.append(
            (
                round(deg),
                {
                    "cn": float(n),
                    "ca": float(a),
                    "xcp_in": float(x),
                    "xcp_header_in": float(x),
                },
            )
        )
    return rows


def convert(source: Path, out: Path, options: Options | None = None) -> Summary:
    """Read the alpha files in ``source``; write the flight_sim aero CSV to ``out``."""
    options = options or Options()
    tables, mach_keys = _load_tables(source, options.max_mach)
    alphas = sorted(tables)
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Mach", "Alpha", "Phi", "Cx", "Cy", "Cz", "CMx", "CMy", "CMz"])
        for mach_key in mach_keys:
            for alpha, block in _with_tumble_rows(tables, alphas, mach_key, options):
                for row in _rows(block, mach_key, alpha, options):
                    writer.writerow(
                        [f"{x + 0.0:.6g}" if isinstance(x, float) else x for x in row]
                    )
                    count += 1
    power_diff = max(b["power_diff"] for t in tables.values() for b in t.values())
    return Summary(
        alphas=alphas,
        mach=(mach_keys[0] / 100, mach_keys[-1] / 100),
        mach_points=len(mach_keys),
        rows=count,
        power_differs=power_diff > 1e-6,
    )


def main(argv: list[str] | None = None, here: Path | None = None) -> None:
    """Build the CSV from the command line; ``here`` anchors the default paths."""
    args = _parse_args(argv, here or Path.cwd())
    options = Options(args.ref_length, args.max_mach, args.phi_step, args.cp)
    summary = convert(args.source, args.out, options)
    print(
        f"Read {len(summary.alphas)} alpha files "
        f"(alpha {summary.alphas[0]}-{summary.alphas[-1]} deg)"
    )
    print(
        f"Mach {summary.mach[0]:g} to {summary.mach[1]:g}: {summary.mach_points} points"
    )
    print(f"Phi 0-360 in {options.phi_step:g} deg steps")
    print(f"Wrote {summary.rows} rows to {args.out}")
    print(f"Centre of pressure source: {options.cp}")
    if summary.power_differs:
        print("WARNING: power-on and power-off coefficients differ; power-off was used")
    else:
        print("Power-on and power-off coefficients are identical (nozzle diameter 0)")


if __name__ == "__main__":
    main()
