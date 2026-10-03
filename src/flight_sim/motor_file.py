"""Read a motor file from the propulsion model: RASP ``.eng`` or RockSim ``.rse``.

The page and the 6-DOF flight need the thrust curve, the propellant mass and the
total mass. An ``.rse`` is converted to the same numbers (grams there, kilograms
here) so either file can be the newest one in the thrust-curve folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class MotorFile:
    """A motor as its file gives it."""

    name: str
    thrust: list[list[float]]  # [time s, thrust N], from zero thrust at ignition
    propellant_kg: float | None
    total_kg: float | None
    diameter_mm: float
    length_mm: float
    maker: str

    @property
    def burn_s(self) -> float:
        """Time of the last sample."""
        return self.thrust[-1][0]

    @property
    def impulse_ns(self) -> float:
        """Total impulse by the trapezoid rule."""
        return sum(
            0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in pairwise(self.thrust)
        )

    @property
    def peak_n(self) -> float:
        """Highest thrust."""
        return max(f for _, f in self.thrust)


def _with_zero(pairs: list[list[float]], path: Path) -> list[list[float]]:
    """Check the samples and start them from zero thrust at ignition."""
    if not pairs:
        raise ValueError(f"{path}: no thrust samples")
    if any(b[0] <= a[0] for a, b in pairwise(pairs)):
        raise ValueError(f"{path}: sample times do not increase")
    return pairs if pairs[0][0] <= 0.0 else [[0.0, 0.0], *pairs]


def _load_eng(path: Path) -> MotorFile:
    """RASP: ``;`` comments, a description line, then time and thrust pairs."""
    header: list[str] | None = None
    pairs: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(";", 1)[0].split()
        if not parts:
            continue
        if header is None:
            header = parts
            continue
        try:
            pairs.append([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            break  # the description line of a second motor
    if header is None or len(header) < 6:
        raise ValueError(f"{path}: no motor description line")
    return MotorFile(
        name=header[0],
        thrust=_with_zero(pairs, path),
        propellant_kg=float(header[4]) or None,
        total_kg=float(header[5]) or None,
        diameter_mm=float(header[1]),
        length_mm=float(header[2]),
        maker=header[6] if len(header) > 6 else "",
    )


def _load_rse(path: Path) -> MotorFile:
    """RockSim: one ``engine`` element with ``eng-data`` samples; masses in grams."""
    engine = ET.parse(path).getroot().find(".//engine")
    if engine is None:
        raise ValueError(f"{path}: no engine in the file")
    pairs = [
        [float(d.get("t", "0")), float(d.get("f", "0"))]
        for d in engine.iter("eng-data")
    ]
    return MotorFile(
        name=engine.get("code", path.stem),
        thrust=_with_zero(pairs, path),
        propellant_kg=float(engine.get("propWt", "0")) / 1000 or None,
        total_kg=float(engine.get("initWt", "0")) / 1000 or None,
        diameter_mm=float(engine.get("dia", "0")),
        length_mm=float(engine.get("len", "0")),
        maker=engine.get("mfg", ""),
    )


def load_motor(path: str | Path) -> MotorFile:
    """Read an ``.eng`` or ``.rse`` file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".eng":
        return _load_eng(path)
    if suffix == ".rse":
        return _load_rse(path)
    raise ValueError(f"{path}: a motor file is .eng or .rse")


def eng_for_sim(path: str | Path, workdir: Path) -> Path:
    """The ``.eng`` the flight sim reads: the file, or a copy made from an ``.rse``."""
    path = Path(path)
    if path.suffix.lower() == ".eng":
        return path
    motor = load_motor(path)
    lines = [
        f"{motor.name} {motor.diameter_mm:g} {motor.length_mm:g} 0 "
        f"{motor.propellant_kg or 0:.6f} {motor.total_kg or 0:.6f} {motor.maker or 'x'}"
    ]
    lines += [f"{t:.6f} {f:.6f}" for t, f in motor.thrust]
    target = workdir / (path.stem + ".eng")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
