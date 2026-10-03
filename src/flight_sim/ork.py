"""Read an OpenRocket ``.ork`` file: geometry, masses and the sims saved in it.

An ``.ork`` is a zip holding one XML document. This module reads what the
flight sim and the predictions page need and nothing else:

* the outer shape (nose, body tubes, transition, trapezoidal fins), as
  absolute distances from the nose tip;
* the mass of every component, using OpenRocket's own overrides where the
  designer set one, and a shell or plate of the component's material otherwise;
* the simulations saved in the file: launch conditions, headline results and
  the time series (mass, CG, inertia, thrust, ...) that OpenRocket stored.

Everything is in SI units, with x measured from the nose tip, positive aft,
unlike the flight sim's body axes (nose forward, so negative aft).
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

# Densities in kg/m**3 of OpenRocket's stock bulk materials; the file's own
# <material density="..."> wins over these when it carries one.
_DEFAULT_DENSITY = {"Carbon fiber": 1780.0, "Fiberglass": 1850.0, "Aluminum": 2700.0}

_BODY_TAGS = ("nosecone", "bodytube", "transition")
_INTERNAL_TAGS = ("bulkhead", "tubecoupler", "centeringring", "innertube")


@dataclass(frozen=True)
class Fins:
    """One set of identical trapezoidal fins, as OpenRocket stores them."""

    count: int
    root_chord_m: float
    tip_chord_m: float
    span_m: float
    sweep_m: float
    thickness_m: float
    root_trailing_edge_x_m: float
    density_kg_m3: float
    airfoil: str

    @property
    def planform_area_m2(self) -> float:
        """Area of one fin."""
        return 0.5 * (self.root_chord_m + self.tip_chord_m) * self.span_m

    @property
    def mass_kg(self) -> float:
        """Mass of the whole set, as a plate of the fin material."""
        return (
            self.count * self.planform_area_m2 * self.thickness_m * self.density_kg_m3
        )

    @property
    def leading_edge_x_m(self) -> float:
        """Distance from the nose tip to the root leading edge."""
        return self.root_trailing_edge_x_m - self.root_chord_m


@dataclass(frozen=True)
class MassItem:
    """One lump of mass: its name, kind, mass and station."""

    name: str
    kind: str
    mass_kg: float
    x_m: float
    overridden: bool = False
    length_m: float = 0.0


@dataclass(frozen=True)
class SavedSim:
    """One simulation stored in the file.

    Attributes:
        name (str): The simulation's name in OpenRocket.
        conditions (dict[str, float]): Launch rod, wind and site settings.
        summary (dict[str, float]): OpenRocket's headline results.
        series (dict[str, np.ndarray]): Stored time series by variable name.
    """

    name: str
    conditions: dict[str, float]
    summary: dict[str, float]
    series: dict[str, np.ndarray] = field(repr=False)


@dataclass(frozen=True)
class OrkRocket:  # pylint: disable=too-many-instance-attributes
    """The parts of an OpenRocket design that the sim uses."""

    name: str
    reference_diameter_m: float
    length_m: float
    nose_length_m: float
    nose_shape: str
    nose_mass_kg: float
    tail_length_m: float
    tail_fore_radius_m: float
    tail_aft_radius_m: float
    tail_wall_m: float
    tail_density_kg_m3: float
    body_start_x_m: float
    body_length_m: float
    body_wall_m: float
    body_density_kg_m3: float
    fins: Fins
    items: tuple[MassItem, ...]
    sims: dict[str, SavedSim]

    @property
    def reference_area_m2(self) -> float:
        """Cross-section of the body tube."""
        return math.pi * (0.5 * self.reference_diameter_m) ** 2

    @property
    def airframe_mass_kg(self) -> float:
        """Mass of every listed component, not counting the motor."""
        return sum(item.mass_kg for item in self.items)


def _kids(element: ET.Element, tag: str = "subcomponents") -> list[ET.Element]:
    """Children of a container element, or none when it has no such container."""
    node = element.find(tag)
    return [] if node is None else list(node)


def _text(element: ET.Element, tag: str, default: float = 0.0) -> float:
    """A child's number; "auto 0.07" style values give the number after "auto"."""
    value = element.findtext(tag)
    if value is None:
        return default
    try:
        return float(value.strip().split()[-1])
    except ValueError:
        return default


def _density(element: ET.Element) -> float:
    """Density of a component's material in kg/m**3."""
    material = element.find("material")
    if material is None:
        return _DEFAULT_DENSITY["Carbon fiber"]
    if "density" in material.attrib:
        return float(material.attrib["density"])
    return _DEFAULT_DENSITY.get((material.text or "").strip(), 1780.0)


def _offset(element: ET.Element, parent_x: float, parent_length: float) -> float:
    """Distance from the nose tip to a component's forward end."""
    node = element.find("axialoffset")
    if node is None:
        return parent_x
    value = float(node.text or 0.0)
    method = node.attrib.get("method", "top")
    if method == "absolute":
        return value
    if method == "bottom":
        return parent_x + parent_length - _own_length(element) + value
    if method == "middle":
        return parent_x + 0.5 * (parent_length - _own_length(element)) + value
    return parent_x + value


def _own_length(element: ET.Element) -> float:
    """Axial length of a component, for placing it from its aft end."""
    if element.tag == "trapezoidfinset":
        return _text(element, "rootchord")
    if element.tag == "masscomponent":
        return _text(element, "packedlength")
    return _text(element, "length")


def _shell_mass(tag: str, element: ET.Element) -> float:
    """Mass of a component with no override: its material, as OpenRocket weighs it."""
    rho = _density(element)
    length = _text(element, "length")
    if tag in ("bodytube", "tubecoupler"):
        outer = _text(element, "radius") or _text(element, "outerradius")
        wall = _text(element, "thickness")
        return rho * math.pi * (outer**2 - (outer - wall) ** 2) * length
    if tag == "bulkhead":
        return rho * math.pi * _text(element, "outerradius", 0.0742) ** 2 * length
    return 0.0


def _component_mass(tag: str, element: ET.Element) -> tuple[float, bool]:
    """Mass of one component and whether the designer overrode it."""
    override = element.findtext("overridemass")
    if override is not None:
        return float(override), True
    if tag == "masscomponent":
        return _text(element, "mass"), False
    return _shell_mass(tag, element), False


def _fins_from(element: ET.Element, x_parent: float, length: float) -> Fins:
    """Read a trapezoidal fin set placed on a body tube."""
    root = _text(element, "rootchord")
    start = _offset(element, x_parent, length)
    return Fins(
        count=int(_text(element, "fincount", 4)),
        root_chord_m=root,
        tip_chord_m=_text(element, "tipchord"),
        span_m=_text(element, "height"),
        sweep_m=_text(element, "sweeplength"),
        thickness_m=_text(element, "thickness"),
        root_trailing_edge_x_m=start + root,
        density_kg_m3=_density(element),
        airfoil=(element.findtext("crosssection") or "").strip(),
    )


def _saved_sims(root: ET.Element) -> dict[str, SavedSim]:
    """Read every simulation with stored flight data."""
    sims: dict[str, SavedSim] = {}
    for sim in root.iter("simulation"):
        data = sim.find("flightdata")
        branch = None if data is None else data.find("databranch")
        if data is None or branch is None:
            continue
        conditions = {
            child.tag: float(child.text)
            for child in _kids(sim, "conditions")
            if child.text and child.text.strip() and _is_number(child.text)
        }
        atmosphere = sim.find("conditions/atmosphere")
        for child in [] if atmosphere is None else list(atmosphere):
            if child.text and _is_number(child.text):
                conditions[child.tag] = float(child.text)
        summary = {key: float(value) for key, value in data.attrib.items()}
        names = branch.attrib["types"].split(",")
        rows = [
            [float(v) for v in point.text.split(",")]
            for point in branch.iter("datapoint")
            if point.text
        ]
        table = np.array(rows, dtype=float)
        series = {name: table[:, i] for i, name in enumerate(names)}
        label = (sim.findtext("name") or "simulation").strip()
        sims[label] = SavedSim(label, conditions, summary, series)
    return sims


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def load_ork(path: str | Path) -> OrkRocket:  # pylint: disable=too-many-locals
    """Read the design in an ``.ork`` file.

    Args:
        path (str | Path): The file.

    Returns:
        OrkRocket: Its geometry, component masses and saved simulations.
    """
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".ork"))
        root = ET.fromstring(archive.read(name))
    rocket = root.find("rocket")
    if rocket is None:
        raise ValueError(f"{path}: no rocket in the file")

    items: list[MassItem] = []
    fins: Fins | None = None
    nose: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    tubes: list[tuple[float, float, float, float]] = []
    station = 0.0
    radius = 0.0
    for stage in rocket.iter("stage"):
        for part in _kids(stage):
            if part.tag not in _BODY_TAGS:
                continue
            length = _text(part, "length")
            start = station
            if part.tag == "nosecone":
                nose = {
                    "length": length,
                    "shape": (part.findtext("shape") or "").strip(),
                    "mass": _component_mass("nosecone", part)[0],
                }
                radius = _text(part, "aftradius")
            elif part.tag == "bodytube":
                radius = _text(part, "radius") or radius
                tubes.append((start, length, _text(part, "thickness"), _density(part)))
            else:
                tail = {
                    "length": length,
                    "fore": _text(part, "foreradius") or radius,
                    "aft": _text(part, "aftradius"),
                    "wall": _text(part, "thickness"),
                    "density": _density(part),
                }
            mass, overridden = _component_mass(part.tag, part)
            items.append(
                MassItem(
                    part.findtext("name") or part.tag,
                    part.tag,
                    mass,
                    start + 0.5 * length,
                    overridden,
                    length if part.tag != "nosecone" else 0.0,
                )
            )
            for child in _kids(part):
                if child.tag == "trapezoidfinset":
                    fins = _fins_from(child, start, length)
                    items.append(
                        MassItem(
                            child.findtext("name") or "fins",
                            "fins",
                            fins.mass_kg,
                            fins.leading_edge_x_m + 0.4 * fins.root_chord_m,
                        )
                    )
                    continue
                if child.tag not in (*_INTERNAL_TAGS, "masscomponent"):
                    continue
                mass, overridden = _component_mass(child.tag, child)
                first = _offset(child, start, length)
                items.append(
                    MassItem(
                        child.findtext("name") or child.tag,
                        child.tag,
                        mass,
                        first + 0.5 * _own_length(child),
                        overridden,
                        _own_length(child) if child.tag != "masscomponent" else 0.0,
                    )
                )
            station += length
    if fins is None:
        raise ValueError(f"{path}: no trapezoidal fin set found")
    # A nose cone's own shell is listed once by the loop above (with the
    # overridden mass); the tubes are the cylinder that body length changes grow.
    body = [t for t in tubes if t[1] > 0.0]
    biggest = max(body, key=lambda t: t[1])
    return OrkRocket(
        name=(rocket.findtext("name") or "rocket").strip(),
        reference_diameter_m=2.0 * radius,
        length_m=station,
        nose_length_m=nose["length"],
        nose_shape=nose["shape"],
        nose_mass_kg=nose["mass"],
        tail_length_m=tail["length"],
        tail_fore_radius_m=tail["fore"],
        tail_aft_radius_m=tail["aft"],
        tail_wall_m=tail["wall"],
        tail_density_kg_m3=tail["density"],
        body_start_x_m=biggest[0],
        body_length_m=biggest[1],
        body_wall_m=biggest[2],
        body_density_kg_m3=biggest[3],
        fins=fins,
        items=tuple(items),
        sims=_saved_sims(root),
    )


@dataclass(frozen=True)
class Motor:
    """The motor as the mass it adds and where: the sim takes thrust from a file."""

    mass_kg: float
    propellant_kg: float
    x_m: float
    length_m: float


def saved_motor(ork: OrkRocket, sim: str | None = None) -> Motor:
    """The motor OpenRocket flew in a saved sim, inferred from its mass history.

    The saved liftoff mass less the airframe is the motor with its propellant;
    the drop to burnout is the propellant. The motor sits at the middle of the
    longest body tube, which is where OpenRocket put it.
    """
    saved = _pick_sim(ork, sim)
    mass = saved.series["Mass"]
    return Motor(
        mass_kg=float(mass[0]) - ork.airframe_mass_kg,
        propellant_kg=float(mass[0] - mass[-1]),
        x_m=ork.body_start_x_m + 0.5 * ork.body_length_m,
        length_m=ork.body_length_m,
    )


def _pick_sim(ork: OrkRocket, sim: str | None) -> SavedSim:
    """A saved sim by name, or the first one."""
    if not ork.sims:
        raise ValueError("the .ork file has no saved simulations")
    if sim is None:
        return next(iter(ork.sims.values()))
    if sim not in ork.sims:
        raise KeyError(f"no simulation {sim!r}; the file has {sorted(ork.sims)}")
    return ork.sims[sim]


def _lumps(
    ork: OrkRocket, motor: Motor, burned: float
) -> list[tuple[float, float, float]]:
    """(mass, station, length) of every lump, with a share of the propellant gone."""
    lumps = [(i.mass_kg, i.x_m, i.length_m) for i in ork.items]
    lumps.append(
        (motor.mass_kg - burned * motor.propellant_kg, motor.x_m, motor.length_m)
    )
    return lumps


def mass_properties(
    ork: OrkRocket, motor: Motor, burned: float
) -> tuple[float, float, float]:
    """Mass, CG station and pitch inertia at a share of the propellant burned.

    Args:
        ork (OrkRocket): The design.
        motor (Motor): The motor mass and station.
        burned (float): Fraction of the propellant gone, 0 to 1.

    Returns:
        tuple[float, float, float]: Mass in kg, CG in m from the nose tip, and
            pitch inertia about the CG in kg*m**2.
    """
    lumps = _lumps(ork, motor, burned)
    total = sum(m for m, _, _ in lumps)
    cg = sum(m * x for m, x, _ in lumps) / total
    inertia = sum(m * ((x - cg) ** 2 + length**2 / 12.0) for m, x, length in lumps)
    return total, cg, inertia
