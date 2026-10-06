# ruff: noqa: E501
# pylint: disable=line-too-long
"""A small synthetic OpenRocket file, aero CSV and thrust curve for the tests."""

import zipfile
from pathlib import Path

import numpy as np

_COLUMNS = [
    "Time",
    "Altitude",
    "Total velocity",
    "Mach number",
    "Total acceleration",
    "Mass",
    "Longitudinal moment of inertia",
    "Rotational moment of inertia",
    "Stability margin calibers",
    "Thrust",
    "Drag coefficient",
]

_ORK = """<?xml version="1.0" encoding="utf-8"?>
<openrocket version="1.10" creator="OpenRocket 24.12"><rocket>
 <name>Test</name>
 <subcomponents><stage><name>Sustainer</name><subcomponents>
  <nosecone><name>Nose</name><material density="1780.0" type="bulk">Carbon fiber</material>
   <length>0.5</length><thickness>0.002</thickness><shape>haack</shape><aftradius>0.05</aftradius>
   <overridemass>0.4</overridemass>
   <subcomponents><masscomponent><name>Payload</name><axialoffset method="top">0.1</axialoffset>
    <packedlength>0.1</packedlength><mass>1.0</mass></masscomponent></subcomponents></nosecone>
  <bodytube><name>Tube</name><material density="2700.0" type="bulk">Aluminum</material>
   <length>1.5</length><thickness>0.002</thickness><radius>0.05</radius>
   <subcomponents><trapezoidfinset><name>Fins</name><fincount>4</fincount>
    <axialoffset method="bottom">0.0</axialoffset><material density="1780.0" type="bulk">Carbon fiber</material>
    <thickness>0.004</thickness><crosssection>airfoil</crosssection><rootchord>0.25</rootchord>
    <tipchord>0.08</tipchord><height>0.12</height><sweeplength>0.2</sweeplength></trapezoidfinset>
    <bulkhead><name>Bulkhead</name><axialoffset method="top">0.0</axialoffset>
     <material density="1780.0" type="bulk">Carbon fiber</material><length>0.02</length>
     <outerradius>0.048</outerradius></bulkhead></subcomponents></bodytube>
  <transition><name>Tail</name><material density="2700.0" type="bulk">Aluminum</material>
   <length>0.06</length><thickness>0.002</thickness><shape>conical</shape>
   <foreradius>auto 0.05</foreradius><aftradius>0.035</aftradius></transition>
 </subcomponents></stage></subcomponents></rocket>
 <simulations><simulation><name>calm</name>
  <conditions><launchrodlength>5.0</launchrodlength><launchrodangle>5.0</launchrodangle>
   <launchroddirection>90.0</launchroddirection><windaverage>3.0</windaverage>
   <winddirection>1.5707963267948966</winddirection><launchaltitude>100.0</launchaltitude>
   <launchlatitude>32.99</launchlatitude><launchlongitude>-106.97</launchlongitude>
   <atmosphere model="extendedisa"><basetemperature>295.0</basetemperature>
    <basepressure>101325.0</basepressure></atmosphere></conditions>
  <flightdata maxaltitude="3000.0" maxvelocity="300.0" maxacceleration="60.0" maxmach="0.9"
   timetoapogee="20.0" launchrodvelocity="25.0"><databranch types="{types}">
{points}
  </databranch></flightdata></simulation></simulations></openrocket>
"""


def write_ork(path: Path) -> Path:
    """Write a small ``.ork`` zip; its motor weighs 10 kg, 4 kg of it propellant."""
    rows = []
    for i, t in enumerate(np.linspace(0.0, 20.0, 41)):
        thrust = 800.0 if t < 4.0 else 0.0
        mass = 10.0 + 6.3 - 4.0 * min(t / 4.0, 1.0)
        values = [
            t,
            50.0 * i,
            10.0 * i,
            0.03 * i,
            15.0,
            mass,
            12.0,
            0.02,
            3.0,
            thrust,
            0.5,
        ]
        rows.append(
            "   <datapoint>" + ",".join(f"{v:g}" for v in values) + "</datapoint>"
        )
    types = ",".join(_COLUMNS)
    text = _ORK.format(types=types, points="\n".join(rows))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rocket.ork", text)
    return path


def write_aero_csv(path: Path) -> Path:
    """A RASAero-style CSV at roll angles 0 and 90 only, with plausible values."""
    lines = ["Mach,Alpha,Phi,Cx,Cy,Cz,CMx,CMy,CMz"]
    for mach in (0.1, 0.5, 0.9, 1.1, 1.5, 2.0, 3.0):
        for alpha in range(31):
            cn = 0.2 * alpha
            cx = -(0.45 + 0.002 * alpha)
            for phi in (0, 90):
                cz = -cn if phi == 0 else 0.0
                cy = 0.0 if phi == 0 else -cn
                cmy = cz * 1.9 / 0.1
                cmz = -cy * 1.9 / 0.1
                lines.append(f"{mach},{alpha},{phi},{cx},{cy},{cz},0,{cmy},{cmz}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_eng(path: Path) -> Path:
    """A flat 800 N, 4 s curve for a 10 kg motor with 4 kg of propellant."""
    path.write_text(
        "T 100 1000 0 4.0 10.0 X\n0.0 0\n0.01 800\n3.99 800\n4.0 0\n", encoding="utf-8"
    )
    return path


def write_cdx(path: Path) -> Path:
    """A RASAero ``.CDX1`` with the fixture's rocket, but bigger fins and a longer tube."""
    path.write_text(
        """<RASAeroDocument><FileVersion>2</FileVersion><RocketDesign>
  <NoseCone><Length>19.685</Length><Diameter>3.937</Diameter></NoseCone>
  <BodyTube><Length>20</Length><Diameter>3.937</Diameter></BodyTube>
  <BodyTube><Length>40</Length><Diameter>3.937</Diameter>
    <Fin><Count>3</Count><Chord>12</Chord><Span>5</Span><SweepDistance>8</SweepDistance>
     <TipChord>4</TipChord><Thickness>0.2</Thickness></Fin></BodyTube>
  <BoatTail><Length>2.4</Length><Diameter>3.937</Diameter><RearDiameter>2.8</RearDiameter></BoatTail>
</RocketDesign></RASAeroDocument>
""",
        encoding="utf-8",
    )
    return path


def write_rse(path: Path) -> Path:
    """The same flat 800 N, 4 s motor as ``write_eng`` but as a RockSim file, in grams."""
    path.write_text(
        """<engine-database><engine-list>
<engine code="TEST-RSE" mfg="Team" dia="100" len="900" delays="0" initWt="9000" propWt="3500">
<data><eng-data f="0" t="0"/><eng-data f="800" t="0.01"/><eng-data f="800" t="3.99"/>
<eng-data f="0" t="4"/></data></engine></engine-list></engine-database>
""",
        encoding="utf-8",
    )
    return path
