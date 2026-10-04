"""The 6-DOF climb from the rail to apogee, with what the IREC checks need."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from flight_sim.descent import air_at
from flight_sim.events import APOGEE
from flight_sim.flight_computer import Sample
from flight_sim.integration import IntegrationConfiguration, adaptive_step
from flight_sim.units import scalar
from flight_sim.vehicle.rocket_properties import RocketProperties
from flight_sim.vehicle.rocket_state import RocketState
from flight_sim.visualize import static_centre_of_pressure

# Stability is counted above this speed, as on the History page
FAST_M_S = 30.0
_SAMPLE_S = 0.1
_MAX_TIME_S = 400.0


@dataclass
class Ascent:  # pylint: disable=too-many-instance-attributes
    """The climb of one flight.

    Attributes:
        samples (list[Sample]): Time and state every 0.1 s, ending at apogee
            (when ``reached_apogee``).
        reached_apogee (bool): Whether the climb ended at apogee.
        rail_speed_m_s (float | None): Speed as the rocket leaves the rail.
        rail_margin_cal (float | None): Stability margin there, in calibres.
        lowest_margin_cal (float | None): Lowest margin from the rail to apogee
            while faster than 30 m/s.
        highest_margin_cal (float | None): The highest of the same.
        max_aoa_deg (float): Largest angle of attack above 30 m/s (not used by
            the checks, but a good sign of a bad launch).
        burnout_s (float | None): Time the motor stopped.
        margin_track (list[tuple[float, float]]): Time and stability margin
            (calibres) at each sample faster than 30 m/s, for the page's chart.
    """

    samples: list[Sample] = field(default_factory=list)
    reached_apogee: bool = False
    rail_speed_m_s: float | None = None
    rail_margin_cal: float | None = None
    lowest_margin_cal: float | None = None
    highest_margin_cal: float | None = None
    burnout_s: float | None = None
    margin_track: list[tuple[float, float]] = field(default_factory=list)

    @property
    def apogee_time_s(self) -> float:
        """Time of the last sample."""
        return self.samples[-1][0]

    @property
    def apogee_state(self) -> RocketState:
        """State of the last sample."""
        return self.samples[-1][1]

    @property
    def apogee_m(self) -> float:
        """Height of the last sample above the pad."""
        return float(self.samples[-1][1].position.m_as("m")[0])


def margin_cal(
    properties: RocketProperties,
    config: IntegrationConfiguration,
    state: RocketState,
    time_s: float,
) -> float:
    """Stability margin in calibres: (centre of pressure - CG) over the diameter."""
    height = float(state.position.m_as("m")[0])
    air, wind = air_at(config, height)
    speed = float(np.linalg.norm(state.velocity.m_as("m/s") - wind))
    cp = static_centre_of_pressure(properties, speed / air.speed_of_sound)
    cg = -float(properties.mass_properties(time_s).cg_location[0])
    return (cp - cg) / float(properties.aero_table.reference_length_m)


def fly_ascent(
    properties: RocketProperties,
    state: RocketState,
    config: IntegrationConfiguration,
) -> Ascent:
    """Fly the 6-DOF climb to apogee, recording the rail exit and the stability."""
    ascent = Ascent()
    margins: list[float] = []
    time, next_sample = 0.0, 0.0
    dt = scalar(0.01, "s")
    while time <= _MAX_TIME_S:
        if time >= next_sample - 1e-9:
            ascent.samples.append((time, state))
            while next_sample <= time + 1e-9:
                next_sample += _SAMPLE_S
            if float(np.linalg.norm(state.velocity.m_as("m/s"))) > FAST_M_S:
                margin = margin_cal(properties, config, state, time)
                if not math.isnan(margin):
                    margins.append(margin)
                    ascent.margin_track.append((time, margin))
        state, taken, dt, hit = adaptive_step(
            time, state, properties, config, dt, events=(APOGEE,)
        )
        time += float(taken.m_as("s"))
        if (
            ascent.burnout_s is None
            and time > 0.5
            and properties.engine.get_thrust(time) <= 0.0
        ):
            ascent.burnout_s = time
        if hit is not None and hit.name == "rail exit":
            ascent.rail_speed_m_s = float(np.linalg.norm(state.velocity.m_as("m/s")))
            ascent.rail_margin_cal = margin_cal(properties, config, state, time)
        if hit is APOGEE:
            ascent.samples.append((time, state))
            ascent.reached_apogee = True
            break
    if margins:
        ascent.lowest_margin_cal = min(margins)
        ascent.highest_margin_cal = max(margins)
    return ascent
