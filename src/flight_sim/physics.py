"""Forces on the rocket, which a flight phase swaps by subclassing.

A mission segment picks its physics with ``config.physics = { type = "..." }``.
Methods take and return SI numbers.
"""

import math
from dataclasses import dataclass

import numpy as np

from flight_sim.environment.atmosphere import AtmosphereConditions
from flight_sim.flight_point import AeroPointSI, FlightPointSI
from flight_sim.utilities.data_loader import AeroCoefficients, AeroTable
from flight_sim.utilities.dcm import aero_angles
from flight_sim.utilities.quaternion import cross
from flight_sim.vehicle.rocket_properties import RocketProperties


@dataclass(frozen=True)
class RocketPhysics:
    """Motor thrust along the nose and aerodynamic loads from the aero table."""

    def thrust(
        self, time: float, conditions: AtmosphereConditions, properties: RocketProperties
    ) -> float:
        """Return the thrust along body +X in N.

        Args:
            time (float): Simulation time in seconds.
            conditions (AtmosphereConditions): Air conditions around the rocket.
            properties (RocketProperties): Aerodynamic and motor properties.
        """
        return properties.engine.get_thrust(time, conditions.pressure)

    def after(self, event: str, point: FlightPointSI) -> "RocketPhysics":
        """Return the physics to fly once an event has fired.

        The sim calls this with each event as it fires, and with the events of
        earlier segments at the start, in firing order. The default ignores
        events.

        Args:
            event (str): Name of the event.
            point (FlightPointSI): The flight when it fired: time, state and
                every load.
        """
        del event, point
        return self

    def aero(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        time: float,  # pylint: disable=unused-argument
        air_velocity: np.ndarray,
        conditions: AtmosphereConditions,
        to_body: np.ndarray,
        cg_location: np.ndarray,
        properties: RocketProperties,
        thrust: float,  # pylint: disable=unused-argument
    ) -> AeroPointSI:
        """Compute the flow angles and the aerodynamic loads, in body axes.

        The torque is the table's moment about the reference point plus the
        moment of the force through the lever arm from the CG.

        Args:
            time (float): Simulation time in seconds.
            air_velocity (np.ndarray): Velocity relative to the wind in the world
                frame in m/s.
            conditions (AtmosphereConditions): Air conditions around the rocket.
            to_body (np.ndarray): World-to-body rotation matrix.
            cg_location (np.ndarray): CG from the nose tip in body axes in m.
            properties (RocketProperties): Aerodynamic and motor properties.
            thrust (float): Motor thrust in N, for power-on effects.

        Returns:
            AeroPointSI: Force in N and torque about the CG in N*m, with the flow
                angles and coefficients they came from.
        """
        table = properties.aero_table
        airspeed_body = to_body @ air_velocity
        speed = math.sqrt(float(airspeed_body @ airspeed_body))
        alpha, phi = aero_angles(airspeed_body)
        mach = speed / conditions.speed_of_sound
        c = table(mach, math.degrees(alpha), math.degrees(phi))
        dynamic_pressure = 0.5 * conditions.air_density * speed**2
        force_body, torque_body = coefficient_loads(
            c, dynamic_pressure, table, cg_location
        )
        return AeroPointSI(
            air_velocity=airspeed_body,
            airspeed=speed,
            mach=mach,
            alpha=alpha,
            phi=phi,
            dynamic_pressure=dynamic_pressure,
            coefficients=c,
            force=force_body,
            torque=torque_body,
        )


def coefficient_loads(
    c: AeroCoefficients,
    dynamic_pressure: float,
    table: AeroTable,
    cg_location: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the body-axes force in N and torque about the CG in N*m of coefficients.

    The torque is the table's moment about the reference point plus the
    moment of the force through the lever arm from the CG.
    """
    # Dynamic pressure times reference area gives force per unit coefficient
    force_scale = dynamic_pressure * table.reference_area_m2
    force = force_scale * np.array([c.cx, c.cy, c.cz])
    # Reference length acts as "lever arm" length
    moment = (force_scale * table.reference_length_m) * np.array([c.cmx, c.cmy, c.cmz])
    return force, moment + cross(table.reference_point_m - cg_location, force)
