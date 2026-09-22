"""Data structures defining physical vehicle properties"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d  # type: ignore

from flight_sim.units import Scalar, Vector, scalar, vector
from flight_sim.utilities.data_loader import (
    interpolator_from_csv,
    time_interpolator_from_csv,
)


@dataclass
class RocketProperties:
    """Aerodynamic properties of the rocket"""

    # pylint: disable=too-many-instance-attributes

    aero_file_path: str
    motor_file_path: str
    propellant_mass: float  # Total weight of solid fuel in kg

    reference_area: Scalar = field(
        default_factory=lambda: scalar(0.0182414692, ".=m**2")
    )
    reference_diameter: Scalar = field(default_factory=lambda: scalar(0.1524, ".m"))

    reference_point: Vector = field(
        default_factory=lambda: vector((0.0, 0.0, 0.0), "m")
    )

    cd_table: RegularGridInterpolator = field(init=False)  # Drag
    cl_table: RegularGridInterpolator = field(init=False)  # Lift (Normal)
    cy_table: RegularGridInterpolator = field(init=False)  # Side Force

    c_roll_table: RegularGridInterpolator = field(init=False)  # Roll
    cm_table: RegularGridInterpolator = field(init=False)  # Pitch
    cn_table: RegularGridInterpolator = field(init=False)  # Yaw

    thrust_curve: interp1d = field(init=False)
    mass_flow_multiplier: float = field(init=False)

    def __post_init__(self) -> None:
        self.cd_table = interpolator_from_csv(self.aero_file_path, "CD")
        self.cl_table = interpolator_from_csv(self.aero_file_path, "CL")
        self.cm_table = interpolator_from_csv(self.aero_file_path, "CM")

        self.cy_table = interpolator_from_csv(self.aero_file_path, "CY")
        self.c_roll_table = interpolator_from_csv(self.aero_file_path, "C_roll")
        self.cn_table = interpolator_from_csv(self.aero_file_path, "CN")

        self.thrust_curve = time_interpolator_from_csv(
            self.motor_file_path, "Time", "Thrust"
        )
        self.mass_curve = time_interpolator_from_csv(
            self.motor_file_path, "Time", "Mass"
        )

        motor_data = np.genfromtxt(self.motor_file_path, delimiter=",", names=True)
        time_vals = motor_data["Time"]
        thrust_vals = motor_data["Thrust"]

        total_impulse = float(np.trapezoid(thrust_vals, time_vals))

        if total_impulse > 0.0:
            self.mass_flow_multiplier = self.propellant_mass / total_impulse
        else:
            self.mass_flow_multiplier = 0.0
