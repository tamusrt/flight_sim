from dataclasses import dataclass, field
from flight_sim.units import scalar, Scalar
from scipy.interpolate import RegularGridInterpolator
from flight_sim.utilities.data_loader import interpolator_from_csv

@dataclass
class RocketProperties:
    """Aerodynamic properties of the rocket"""

    reference_area: Scalar = scalar(0.0182414692, ".=m**2")
    reference_diameter: Scalar = scalar(0.1524, ".m")

    aero_file_path: str

    cd_table: RegularGridInterpolator = field(init = False)
    cl_table: RegularGridInterpolator = field(init = False)

    def __post_init__(self):
        self.cd_table = interpolator_from_csv(self.aero_file_path, "CD")
        self.cl_table = interpolator_from_csv(self.aero_file_path, "CL")