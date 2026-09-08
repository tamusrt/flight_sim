"""Data structures defining physical vehicle properties"""

from dataclasses import dataclass, field

from scipy.interpolate import RegularGridInterpolator  # type: ignore

from flight_sim.units import Scalar, scalar
from flight_sim.utilities.data_loader import interpolator_from_csv


@dataclass
class RocketProperties:
    """Aerodynamic properties of the rocket"""

    aero_file_path: str

    reference_area: Scalar = field(
        default_factory=lambda: scalar(0.0182414692, ".=m**2")
    )
    reference_diameter: Scalar = field(default_factory=lambda: scalar(0.1524, ".m"))

    cd_table: RegularGridInterpolator = field(init=False)
    cl_table: RegularGridInterpolator = field(init=False)

    def __post_init__(self) -> None:
        self.cd_table = interpolator_from_csv(self.aero_file_path, "CD")
        self.cl_table = interpolator_from_csv(self.aero_file_path, "CL")
