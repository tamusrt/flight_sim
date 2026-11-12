"""Outer shape of the rocket, for the models the aero table leaves out"""

import math
from dataclasses import dataclass, field
from typing import Annotated

from flight_sim.units import Scalar, UnitChecked


@dataclass(frozen=True)
class RocketShape(UnitChecked):
    """Length and fins of the rocket, such as for FS1.0's pitch damping."""

    # Nose tip to the aft end of the body
    length: Annotated[Scalar, "m"]

    num_fins: int

    # Planform area of one fin
    fin_area: Annotated[Scalar, "m**2"]

    # Centroid of a fin's planform along body X from the nose tip, so aft is
    # negative, as for cg_location
    fin_pos: Annotated[Scalar, "m"]

    # The shape in SI units, cached on construction
    length_m: float = field(init=False, repr=False, compare=False)
    fin_area_m2: float = field(init=False, repr=False, compare=False)
    fin_pos_m: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the shape and cache it in SI units.

        Raises:
            ValueError: If the length is not positive, the fin count or area
                is negative, or the fins' centroid lies off the body.
        """
        super().__post_init__()
        length_m = float(self.length.m_as("m"))
        fin_area_m2 = float(self.fin_area.m_as("m**2"))
        fin_pos_m = float(self.fin_pos.m_as("m"))
        if not (math.isfinite(length_m) and length_m > 0.0):
            raise ValueError(f"length must be finite and positive, got {length_m} m")
        if self.num_fins < 0:
            raise ValueError(f"num_fins must be 0 or more, got {self.num_fins}")
        if not (math.isfinite(fin_area_m2) and fin_area_m2 >= 0.0):
            raise ValueError(
                f"fin_area must be finite and 0 or more, got {fin_area_m2} m**2"
            )
        if not -length_m <= fin_pos_m <= 0.0:
            raise ValueError(
                f"fin_pos must lie on the body, from 0 m at the nose tip to"
                f" -{length_m} m at its aft end (aft is -X), got {fin_pos_m} m"
            )
        object.__setattr__(self, "length_m", length_m)
        object.__setattr__(self, "fin_area_m2", fin_area_m2)
        object.__setattr__(self, "fin_pos_m", fin_pos_m)
