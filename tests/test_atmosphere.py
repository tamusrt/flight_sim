"""Atmosphere model tests."""

import numpy as np
import pytest

from flight_sim.environment.atmosphere import (
    AtmosphereConditions,
    StandardAtmosphere1976,
    VacuumAtmosphere,
)

_EARTH_RADIUS_M = 6356766.0

# Geopotential altitude in km with the temperature in K, pressure in Pa and,
# where given, density in kg/m**3 the US Standard Atmosphere 1976 tables list
_TABLE = [
    (0.0, 288.15, 101325.0, 1.2250),
    (11.0, 216.65, 22632.0, 0.36391),
    (20.0, 216.65, 5474.9, 0.088035),
    (32.0, 228.65, 868.02, 0.013225),
    (47.0, 270.65, 110.91, None),
    (51.0, 270.65, 66.939, None),
    (71.0, 214.65, 3.9564, None),
    (84.852, 186.946, 0.3734, None),
]


def _geometric_altitude(geopotential_m: float) -> float:
    """Return the geometric altitude in metres of a geopotential one."""
    return _EARTH_RADIUS_M * geopotential_m / (_EARTH_RADIUS_M - geopotential_m)


@pytest.mark.parametrize(("height_km", "temperature", "pressure", "density"), _TABLE)
def test_standard_atmosphere_matches_the_1976_tables(
    height_km: float, temperature: float, pressure: float, density: float | None
) -> None:
    """Every layer base agrees with the published temperature, pressure and density."""
    conditions = StandardAtmosphere1976().conditions(
        _geometric_altitude(height_km * 1000.0)
    )
    # The published values above 71 km carry fewer significant figures
    tolerance = 1e-3 if height_km > 71.0 else 1e-4

    assert conditions.temperature == pytest.approx(temperature, rel=tolerance)
    assert conditions.pressure == pytest.approx(pressure, rel=tolerance)
    if density is not None:
        assert conditions.air_density == pytest.approx(density, rel=tolerance)


def test_sea_level_speed_of_sound() -> None:
    """The sea-level speed of sound is the tabulated 340.29 m/s."""
    assert StandardAtmosphere1976().conditions(0.0).speed_of_sound == pytest.approx(
        340.29, rel=1e-4
    )


@pytest.mark.parametrize("height_km", [11.0, 20.0, 32.0, 47.0, 51.0, 71.0])
def test_layers_are_continuous(height_km: float) -> None:
    """Conditions just below and just above each layer boundary agree."""
    boundary = _geometric_altitude(height_km * 1000.0)
    below = StandardAtmosphere1976().conditions(boundary - 1e-3)
    above = StandardAtmosphere1976().conditions(boundary + 1e-3)

    for name in ("temperature", "pressure", "air_density", "speed_of_sound"):
        assert getattr(below, name) == pytest.approx(getattr(above, name), rel=1e-6)


def test_conditions_are_held_below_minus_five_kilometres() -> None:
    """Below -5 km the troposphere values at -5 km are held."""
    model = StandardAtmosphere1976()
    floor = model.conditions(-5000.0)

    assert floor.temperature > model.conditions(0.0).temperature
    assert floor.pressure > model.conditions(0.0).pressure
    assert model.conditions(-20000.0) == floor


def test_conditions_are_held_above_86_kilometres() -> None:
    """Above 86 km geometric the values at 86 km are held."""
    model = StandardAtmosphere1976()
    ceiling = model.conditions(86000.0)

    assert ceiling.pressure < model.conditions(80000.0).pressure
    assert model.conditions(200000.0) == ceiling


def test_wind_is_passed_through() -> None:
    """The configured wind is reported at every altitude."""
    wind = np.array([5.0, -3.0, 1.0])
    model = StandardAtmosphere1976(wind_m_s=wind)

    assert np.array_equal(model.conditions(0.0).wind, wind)
    assert np.array_equal(model.conditions(30000.0).wind, wind)
    assert np.array_equal(StandardAtmosphere1976().conditions(0.0).wind, np.zeros(3))


def test_vacuum_has_no_air_but_a_finite_speed_of_sound() -> None:
    """A vacuum reports zero density and pressure with a finite speed of sound."""
    conditions: AtmosphereConditions = VacuumAtmosphere().conditions(1000.0)

    assert conditions.air_density == 0.0
    assert conditions.pressure == 0.0
    assert conditions.speed_of_sound > 0.0
    assert np.array_equal(conditions.wind, np.zeros(3))
