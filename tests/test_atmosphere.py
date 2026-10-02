"""Atmosphere model tests."""

import numpy as np
import pytest

from flight_sim.environment.atmosphere import (
    AtmosphereConditions,
    LaunchSiteAtmosphere,
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


def test_vacuum_has_no_air_but_a_finite_speed_of_sound() -> None:
    """A vacuum reports zero density and pressure with a finite speed of sound."""
    conditions: AtmosphereConditions = VacuumAtmosphere().conditions(1000.0)

    assert conditions.air_density == 0.0
    assert conditions.pressure == 0.0
    assert conditions.speed_of_sound > 0.0


def test_launch_site_atmosphere_defaults_to_the_standard() -> None:
    """Sea-level standard pad values reproduce the 1976 model below 30 km."""
    site = LaunchSiteAtmosphere()
    standard = StandardAtmosphere1976()
    for altitude in (0.0, 3000.0, 10000.0, 15000.0, 30000.0):
        assert site.conditions(altitude).air_density == pytest.approx(
            standard.conditions(altitude).air_density, rel=1e-9
        )


def test_launch_site_atmosphere_matches_openrocket_extended_isa() -> None:
    """A hot elevated pad reproduces OpenRocket's own conditions for it.

    OpenRocket, with the pad at 890 m, 303.15 K and 91432.8 Pa, reports
    1.051 kg/m**3 at the pad and 248.0 K with 0.576 kg/m**3 at 6445 m above it.
    """
    site = LaunchSiteAtmosphere(
        pad_elevation_m=890.016, pad_temperature_k=303.15, pad_pressure_pa=91432.755
    )
    pad = site.conditions(890.016)
    assert pad.temperature == pytest.approx(303.15)
    assert pad.pressure == pytest.approx(91432.755)
    assert pad.air_density == pytest.approx(1.051, abs=5e-4)
    high = site.conditions(890.016 + 6445.0)
    assert high.temperature == pytest.approx(248.0, abs=0.1)
    assert high.air_density == pytest.approx(0.576, abs=1e-3)


def test_launch_site_atmosphere_is_continuous_at_the_tropopause() -> None:
    """Pressure and temperature join the standard stratosphere smoothly."""
    site = LaunchSiteAtmosphere(
        pad_elevation_m=1400.0, pad_temperature_k=300.0, pad_pressure_pa=86000.0
    )
    tropopause = _geometric_altitude(11000.0)
    below = site.conditions(tropopause - 0.01)
    above = site.conditions(tropopause + 0.01)
    assert above.temperature == pytest.approx(below.temperature, abs=1e-3)
    assert above.pressure == pytest.approx(below.pressure, rel=1e-5)


def test_launch_site_atmosphere_handles_an_isothermal_column() -> None:
    """A pad already at the tropopause temperature gives an isothermal layer."""
    site = LaunchSiteAtmosphere(pad_temperature_k=216.65)
    conditions = site.conditions(1000.0)
    expected = 101325.0 * np.exp(-9.80665 * 999.84 / (287.0531 * 216.65))
    assert conditions.temperature == pytest.approx(216.65)
    assert conditions.pressure == pytest.approx(expected, rel=1e-4)
