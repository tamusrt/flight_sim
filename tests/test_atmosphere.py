"""Atmosphere model tests."""

import pytest
from support import truth

from flight_sim.environment.atmosphere import (
    AtmosphereConditions,
    StandardAtmosphere1976,
    VacuumAtmosphere,
)
from flight_sim.units import scalar

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


def test_standard_day_launch_site_is_the_standard_atmosphere() -> None:
    """Sea level at 59 deg F and 29.92 inHg reproduces the default atmosphere."""
    site = StandardAtmosphere1976(
        launch_temperature=scalar(59.0, "degF"),
        barometric_pressure=scalar(101325.0, "Pa"),
    )

    for altitude in (0.0, 1500.0, 9000.0, 25000.0):
        assert site.conditions(altitude) == pytest.approx(
            StandardAtmosphere1976().conditions(altitude)
        )


def test_launch_site_anchors_each_quantity_at_the_site() -> None:
    """Site temperature is as given, site pressure is the barometric one reduced."""
    elevation = scalar(2950.0, "ft").m_as("m")
    site = StandardAtmosphere1976(
        launch_temperature=scalar(76.0, "degF"),
        barometric_pressure=scalar(27.0, "inHg"),
    )
    site.anchor(elevation)
    standard = StandardAtmosphere1976()
    at_site = site.conditions(elevation)
    standard_site = standard.conditions(elevation)

    assert at_site.temperature == pytest.approx(scalar(76.0, "degF").m_as("K"))
    assert at_site.pressure == pytest.approx(
        scalar(27.0, "inHg").m_as("Pa") * standard_site.pressure / 101325.0
    )
    assert at_site.air_density == pytest.approx(
        at_site.pressure / (287.0531 * at_site.temperature), rel=1e-4
    )
    # Above the site, each quantity keeps the standard variation with altitude
    above, standard_above = site.conditions(5000.0), standard.conditions(5000.0)
    for name in ("temperature", "pressure", "air_density"):
        assert getattr(above, name) / getattr(at_site, name) == pytest.approx(
            getattr(standard_above, name) / getattr(standard_site, name)
        )


def test_launch_temperature_defaults_to_the_standard_day() -> None:
    """Without a site temperature, only the barometric pressure moves the air."""
    site = StandardAtmosphere1976(barometric_pressure=scalar(27.0, "inHg"))
    site.anchor(scalar(2950.0, "ft").m_as("m"))
    standard = StandardAtmosphere1976().conditions(3000.0)

    assert site.conditions(3000.0).temperature == pytest.approx(standard.temperature)
    assert site.conditions(3000.0).pressure == pytest.approx(
        standard.pressure * scalar(27.0, "inHg").m_as("Pa") / 101325.0
    )


def test_humidity_lowers_the_density_only() -> None:
    """Moist air at 30 C is about 1.6% lighter at full humidity, at the same T and p."""
    dry = StandardAtmosphere1976(launch_temperature=scalar(30.0, "degC"))
    humid = StandardAtmosphere1976(
        launch_temperature=scalar(30.0, "degC"), relative_humidity=1.0
    )

    # The vapour pressure of 4246 Pa thins tenfold every 6.3 km
    for altitude, vapour_pressure in ((0.0, 4246.0), (6300.0, 424.6)):
        vapour_share = vapour_pressure / dry.conditions(altitude).pressure
        assert (
            humid.conditions(altitude).temperature
            == dry.conditions(altitude).temperature
        )
        assert humid.conditions(altitude).pressure == dry.conditions(altitude).pressure
        assert humid.conditions(altitude).air_density / dry.conditions(
            altitude
        ).air_density == pytest.approx(1 - 0.378 * vapour_share, rel=1e-3)


def test_humidity_must_be_a_fraction() -> None:
    """A humidity given as a percentage is refused."""
    with pytest.raises(ValueError, match="relative_humidity"):
        StandardAtmosphere1976(relative_humidity=50.0)


def test_truth_configuration_anchors_the_atmosphere_at_the_launch_elevation() -> None:
    """The site temperature applies at the launch elevation, not at sea level."""
    site = truth(
        atmosphere=StandardAtmosphere1976(launch_temperature=scalar(30.0, "degC")),
        launch_elevation=scalar(1400.0, "m"),
    )

    assert site.atmosphere.conditions(1400.0).temperature == pytest.approx(303.15)
