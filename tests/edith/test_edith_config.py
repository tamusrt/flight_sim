"""EDITH campaign file tests."""

import pickle
import shutil
from pathlib import Path

import pytest

from edith.config import (
    AeroTarget,
    Dispersion,
    EdithConfig,
    EdithConfigError,
    Normal,
    Uniform,
    ValueTarget,
)
from flight_sim.config import ConfigError

_EXAMPLE = Path("examples/sol_invictus")
_CAMPAIGN = Path("examples/sol_invictus_mc/edith_config.toml")

_HEADER = """
N = 10
master_seed = 1
base_sim = "../base/sim.toml"
output_dir = "../out"
"""

_MASS = """
[dispersions."properties.dry_mass_properties.mass"]
dist_type = "normal"
sigma = "0.25 kg"
"""


def _campaign(tmp_path: Path, dispersions: str, header: str = _HEADER) -> Path:
    """Write a campaign file beside a copy of the example, and return its path."""
    base = tmp_path / "base"
    if not base.exists():
        shutil.copytree(_EXAMPLE, base, ignore=shutil.ignore_patterns("out"))
    folder = tmp_path / "mc"
    folder.mkdir(exist_ok=True)
    path = folder / "edith_config.toml"
    path.write_text(header + dispersions, encoding="utf-8")
    return path


def _error(tmp_path: Path, dispersions: str, header: str = _HEADER) -> str:
    """Return the message of the error a campaign file raises."""
    with pytest.raises(EdithConfigError) as error:
        EdithConfig.from_file(_campaign(tmp_path, dispersions, header))
    print(error.value)
    return str(error.value)


def _dispersion(tmp_path: Path, table: str) -> Dispersion:
    """Return the one dispersion a campaign file of one table describes."""
    (dispersion,) = EdithConfig.from_file(_campaign(tmp_path, table)).dispersions
    return dispersion


def test_example_campaign_reads_every_dispersion() -> None:
    """The example builds its settings and nine dispersions in base-sim units."""
    config = EdithConfig.from_file(_CAMPAIGN)
    aero = (_EXAMPLE / "aero.csv").resolve()
    wind_speed = config.dispersions[4]

    assert (config.trials, config.master_seed, config.sampling) == (500, 9999, "lhs")
    assert config.base_sim == (_EXAMPLE / "sim.toml").resolve()
    assert config.source.path == config.base_sim
    assert config.output_dir == Path("out/edith_sol_invictus").resolve()
    assert (config.workers, config.batch_size, config.trial_timeout_s) == (
        None,
        None,
        None,
    )
    assert config.dispersions[:4] == (
        Dispersion(
            "properties.dry_mass_properties.mass",
            ValueTarget(("properties", "dry_mass_properties", "mass")),
            17.0,
            Normal(0.25),
            "kg",
        ),
        Dispersion(
            "properties.dry_mass_properties.cg_location.value[0]",
            ValueTarget(
                ("properties", "dry_mass_properties", "cg_location", "value", 0)
            ),
            -1.0,
            Normal(0.02),
            "m",
        ),
        Dispersion(
            "config.truth.launch_rail.elevation",
            ValueTarget(("config", "truth", "launch_rail", "elevation")),
            85.0,
            Normal(0.5, high=90.0),
            "deg",
        ),
        Dispersion(
            "config.truth.launch_rail.friction_coefficient",
            ValueTarget(("config", "truth", "launch_rail", "friction_coefficient")),
            0.0,
            Uniform(0.0, 0.2),
            "",
        ),
    )
    assert (wind_speed.key, wind_speed.nominal, wind_speed.units) == (
        "config.truth.wind.speed",
        0.0,
        "m/s",
    )
    assert isinstance(wind_speed.distribution, Uniform)
    assert wind_speed.distribution.low == 0.0
    assert wind_speed.distribution.high == pytest.approx(15 * 0.44704)
    assert config.dispersions[5:] == (
        Dispersion(
            "config.truth.wind.from_azimuth",
            ValueTarget(("config", "truth", "wind", "from_azimuth")),
            0.0,
            Uniform(-30.0, 30.0),
            "deg",
        ),
        Dispersion(
            "properties.aero_table:Cx",
            AeroTarget(("properties", "aero_table"), aero, ("Cx",)),
            1.0,
            Normal(0.05),
            "",
        ),
        Dispersion(
            "properties.aero_table:Cy+Cz",
            AeroTarget(("properties", "aero_table"), aero, ("Cy", "Cz")),
            1.0,
            Normal(0.1),
            "",
        ),
        Dispersion(
            "properties.aero_table:xCP",
            AeroTarget(("properties", "aero_table"), aero, ("xCP",)),
            1.0,
            Normal(0.02),
            "",
        ),
    )


def test_config_survives_pickling() -> None:
    """A campaign reaches worker processes intact."""
    config = EdithConfig.from_file(_CAMPAIGN)

    assert pickle.loads(pickle.dumps(config)) == config


def test_errors_are_config_errors() -> None:
    """One except catches the errors of flight_sim's loader and EDITH's."""
    assert issubclass(EdithConfigError, ConfigError)
    assert issubclass(EdithConfigError, ValueError)


def test_optional_settings(tmp_path: Path) -> None:
    """Workers, batch size, timeout and sampling are read when given."""
    header = _HEADER + (
        'sampling = "random"\nworkers = 3\nbatch_size = 4\ntrial_timeout = "10 min"\n'
    )
    config = EdithConfig.from_file(_campaign(tmp_path, _MASS, header))

    assert config.sampling == "random"
    assert (config.workers, config.batch_size) == (3, 4)
    assert config.trial_timeout_s == pytest.approx(600.0)


def test_byte_order_mark_is_read(tmp_path: Path) -> None:
    """A campaign file saved with a byte-order mark still reads."""
    path = _campaign(tmp_path, _MASS)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    assert EdithConfig.from_file(path).trials == 10


def test_invalid_toml_names_the_file_and_line(tmp_path: Path) -> None:
    """A syntax error is reported with the file and its line."""
    message = _error(tmp_path, '\n[dispersions."properties.x"\n')

    assert "edith_config.toml" in message
    assert "line 7" in message


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (_HEADER.replace("master_seed", "master_sead"), "master_sead: unknown key"),
        (_HEADER.replace("master_seed", "master_sead"), "did you mean master_seed"),
        (_HEADER.replace("N =", "n ="), "did you mean N?"),
        (_HEADER.replace("N = 10", ""), "missing N;"),
        (_HEADER.replace("N = 10", "N = true"), "N: expected a whole number"),
        (_HEADER.replace("N = 10", "N = 0"), "of at least 1, got 0"),
        (_HEADER.replace("N = 10", "N = 2.5"), "got 2.5"),
        (_HEADER.replace("seed = 1", "seed = -1"), "master_seed: expected"),
        (_HEADER + 'sampling = "LHS"\n', 'sampling: expected "lhs" or "random"'),
        (_HEADER + "workers = 0\n", "workers: expected a whole number"),
        (_HEADER + 'trial_timeout = "10 m"\n', "trial_timeout: expected a time"),
        (_HEADER + "trial_timeout = 600\n", "trial_timeout: expected a time"),
        (_HEADER + 'trial_timeout = "-1 s"\n', "trial_timeout: expected a time"),
        (_HEADER.replace("base/sim.toml", "base/sim.tom"), "base_sim: expected"),
        (_HEADER.replace("base/sim.toml", "base/nope.toml"), "does not exist"),
    ],
)
def test_top_level_errors_name_the_key(
    tmp_path: Path, header: str, expected: str
) -> None:
    """Each bad top-level setting is reported with its key."""
    assert expected in _error(tmp_path, _MASS, header)


def test_a_base_sim_that_does_not_build_is_named(tmp_path: Path) -> None:
    """flight_sim's error for the base sim is reported under base_sim."""
    _campaign(tmp_path, _MASS)
    sim = tmp_path / "base" / "sim.toml"
    sim.write_text(
        sim.read_text(encoding="utf-8").replace('max_time = "300 s"', "max_time = 300"),
        encoding="utf-8",
    )

    message = _error(tmp_path, _MASS)

    assert "base_sim: " in message
    assert "max_time" in message


def test_a_campaign_file_may_sit_beside_the_base_sim(tmp_path: Path) -> None:
    """The campaign file and output folder may lie anywhere, even beside the sim."""
    _campaign(tmp_path, _MASS)
    inside = tmp_path / "base" / "edith_config.toml"
    header = _HEADER.replace("../base/", "").replace('"../out"', '"out/edith"')
    inside.write_text(header + _MASS, encoding="utf-8")

    config = EdithConfig.from_file(inside)

    assert config.output_dir == (tmp_path / "base" / "out" / "edith").resolve()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("properties.dry_mass_properties.mas", "did you mean"),
        ("propertys.dry_mass_properties.mass", "has no propertys (did you mean"),
        ("properties.dry_mass_properties.cg_location", "cg_location.value[0]"),
        ("properties.dry_mass_properties.cg_location.value[3]", "has 3 items"),
        ("properties.aero_table.frame", 'is "missile", not a number'),
        ("properties.dry mass", "cannot read the key path"),
        ("vehicle.toml:dry_mass_properties.mass", "the base sim has no vehicle"),
        ("properties.dry_mass_properties", "the value is a table"),
        ("properties.dry_mass_properties.cg_location.value", "the value is a list"),
        (
            "properties.aero_table",
            'scale its columns, such as "properties.aero_table:Cx"',
        ),
        ("config.truth.launch_rail:Cx", "launch_rail is not an aero table"),
        ("config.truth.atmosphere.launch_temperature", "the value is None"),
    ],
)
def test_bad_targets_are_named(tmp_path: Path, key: str, expected: str) -> None:
    """A key that names no number of the base sim says what is wrong."""
    dispersion = f'[dispersions."{key}"]\ndist_type = "normal"\nsigma = "1 m"\n'

    message = _error(tmp_path, dispersion)

    assert f'dispersions."{key}"' in message
    assert expected in message


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "properties.aero_table:Cd",
            "'Cd' is not a column the aero table reads; scale Cx, Cy, Cz, CMx, "
            "xCP or CxPowerOn, where Cx holds the drag",
        ),
        ("properties.aero_table:Mach", "'Mach' is not a column the aero table reads"),
        ("properties.aero_table:Cx+Cx", "names a column twice"),
        (
            "properties.aero_table:CxPowerOn",
            "aero.csv has no CxPowerOn column; its columns are Mach, Alpha, Phi, Cx",
        ),
    ],
)
def test_bad_aero_columns_are_named(tmp_path: Path, key: str, expected: str) -> None:
    """Only coefficient columns the CSV has can be scaled, each by one dispersion."""
    dispersion = f'[dispersions."{key}"]\ndist_type = "normal"\nsigma = 0.05\n'

    assert expected in _error(tmp_path, dispersion)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("sigma = 0.25", 'needs units, since the nominal value is "17 kg"'),
        ("sigma = 0.25", 'write "0.25 kg"'),
        ('sigma = "abc"', 'expected a number such as "1 kg"'),
        ('sigma = "2 m"', '"2 m" does not convert to kg, the units of the nominal'),
        ('sigma = "0.25 kgg"', "cannot read the units of '0.25 kgg'"),
        ('sigma = "0.25 m/"', "cannot read the units of '0.25 m/'"),
        ('sigma = "-0.25 kg"', "sigma: expected 0 or more, got -0.25 kg"),
        ('sigma = "1e400 kg"', "sigma: expected a finite number"),
        ('sigma = "1e306 Mg"', "sigma: expected a finite number"),
        ("sigma = inf", "expected a finite number"),
        ('sigm = "1 kg"', "sigm: unknown key (did you mean sigma?)"),
        ('sigma = "1 kg"\nlow = "18 kg"', "low: 18 kg is above the nominal value"),
        ('sigma = "1 kg"\nhigh = "30 lb"', "high: 30 lb is below the nominal value"),
        ('sigma = "1 kg"\nlow = "17 kg"\nhigh = "17 kg"', "low (17 kg) must be"),
    ],
)
def test_bad_normal_numbers_are_named(tmp_path: Path, body: str, expected: str) -> None:
    """A normal's numbers convert to the nominal's units and bracket it."""
    dispersion = (
        '[dispersions."properties.dry_mass_properties.mass"]\n'
        f'dist_type = "normal"\n{body}\n'
    )

    assert expected in _error(tmp_path, dispersion)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('dist_type = "gaussian"', 'dist_type: expected "normal" or "uniform"'),
        ('sigma = "1 m/s"', 'missing dist_type, "normal" or "uniform"'),
        ('dist_type = "uniform"\nsigma = "1 m/s"', "sigma: unknown key; a uniform"),
        (
            'dist_type = "uniform"\nhalf_width = "1 m/s"\nlow = "0 m/s"',
            "give half_width, or low and high, not both",
        ),
        ('dist_type = "uniform"\nlow = "0 m/s"', "missing high;"),
        ('dist_type = "uniform"\nhigh = "1 m/s"', "missing low;"),
        ('dist_type = "uniform"', "missing half_width, or low and high;"),
        ('dist_type = "uniform"\nhalf_width = "-1 m/s"', "expected 0 or more"),
        (
            'dist_type = "uniform"\nlow = "20 mph"\nhigh = "8 m/s"',
            "low (20 mph) must be below high (8 m/s)",
        ),
        ('dist_type = "normal"', "missing sigma;"),
    ],
)
def test_bad_distributions_are_named(tmp_path: Path, body: str, expected: str) -> None:
    """Each distribution type takes its own keys."""
    dispersion = f'[dispersions."config.truth.wind.speed"]\n{body}\n'

    assert expected in _error(tmp_path, dispersion)


@pytest.mark.parametrize("sigma", ["0.1 m", "20 percent"])
def test_a_plain_nominal_takes_plain_numbers(tmp_path: Path, sigma: str) -> None:
    """Rail friction is a plain number in sim.toml, so its spread is too."""
    key = "config.truth.launch_rail.friction_coefficient"

    message = _error(
        tmp_path, f'[dispersions."{key}"]\ndist_type = "normal"\nsigma = "{sigma}"\n'
    )

    assert "takes a plain number, since the nominal value is 0.0, a plain" in message
    assert message.endswith(f"got '{sigma}'")


def test_numbers_convert_to_the_nominal_units(tmp_path: Path) -> None:
    """Spreads in other units arrive in the units the base sim writes."""
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."properties.dry_mass_properties.mass"]\n'
        'dist_type = "normal"\nsigma = "250 g"\nlow = "30 lb"\nhigh = "18 kg"\n',
    )

    assert (dispersion.nominal, dispersion.units) == (17.0, "kg")
    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.sigma == pytest.approx(0.25)
    assert dispersion.distribution.low == pytest.approx(30 * 0.45359237)
    assert dispersion.distribution.high == pytest.approx(18.0)


def test_a_default_value_can_be_dispersed(tmp_path: Path) -> None:
    """A value the base sim leaves to its default is in the resolved input."""
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."config.truth.atmosphere.barometric_pressure"]\n'
        'dist_type = "normal"\nsigma = "1 kPa"\n',
    )

    assert (dispersion.nominal, dispersion.units) == (101325.0, "Pa")
    assert dispersion.distribution == Normal(1000.0)


@pytest.mark.parametrize("sigma", ["18 degF", "10 delta_degC", "10 K"])
def test_spreads_convert_as_differences(tmp_path: Path, sigma: str) -> None:
    """A sigma of 18 degF is 10 degC, while a bound of 50 degF is 10 degC too."""
    _campaign(tmp_path, "")
    site = tmp_path / "base" / "site.toml"
    site.write_text(
        site.read_text(encoding="utf-8").replace(
            "[atmosphere]", '[atmosphere]\nlaunch_temperature = "20 degC"'
        ),
        encoding="utf-8",
    )
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."config.truth.atmosphere.launch_temperature"]\n'
        f'dist_type = "normal"\nsigma = "{sigma}"\nlow = "50 degF"\n',
    )

    assert (dispersion.nominal, dispersion.units) == (20.0, "degC")
    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.sigma == pytest.approx(10.0)
    assert dispersion.distribution.low == pytest.approx(10.0)


def test_angles_are_not_plain_numbers(tmp_path: Path) -> None:
    """Pint converts "5 percent" to 0.05 rad, but an angle's spread is an angle."""
    table = (
        '[dispersions."config.truth.launch_rail.elevation"]\n'
        'dist_type = "normal"\nsigma = "{}"\n'
    )

    dispersion = _dispersion(tmp_path, table.format("0.01 rad"))

    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.sigma == pytest.approx(0.5729577951308232)
    assert '"5 percent" does not convert to deg' in _error(
        tmp_path, table.format("5 percent")
    )


def test_a_uniform_may_leave_out_the_nominal(tmp_path: Path) -> None:
    """A uniform's low and high are absolute, so a calm nominal wind may lie outside."""
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."config.truth.wind.speed"]\n'
        'dist_type = "uniform"\nlow = "2 m/s"\nhigh = "8 m/s"\n',
    )

    assert dispersion.nominal == 0.0
    assert dispersion.distribution == Uniform(2.0, 8.0)


def test_a_normal_may_be_truncated_at_its_nominal(tmp_path: Path) -> None:
    """A bound equal to the nominal value, in any units, is allowed."""
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."config.truth.launch_rail.elevation"]\n'
        'dist_type = "normal"\nsigma = "1 deg"\nhigh = "1.4835298641951802 rad"\n',
    )

    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.high == pytest.approx(85.0)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('dist_type = "normal"\nsigma = "0.05 kg"', "takes a plain number"),
        ('dist_type = "normal"\nsigma = 0.25', "sigma: 0.25 puts 0 within 5 sigma"),
        ('dist_type = "normal"\nsigma = 0.05\nlow = 0', "the factor's low end is 0"),
        ('dist_type = "uniform"\nhalf_width = 1.5', "the factor's low end is -0.5"),
    ],
)
def test_aero_factors_are_plain_and_stay_positive(
    tmp_path: Path, body: str, expected: str
) -> None:
    """An aero factor about 1 takes plain numbers that keep it above 0."""
    assert expected in _error(
        tmp_path, f'[dispersions."properties.aero_table:Cx"]\n{body}\n'
    )


def test_a_wide_aero_factor_with_low_is_allowed(tmp_path: Path) -> None:
    """A large sigma is fine once low keeps the factor above 0."""
    dispersion = _dispersion(
        tmp_path,
        '[dispersions."properties.aero_table:Cx"]\n'
        'dist_type = "normal"\nsigma = 0.25\nlow = 0.5\n',
    )

    assert dispersion.distribution == Normal(0.25, low=0.5)


def test_two_dispersions_of_one_value_are_rejected(tmp_path: Path) -> None:
    """A value or column is dispersed by one dispersion at most."""
    twice = """
[dispersions."properties.dry_mass_properties.cg_location.value[0]"]
dist_type = "normal"
sigma = "1 cm"

[dispersions."properties.dry_mass_properties.cg_location.value[00]"]
dist_type = "normal"
sigma = "1 cm"
"""
    columns = """
[dispersions."properties.aero_table:Cx"]
dist_type = "normal"
sigma = 0.05

[dispersions."properties.aero_table:Cz+Cx"]
dist_type = "normal"
sigma = 0.05
"""
    assert "changes properties.dry_mass_properties.cg_location.value[0], which" in (
        _error(tmp_path, twice)
    )
    assert "changes the Cx column of aero.csv" in _error(tmp_path, columns)


def test_no_dispersions_is_rejected(tmp_path: Path) -> None:
    """A campaign disperses at least one value."""
    assert "dispersions: expected one or more tables" in _error(
        tmp_path, "dispersions = {}\n"
    )


def test_zero_spread_is_allowed(tmp_path: Path) -> None:
    """A sigma of 0 keeps a value at its nominal."""
    dispersion = _dispersion(tmp_path, _MASS.replace('"0.25 kg"', '"0 kg"'))

    assert dispersion.distribution == Normal(0.0)


def test_errors_name_the_campaign_file(tmp_path: Path) -> None:
    """Every error starts with the campaign file it is about."""
    message = _error(tmp_path, _MASS.replace("normal", "gaussian"))

    assert message.split(": ", maxsplit=1)[0].endswith("mc/edith_config.toml")
