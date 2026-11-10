"""EDITH campaign file tests."""

import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from edith.config import DEFAULT_MAX_FAILED_FRACTION, EdithConfig
from edith.dispersion import (
    DISPERSION_CLASSES,
    DISTRIBUTIONS,
    STANDARD_DISPERSION,
    Normal,
    StandardDispersion,
    Uniform,
)
from edith.errors import EdithConfigError
from flight_sim.config import ConfigError
from flight_sim.units import scalar

_EXAMPLE = Path("examples/sol_invictus")
_CAMPAIGN = Path("examples/sol_invictus_mc/edith_config.toml")

_HEADER = """
num_batches = 2
flight_per_batch = 5
master_seed = 1
base_sim = "../base/sim.toml"
output_dir = "../out"
generator = "LatinHypercubeGenerator"
"""

_STANDARD = 'dispersion = "StandardDispersion"'

_MASS_KEY = "properties.dry_mass_properties.mass"
_MASS_BODY = 'distribution = "Normal"\nsigma = "0.25 kg"'


def _table(key: str, body: str, dispersion: str = _STANDARD) -> str:
    """Return a dispersion table: its key, its dispersion class and its body."""
    return f'\n[dispersions."{key}"]\n{dispersion}\n{body}\n'


_MASS = _table(_MASS_KEY, _MASS_BODY)


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
    return str(error.value)


def _dispersion(tmp_path: Path, table: str) -> StandardDispersion:
    """Return the one dispersion a campaign file of one table describes."""
    (dispersion,) = EdithConfig.from_file(_campaign(tmp_path, table)).dispersions
    assert isinstance(dispersion, StandardDispersion)
    return dispersion


def test_the_trials_in_all_are_the_batches_times_their_size(tmp_path: Path) -> None:
    """A campaign gives its batches and the trials of each, and flies their product."""
    config = EdithConfig.from_file(_campaign(tmp_path, _MASS, _HEADER))

    assert (config.num_batches, config.flight_per_batch, config.num_sims) == (2, 5, 10)


def test_the_example_campaign_loads() -> None:
    """The shipped example reads, with every dispersion the standard class."""
    config = EdithConfig.from_file(_CAMPAIGN)

    assert config.source.path == config.base_sim == (_EXAMPLE / "sim.toml").resolve()
    assert config.output_dir.parent == Path("out").resolve()
    assert config.dispersions
    assert all(isinstance(d, StandardDispersion) for d in config.dispersions)


def test_dispersions_read_in_file_order_in_the_base_sims_units(tmp_path: Path) -> None:
    """Dispersions keep their order, and their numbers arrive in the nominal's units."""
    tables = (
        _MASS
        + _table(
            "config.truth.wind.speed",
            'distribution = "Uniform"\nlow = "0 mph"\nhigh = "15 mph"',
        )
        + _table(
            "config.truth.wind.from_azimuth",
            'distribution = "Uniform"\nhalf_width = "30 deg"',
        )
        + _table(
            "config.truth.launch_rail.friction_coefficient",
            'distribution = "Uniform"\nlow = 0.0\nhigh = 0.2',
        )
    )

    config = EdithConfig.from_file(_campaign(tmp_path, tables))

    mass, speed, azimuth, friction = config.dispersions
    assert mass == StandardDispersion(_MASS_KEY, 17.0, Normal(0.25), "kg")
    assert isinstance(speed, StandardDispersion)
    assert isinstance(speed.distribution, Uniform)
    assert (speed.nominal, speed.units, speed.distribution.low) == (0.0, "m/s", 0.0)
    assert speed.distribution.high == pytest.approx(15 * 0.44704)
    assert azimuth == StandardDispersion(
        "config.truth.wind.from_azimuth", 0.0, Uniform(-30.0, 30.0), "deg"
    )
    assert friction == StandardDispersion(
        "config.truth.launch_rail.friction_coefficient", 0.0, Uniform(0.0, 0.2), ""
    )


def test_config_survives_pickling(tmp_path: Path) -> None:
    """A campaign reaches worker processes intact."""
    config = EdithConfig.from_file(_campaign(tmp_path, _MASS))

    assert pickle.loads(pickle.dumps(config)) == config


def test_errors_are_config_errors() -> None:
    """One except catches the errors of flight_sim's loader and EDITH's."""
    assert issubclass(EdithConfigError, ConfigError)
    assert issubclass(EdithConfigError, ValueError)


def test_left_out_settings_take_their_defaults(tmp_path: Path) -> None:
    """Left out, the failed fraction is the default and there is no confidence."""
    config = EdithConfig.from_file(_campaign(tmp_path, _MASS, _HEADER))

    assert config.max_failed_fraction == DEFAULT_MAX_FAILED_FRACTION
    assert config.stopping.confidence is None


def test_generator_is_required(tmp_path: Path) -> None:
    """The generator has no default, so a campaign file must give it."""
    header = "\n".join(
        line for line in _HEADER.splitlines() if not line.startswith("generator")
    )

    assert "missing generator" in _error(tmp_path, _MASS, header + "\n")


def test_the_confidence_interval_rule_needs_a_confidence(tmp_path: Path) -> None:
    """Only the confidence-interval rule needs a confidence."""
    stopping = '\n[stopping]\nrule = "ConfidenceIntervalTarget"\ntarget = "5 m"\n'

    message = _error(tmp_path, _MASS, _HEADER + stopping)

    assert 'rule "ConfidenceIntervalTarget" needs a confidence' in message


def test_optional_settings(tmp_path: Path) -> None:
    """Flights per batch, timeout, generator, input mode and the rest are read."""
    header = _HEADER.replace("flight_per_batch = 5", "flight_per_batch = 2").replace(
        "LatinHypercubeGenerator", "RandomGenerator"
    )
    header += 'trial_timeout = "10 min"\ninput_mode = "lightweight"\n'
    header += "max_failed_fraction = 0.25\n"
    config = EdithConfig.from_file(_campaign(tmp_path, _MASS, header))

    assert config.generator == "RandomGenerator"
    assert config.flight_per_batch == 2
    assert config.max_failed_fraction == 0.25
    assert config.trial_timeout == scalar(600.0, "s")
    assert config.input_mode == "lightweight"


def test_a_file_that_is_not_utf8_is_named(tmp_path: Path) -> None:
    """A campaign file in another encoding is reported, not a traceback."""
    path = _campaign(tmp_path, _MASS)
    path.write_bytes(b"\xff\xfe" + path.read_bytes())

    with pytest.raises(EdithConfigError, match=r"edith_config\.toml is not UTF-8 text"):
        EdithConfig.from_file(path)


def test_a_campaign_file_may_extend_another(tmp_path: Path) -> None:
    """flight_sim reads campaign files as it reads sim files, so extends works."""
    path = _campaign(tmp_path, _MASS)
    (path.parent / "common.toml").write_text(_HEADER, encoding="utf-8")
    path.write_text(
        'extends = "common.toml"\nnum_batches = 4\n' + _MASS, encoding="utf-8"
    )

    config = EdithConfig.from_file(path)

    assert (config.num_sims, config.master_seed) == (20, 1)
    assert config.base_sim == (tmp_path / "base" / "sim.toml").resolve()


def test_invalid_toml_names_the_file_and_line(tmp_path: Path) -> None:
    """A syntax error is reported with the file and its line."""
    message = _error(tmp_path, '\n[dispersions."properties.x"\n')

    assert "edith_config.toml" in message
    assert "line 9" in message


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            _HEADER.replace("master_seed", "master_sead"),
            "top level: unknown keys ['master_sead']; "
            "CampaignSettings needs num_batches",
        ),
        (_HEADER.replace("num_batches =", "N ="), "top level: unknown keys ['N']"),
        (_HEADER.replace("num_batches = 2\n", ""), "top level: missing num_batches;"),
        (
            _HEADER.replace("num_batches = 2", "num_batches = true"),
            "num_batches: expected an integer, got True",
        ),
        (
            _HEADER.replace("num_batches = 2", "num_batches = 2.5"),
            "num_batches: expected an integer, got 2.5",
        ),
        (
            _HEADER.replace("num_batches = 2", "num_batches = 0"),
            "num_batches must be at least 1, got 0",
        ),
        (
            _HEADER.replace("seed = 1", "seed = -1"),
            "master_seed must be at least 0, got -1",
        ),
        (
            _HEADER.replace('"LatinHypercubeGenerator"', '"LHS"'),
            'generator must be "RandomGenerator" or "LatinHypercubeGenerator", '
            "got 'LHS'",
        ),
        (
            _HEADER.replace("flight_per_batch = 5", "flight_per_batch = 0"),
            "flight_per_batch must be at least 1, got 0",
        ),
        (
            _HEADER.replace("flight_per_batch = 5\n", ""),
            "top level: missing flight_per_batch;",
        ),
        (_HEADER + "num_sims = 10\n", "top level: unknown keys ['num_sims']"),
        (
            _HEADER + "max_failed_fraction = 1.5\n",
            "max_failed_fraction must be between 0 and 1, got 1.5",
        ),
        (
            _HEADER + 'trial_timeout = "10 m"\n',
            "(assigned to CampaignSettings.trial_timeout)",
        ),
        (_HEADER + "trial_timeout = 600\n", "trial_timeout: 600 needs units"),
        (_HEADER + 'trial_timeout = "-1 s"\n', "trial_timeout must be above 0"),
        (_HEADER + 'trial_timeout = "1e400 s"\n', "trial_timeout must be above 0"),
        (_HEADER + 'trial_timeout = "abc"\n', "trial_timeout: cannot read 'abc'"),
        (
            _HEADER.replace("base/sim.toml", "base/sim.tom"),
            "base_sim must be a sim input file ending in .toml, got 'sim.tom'",
        ),
        (
            _HEADER.replace("base/sim.toml", "base/nope.toml"),
            "nope.toml does not exist",
        ),
        (_HEADER.replace('"../base/sim.toml"', "5"), "base_sim must be a path, got 5"),
        (_HEADER.replace('"../out"', '"../base/sim.toml"'), "sim.toml is a file"),
        (
            _HEADER.replace('"../out"', '"../out/{nope}"'),
            "has a bad placeholder ('nope'); use {name}, {datetime} or "
            "{datetime:%Y%m%d}",
        ),
        (_HEADER + 'type = "os.getcwd"\n', "type: a campaign file takes no type"),
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


def test_output_dir_takes_the_sims_placeholders(tmp_path: Path) -> None:
    """{name} is the base sim's name as a file name, and {datetime} the start."""
    header = _HEADER.replace('"../out"', '"../out/{name}_{datetime:%Y}"')

    config = EdithConfig.from_file(_campaign(tmp_path, _MASS, header))

    expected = tmp_path / "out" / f"sol_invictus_example_{datetime.now():%Y}"
    assert config.output_dir == expected.resolve()


def test_a_campaign_file_may_sit_beside_the_base_sim(tmp_path: Path) -> None:
    """The campaign file and output folder may lie anywhere, even beside the sim."""
    _campaign(tmp_path, _MASS)
    inside = tmp_path / "base" / "edith_config.toml"
    header = _HEADER.replace("../base/", "").replace('"../out"', '"out/edith"')
    inside.write_text(header + _MASS, encoding="utf-8")

    config = EdithConfig.from_file(inside)

    assert config.output_dir == (tmp_path / "base" / "out" / "edith").resolve()


def test_included_files_may_lie_anywhere(tmp_path: Path) -> None:
    """flight_sim resolves includes in other folders, so the key path is the same."""
    _campaign(tmp_path, _MASS)
    elsewhere = tmp_path / "elsewhere" / "vehicles"
    elsewhere.mkdir(parents=True)
    for name in ("vehicle.toml", "aero.csv", "Loki_M3464LB.eng"):
        shutil.move(tmp_path / "base" / name, elsewhere / name)
    sim = tmp_path / "base" / "sim.toml"
    sim.write_text(
        sim.read_text(encoding="utf-8").replace(
            '"vehicle.toml"', '"../elsewhere/vehicles/vehicle.toml"'
        ),
        encoding="utf-8",
    )

    dispersion = _dispersion(tmp_path, _MASS)

    assert dispersion.key == "properties.dry_mass_properties.mass"
    assert (dispersion.nominal, dispersion.units) == (17.0, "kg")


@pytest.mark.parametrize("dispersion", [_STANDARD, ""])
def test_the_dispersion_class_is_read(tmp_path: Path, dispersion: str) -> None:
    """A dispersion names StandardDispersion, or leaves it out for the default."""
    found = _dispersion(tmp_path, _table(_MASS_KEY, _MASS_BODY, dispersion))

    assert type(found).__name__ == STANDARD_DISPERSION == "StandardDispersion"
    assert pickle.loads(pickle.dumps(found)) == found


@pytest.mark.parametrize(
    ("dispersion", "expected"),
    [
        (
            'dispersion = "AeroDispersion"',
            "dispersion: EDITH has no dispersion class 'AeroDispersion'; for now "
            'every dispersion leaves dispersion out, or writes dispersion = "'
            'StandardDispersion"',
        ),
        (
            'dispersion = "standarddispersion"',
            "dispersion: EDITH has no dispersion class 'standarddispersion'",
        ),
        (
            'dispersion = { module = "StandardDispersion", '
            'class = "StandardDispersion" }',
            "dispersion: expected a string, got {'module': 'StandardDispersion', "
            "'class': 'StandardDispersion'}; for now every dispersion leaves "
            "dispersion out",
        ),
        ("dispersion = 1", "dispersion: expected a string, got 1; for now every"),
        ('dispersion = "Normal"', "EDITH has no dispersion class 'Normal';"),
        ('dispersion = "__doc__"', "EDITH has no dispersion class '__doc__';"),
        (
            'dispersion = "BaseDispersion"',
            "EDITH has no dispersion class 'BaseDispersion';",
        ),
        (
            'dispersoin = "StandardDispersion"',
            "dispersoin: unknown key (did you mean dispersion?)",
        ),
    ],
)
def test_bad_dispersion_classes_are_named(
    tmp_path: Path, dispersion: str, expected: str
) -> None:
    """A dispersion that names anything but StandardDispersion is rejected."""
    assert expected in _error(tmp_path, _table(_MASS_KEY, _MASS_BODY, dispersion))


def test_distributions_are_found_by_their_class_name() -> None:
    """Each distribution class is found by its name."""
    assert {"Normal": Normal, "Uniform": Uniform} == DISTRIBUTIONS


def test_output_settings_override_the_base_sims(tmp_path: Path) -> None:
    """An output table is kept as written, for every trial to take."""
    output = '\n[output]\nplots = []\nformats = ["csv", "parquet"]\n'

    config = EdithConfig.from_file(_campaign(tmp_path, _MASS + output))

    assert config.output == {"plots": [], "formats": ["csv", "parquet"]}
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('[output]\ndirectory = "x"\n', "output.directory: each trial writes to"),
        ('[output]\ntype = "os.getcwd"\n', "output.type: unknown key; output takes"),
        ('[output]\nplots = ["nope"]\n', "output: Unknown plots ['nope']"),
        (
            '[output]\nformat = ["csv"]\n',
            "output.format: unknown key (did you mean formats?); output takes the "
            "keys of a sim's [output] table but directory: formats, units, "
            "channels, sample_interval, plots",
        ),
        ('[output]\nsample_interval = "-1 s"\n', "output: Sample interval must be"),
    ],
)
def test_bad_output_settings_are_named(
    tmp_path: Path, output: str, expected: str
) -> None:
    """An output table is checked as flight_sim checks a sim's, under output."""
    assert expected in _error(tmp_path, _MASS + "\n" + output)


def test_output_is_a_table(tmp_path: Path) -> None:
    """A bare output value is rejected before the settings are built."""
    message = _error(tmp_path, _MASS, _HEADER + "output = 5\n")

    assert "output: expected a table of flight_sim output settings" in message


def test_dispersion_classes_are_found_by_name() -> None:
    """Each subclass of BaseDispersion is found by its name, and the base is not."""
    assert {"StandardDispersion": StandardDispersion} == DISPERSION_CLASSES


def test_a_dispersion_is_a_table(tmp_path: Path) -> None:
    """Each item under dispersions is a table, not a bare value."""
    assert "dispersions.mass: expected a table of dispersion, distribution and its" in (
        _error(tmp_path, "dispersions = { mass = 5 }\n")
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("properties.dry_mass_properties.mas", "did you mean"),
        ("propertys.dry_mass_properties.mass", "has no propertys (did you mean"),
        ("properties.dry_mass_properties.cg_location", "cg_location.value[0]"),
        ("properties.dry_mass_properties.cg_location.value[3]", "has 3 items"),
        ("properties.dry_mass_properties.mass.value", "mass is '17 kg'"),
        ("properties.aero_table.frame", 'is "missile", not a number'),
        ("properties.dry mass", "cannot read the key path"),
        ("vehicle.toml:dry_mass_properties.mass", "cannot read the key path"),
        ("properties.dry_mass_properties", "the value is a table"),
        ("properties.aero_table", "the value is a table"),
        ("properties.dry_mass_properties.cg_location.value", "the value is a list"),
        ("config.truth.atmosphere.launch_temperature", "the value is None"),
    ],
)
def test_bad_targets_are_named(tmp_path: Path, key: str, expected: str) -> None:
    """A key that names no number of the base sim says what is wrong."""
    message = _error(tmp_path, _table(key, 'distribution = "Normal"\nsigma = "1 m"'))

    assert f'dispersions."{key}"' in message
    assert expected in message


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
    dispersion = _table(_MASS_KEY, f'distribution = "Normal"\n{body}')

    assert expected in _error(tmp_path, dispersion)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('distribution = "gaussian"', 'distribution: expected "Normal" or "Uniform"'),
        ('sigma = "1 m/s"', 'missing distribution, "Normal" or "Uniform"'),
        ('distribution = "Uniform"\nsigma = "1 m/s"', "sigma: unknown key; a uniform"),
        (
            'distribution = "Uniform"\nhalf_width = "1 m/s"\nlow = "0 m/s"',
            "give half_width, or low and high, not both",
        ),
        ('distribution = "Uniform"\nlow = "0 m/s"', "missing high;"),
        ('distribution = "Uniform"\nhigh = "1 m/s"', "missing low;"),
        ('distribution = "Uniform"', "missing half_width, or low and high;"),
        ('distribution = "Uniform"\nhalf_width = "-1 m/s"', "expected 0 or more"),
        (
            'distribution = "Uniform"\nlow = "20 mph"\nhigh = "8 m/s"',
            "low (20 mph) must be below high (8 m/s)",
        ),
        ('distribution = "Normal"', "missing sigma;"),
    ],
)
def test_bad_distributions_are_named(tmp_path: Path, body: str, expected: str) -> None:
    """Each distribution type takes its own keys."""
    assert expected in _error(tmp_path, _table("config.truth.wind.speed", body))


@pytest.mark.parametrize("sigma", ["0.1 m", "20 percent"])
def test_a_plain_nominal_takes_plain_numbers(tmp_path: Path, sigma: str) -> None:
    """Rail friction is a plain number in sim.toml, so its spread is too."""
    key = "config.truth.launch_rail.friction_coefficient"

    message = _error(
        tmp_path, _table(key, f'distribution = "Normal"\nsigma = "{sigma}"')
    )

    assert "takes a plain number, since the nominal value is 0.0, a plain" in message
    assert message.endswith(f"got '{sigma}'")


def test_numbers_convert_to_the_nominal_units(tmp_path: Path) -> None:
    """Spreads in other units arrive in the units the base sim writes."""
    dispersion = _dispersion(
        tmp_path,
        _table(
            _MASS_KEY,
            'distribution = "Normal"\nsigma = "250 g"\nlow = "30 lb"\nhigh = "18 kg"',
        ),
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
        _table(
            "config.truth.atmosphere.barometric_pressure",
            'distribution = "Normal"\nsigma = "1 kPa"',
        ),
    )

    assert (dispersion.nominal, dispersion.units) == (101325.0, "Pa")
    assert dispersion.distribution == Normal(1000.0)


@pytest.mark.parametrize("sigma", ["18 degF", "10 delta_degC", "10 K"])
def test_offset_nominals_take_their_own_units(tmp_path: Path, sigma: str) -> None:
    """A degC nominal takes spreads in degC, and no other temperature units."""
    _campaign(tmp_path, "")
    site = tmp_path / "base" / "site.toml"
    site.write_text(
        site.read_text(encoding="utf-8").replace(
            "[atmosphere]", '[atmosphere]\nlaunch_temperature = "20 degC"'
        ),
        encoding="utf-8",
    )
    key = "config.truth.atmosphere.launch_temperature"

    dispersion = _dispersion(
        tmp_path,
        _table(key, 'distribution = "Normal"\nsigma = "10 degC"\nlow = "10 degC"'),
    )

    assert (dispersion.nominal, dispersion.units) == (20.0, "degC")
    assert dispersion.distribution == Normal(10.0, low=10.0)
    assert f"write '{sigma}' in degC" in _error(
        tmp_path, _table(key, f'distribution = "Normal"\nsigma = "{sigma}"')
    )


def test_angles_are_not_plain_numbers(tmp_path: Path) -> None:
    """Pint converts "5 percent" to 0.05 rad, but an angle's spread is an angle."""
    key = "config.truth.launch_rail.elevation"

    dispersion = _dispersion(
        tmp_path, _table(key, 'distribution = "Normal"\nsigma = "0.01 rad"')
    )

    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.sigma == pytest.approx(0.5729577951308232)
    assert '"5 percent" does not convert to deg' in _error(
        tmp_path, _table(key, 'distribution = "Normal"\nsigma = "5 percent"')
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('low = "2 m/s"\nhigh = "8 m/s"', "low: 2 m/s is above the nominal value"),
        ('low = "-8 m/s"\nhigh = "-2 m/s"', "high: -2 m/s is below the nominal value"),
    ],
)
def test_a_uniform_must_contain_the_nominal(
    tmp_path: Path, body: str, expected: str
) -> None:
    """A uniform's low and high bracket the nominal value, wind speed 0 here."""
    dispersion = _table("config.truth.wind.speed", f'distribution = "Uniform"\n{body}')

    assert expected in _error(tmp_path, dispersion)


def test_a_normal_may_be_truncated_at_its_nominal(tmp_path: Path) -> None:
    """A bound equal to the nominal value, in any units, is allowed."""
    dispersion = _dispersion(
        tmp_path,
        _table(
            "config.truth.launch_rail.elevation",
            'distribution = "Normal"\nsigma = "1 deg"\nhigh = "1.4835298641951802 rad"',
        ),
    )

    assert isinstance(dispersion.distribution, Normal)
    assert dispersion.distribution.high == pytest.approx(85.0)


def test_two_dispersions_of_one_value_are_rejected(tmp_path: Path) -> None:
    """A value is dispersed by one dispersion at most, however its key is written."""
    cg = "properties.dry_mass_properties.cg_location.value"
    body = 'distribution = "Normal"\nsigma = "1 cm"'
    twice = _table(f"{cg}[0]", body) + _table(f"{cg}[00]", body)

    assert f'dispersions."{cg}[00]": changes {cg}[0], which' in (
        _error(tmp_path, twice)
    )


@pytest.mark.parametrize(
    ("dispersions", "expected"),
    [
        ("dispersions = {}\n", "dispersions: expected one or more tables"),
        ("", "missing dispersions, a table of each value to disperse"),
    ],
)
def test_no_dispersions_is_rejected(
    tmp_path: Path, dispersions: str, expected: str
) -> None:
    """A campaign disperses at least one value."""
    assert expected in _error(tmp_path, dispersions)


def test_zero_spread_is_allowed(tmp_path: Path) -> None:
    """A sigma of 0 keeps a value at its nominal."""
    dispersion = _dispersion(tmp_path, _MASS.replace('"0.25 kg"', '"0 kg"'))

    assert dispersion.distribution == Normal(0.0)


@pytest.mark.parametrize(
    ("header", "dispersions"),
    [
        (_HEADER, _MASS.replace("Normal", "gaussian")),
        (_HEADER.replace("master_seed", "master_sead"), _MASS),
        (_HEADER.replace("num_batches = 2", "num_batches = 0"), _MASS),
        (_HEADER.replace("flight_per_batch = 5", "flight_per_batch = true"), _MASS),
    ],
)
def test_errors_name_the_campaign_file(
    tmp_path: Path, header: str, dispersions: str
) -> None:
    """Every error starts with the campaign file it is about."""
    message = _error(tmp_path, dispersions, header)

    assert message.split(": ", maxsplit=1)[0].endswith("mc/edith_config.toml")
