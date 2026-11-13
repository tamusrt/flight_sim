"""EDITH statistics tests: distributions, generators, estimators and the loop."""

import math
import shutil
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from edith.config import EdithConfig
from edith.dispersion import _distribution_name
from edith.errors import EdithConfigError
from edith.parallel import Workers
from edith.runner import CampaignResult, fly_campaign, prepare_batches
from edith.statistics.distributions import (
    DISTRIBUTIONS,
    Beta,
    Distribution,
    LogNormal,
    Normal,
    Parameter,
    ParameterError,
    StudentT,
    Triangular,
    Uniform,
    VonMises,
    Weibull,
    clip_percentiles,
)
from edith.statistics.estimators import RELIABLE_TRIALS, BatchStats
from edith.statistics.generators import GENERATORS, PointGenerator, stream
from edith.statistics.stopping import (
    STOPPING_RULES,
    ConfidenceIntervalTarget,
    FixedTrials,
    StandardErrorTarget,
)
from edith.targets import Nominal

_EXAMPLE = Path("examples/sol_invictus")


def test_normal_median_is_the_nominal_value() -> None:
    """The 50th percentile of a normal distribution is its nominal value."""
    assert Normal(2.0).ppf(np.array([0.5]), 10.0)[0] == pytest.approx(10.0)


def test_truncated_normal_stays_inside_its_bounds() -> None:
    """Percentiles from 0 to 1 give values between low and high."""
    values = Normal(1.0, low=9.0, high=10.5).ppf(np.linspace(0, 1, 101), 10.0)

    assert values.min() >= 9.0 - 1e-9
    assert values.max() <= 10.5 + 1e-9


def test_zero_sigma_keeps_the_nominal_value() -> None:
    """A normal distribution with no spread always gives the nominal value."""
    assert np.all(Normal(0.0).ppf(np.array([0.0, 0.3, 1.0]), 4.0) == 4.0)


def test_extreme_percentiles_give_finite_values() -> None:
    """Percentiles of exactly 0 and 1 are clipped, so no value is infinite."""
    extremes = np.array([0.0, 1.0])

    assert np.all(np.isfinite(Normal(1.0).ppf(extremes, 0.0)))
    assert np.all(np.isfinite(Normal(1.0, high=2.0).ppf(extremes, 0.0)))
    assert clip_percentiles(extremes).tolist() == [1e-6, 1 - 1e-6]


def test_uniform_ppf_spans_low_to_high() -> None:
    """A uniform distribution maps the percentiles linearly onto its range."""
    values = Uniform(2.0, 4.0).ppf(np.array([0.0, 0.5, 1.0]), 3.0)

    assert values.tolist() == [2.0, 3.0, 4.0]  # The ends are exact, with no clip


@pytest.mark.parametrize("name", ["LatinHypercubeGenerator", "RandomGenerator"])
def test_generators_are_reproducible(name: str) -> None:
    """The same seed, batch and key give the same percentiles."""
    first = GENERATORS[name]().percentiles(10, stream(1, 0, "a"))
    second = GENERATORS[name]().percentiles(10, stream(1, 0, "a"))
    other_batch = GENERATORS[name]().percentiles(10, stream(1, 1, "a"))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_batch)


def test_lhs_puts_one_point_in_each_stratum() -> None:
    """Latin hypercube percentiles fill every tenth of [0, 1] once."""
    percentiles = GENERATORS["LatinHypercubeGenerator"]().percentiles(
        10, stream(3, 0, "a")
    )

    assert sorted(np.floor(percentiles * 10).astype(int)) == list(range(10))


def test_streams_are_independent_of_other_keys() -> None:
    """One dispersion's percentiles do not depend on which others exist."""
    generator = GENERATORS["LatinHypercubeGenerator"]()
    alone = generator.percentiles(8, stream(5, 2, "mass"))
    with_another = [
        generator.percentiles(8, stream(5, 2, key)) for key in ("wind", "mass")
    ]

    assert np.array_equal(alone, with_another[1])


def test_batch_statistics() -> None:
    """The standard error and interval come from the spread of batch means."""
    stats = BatchStats(independent_trials=False)
    assert stats.standard_error is None
    stats.add_batch([1.0, 3.0])  # mean 2
    assert stats.standard_error is None  # One batch is not enough
    assert stats.confidence_interval_half_width(0.95) is None
    stats.add_batch([3.0, 5.0])  # mean 4
    stats.add_batch([5.0, 7.0])  # mean 6

    assert (stats.batches, stats.trials, stats.mean) == (3, 6, 4.0)
    assert stats.standard_error == pytest.approx(2.0 / math.sqrt(3))
    assert stats.confidence_interval_half_width(0.95) == pytest.approx(
        4.302653 * 2.0 / math.sqrt(3)
    )


def test_stratified_trials_need_a_second_batch_for_a_standard_error() -> None:
    """With one batch of stratified trials there is no standard error, only a note."""
    stats = BatchStats(independent_trials=False)
    stats.add_batch([1.0, 2.0, 3.0, 4.0])

    assert stats.mean == 2.5
    before = stats.requirement
    stats.add_batch([2.0, 3.0, 4.0, 5.0])

    assert before == "Requires additional batches for standard error"
    assert stats.requirement is None
    assert stats.confidence_interval_half_width(0.95) is not None


def test_independent_trials_give_a_standard_error_from_one_batch() -> None:
    """Independent trials are pooled, so their spread is enough, and batches add up."""
    stats = BatchStats(independent_trials=True)
    stats.add_batch([1.0])
    assert stats.standard_error is None
    assert stats.requirement == "Requires at least two trials for standard error"
    stats.add_batch([3.0])  # Two trials, from two batches of one

    assert (stats.batches, stats.trials, stats.mean) == (2, 2, 2.0)
    assert stats.standard_error == pytest.approx(1.0)  # s = sqrt(2), n = 2
    assert stats.confidence_interval_half_width(0.95) == pytest.approx(12.706205)

    one_batch = BatchStats(independent_trials=True)
    one_batch.add_batch([1.0, 3.0, 5.0, 7.0])
    assert one_batch.standard_error == pytest.approx(math.sqrt(20 / 3) / 2.0)


def test_generators_say_whether_their_trials_are_independent() -> None:
    """Random draws are independent; stratified Latin hypercube draws are not."""
    assert GENERATORS["RandomGenerator"]().trials_are_independent
    assert not GENERATORS["LatinHypercubeGenerator"]().trials_are_independent


def test_a_single_trial_gives_no_standard_error() -> None:
    """With independent trials the standard error needs two of them."""
    stats = BatchStats(independent_trials=True)
    stats.add_batch([1.0])

    assert stats.mean == 1.0  # The mean itself is always given
    assert stats.standard_error is None
    assert stats.confidence_interval_half_width(0.95) is None
    assert stats.requirement == "Requires at least two trials for standard error"
    assert not StandardErrorTarget(1e9).done(stats)
    stats.add_batch([2.0])
    standard_error = stats.standard_error
    assert standard_error is not None


def test_stratified_trials_need_two_batches() -> None:
    """The standard error is the spread of the batch means, so it needs two."""
    stats = BatchStats(independent_trials=False)
    stats.add_batch([1.0, 2.0])
    first = stats.requirement
    stats.add_batch([2.0, 3.0])

    assert first == "Requires additional batches for standard error"
    assert stats.requirement is None


def test_fewer_than_the_reliable_number_of_trials_warns() -> None:
    """Under RELIABLE_TRIALS the output might not be reliable; at it, no warning."""
    stats = BatchStats(independent_trials=True)
    stats.add_batch([float(i) for i in range(RELIABLE_TRIALS - 1)])
    few = stats.warning
    stats.add_batch([0.0])

    assert few is not None
    assert "might not be reliable" in few
    assert stats.warning is None


def test_a_generator_must_define_its_abstract_members() -> None:
    """A subclass without percentiles or the independence flag is refused at once."""
    with pytest.raises(TypeError, match="percentiles, trials_are_independent"):
        type("Hollow", (PointGenerator,), {})


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_a_generators_independence_claim_matches_its_draws(name: str) -> None:
    """A mean of n uniform draws has variance 1/(12 n) only if they are independent."""
    generator = GENERATORS[name]()
    n = 20
    means = [
        generator.percentiles(n, np.random.default_rng(i)).mean() for i in range(400)
    ]
    ratio = float(np.var(means, ddof=1)) * 12 * n

    if generator.trials_are_independent:
        assert 0.75 < ratio < 1.25
    else:
        assert ratio < 0.5  # Stratified draws have a far steadier mean


def test_the_mean_is_that_of_all_the_trials_even_if_batches_differ() -> None:
    """A batch that lost trials counts for the trials it has, not as a whole batch."""
    stats = BatchStats(independent_trials=False)
    stats.add_batch([1.0, 2.0, 3.0])
    stats.add_batch([10.0])

    assert stats.mean == 4.0  # Not 6, the mean of the batch means 2 and 10


def test_a_rule_waits_while_there_is_no_standard_error() -> None:
    """With too little data neither target rule is met, however loose it is."""
    stats = BatchStats(independent_trials=False)
    stats.add_batch([10.0, 11.0])

    assert not StandardErrorTarget(1e9).done(stats)
    assert not ConfidenceIntervalTarget(1e9, 0.95).done(stats)


def _result(stats: BatchStats, tmp_path: Path) -> CampaignResult:
    """Return a campaign's result holding only its statistics."""
    return CampaignResult(
        stats, "all_batches", 0.95, tmp_path / "trials.csv", [], prepare_s=0.0
    )


def test_the_summary_says_what_a_missing_standard_error_needs(tmp_path: Path) -> None:
    """One Latin hypercube batch gives a note instead of an interval."""
    stats = BatchStats(independent_trials=False)
    stats.add_batch([3000.0, 3010.0])

    assert _result(stats, tmp_path).summary() == (
        "apogee 3005 m (2 trials in 1 batch). "
        "Requires additional batches for standard error"
    )


def test_the_summary_gives_an_interval_from_one_random_batch(tmp_path: Path) -> None:
    """One batch of independent trials already has a standard error."""
    stats = BatchStats(independent_trials=True)
    stats.add_batch([3000.0, 3010.0, 3020.0])

    summary = _result(stats, tmp_path).summary()

    assert summary.startswith("apogee 3010 +/- ")
    assert summary.endswith("m (95% CI, 3 trials in 1 batch)")


def test_stopping_rules() -> None:
    """Fixed never stops early; the others stop once their target is met."""
    stats = BatchStats(independent_trials=False)
    for mean in (10.0, 11.0, 9.0, 10.0):
        stats.add_batch([mean])

    assert set(STOPPING_RULES) == {
        "FixedTrials",
        "StandardErrorTarget",
        "ConfidenceIntervalTarget",
    }
    assert not FixedTrials().done(stats)
    assert StandardErrorTarget(1.0).done(stats)
    assert not StandardErrorTarget(0.1).done(stats)
    assert ConfidenceIntervalTarget(5.0, 0.95).done(stats)
    assert not ConfidenceIntervalTarget(0.5, 0.95).done(stats)


def _campaign(tmp_path: Path, num_batches: int = 2, stopping: str = "") -> Path:
    """Write a small campaign beside a copy of the example sim."""
    shutil.copytree(_EXAMPLE, tmp_path / "base", ignore=shutil.ignore_patterns("out"))
    path = tmp_path / "edith_config.toml"
    path.write_text(
        f"""
num_batches = {num_batches}
master_seed = 7
base_sim = "base/sim.toml"
output_dir = "out"
flight_per_batch = 2
generator = "LatinHypercubeGenerator"
{stopping}
[output]
plots = []

[dispersions."config.truth.launch_rail.elevation"]
distribution = "Normal"
sigma = "0.5 deg"
high = "90 deg"

[dispersions."properties.dry_mass_properties.mass"]
distribution = "Uniform"
half_width = "0.25 kg"
""",
        encoding="utf-8",
    )
    return path


def _fly(config: EdithConfig) -> tuple[pd.DataFrame, CampaignResult]:
    """Prepare and fly a campaign on two workers, returning its trials and result."""
    with (
        Workers(2, config.output_dir / "workers") as workers,
        closing(prepare_batches(config)) as batches,
    ):
        result = fly_campaign(config, workers, batches)
    return pd.read_csv(result.trials_path), result


def test_a_small_campaign_flies_and_records_every_trial(tmp_path: Path) -> None:
    """Four trials in two batches give four finite apogees and their folders."""
    config = EdithConfig.from_file(_campaign(tmp_path))

    frame, result = _fly(config)

    assert len(frame) == 4
    assert frame["status"].eq("OK").all()
    assert np.isfinite(frame["apogee"]).all()
    assert (frame["apogee"] > 100).all()
    assert list(frame["batch"]) == [0, 0, 1, 1]
    assert "properties.dry_mass_properties.mass" in frame.columns
    assert (result.stats.batches, result.stats.trials) == (2, 4)
    assert result.stopped_by == "all_batches"
    assert (config.output_dir / "runs" / "0003" / "run.toml").is_file()
    assert "apogee" in result.summary()


def test_a_campaign_stops_early_when_its_target_is_met(tmp_path: Path) -> None:
    """A loose standard-error target ends the campaign after two batches."""
    stopping = '[stopping]\nrule = "StandardErrorTarget"\ntarget = "100000 m"\n'
    config = EdithConfig.from_file(
        _campaign(tmp_path, num_batches=10, stopping=stopping)
    )

    frame, result = _fly(config)

    assert result.stopped_by == "rule"
    assert result.stats.batches == 2
    assert len(frame) == 4
    assert len(result.results) == 4
    runs = config.output_dir / "runs"
    assert sorted(path.name for path in runs.iterdir()) == [
        "0000",
        "0001",
        "0002",
        "0003",
    ]


def test_distributions_are_subclasses_of_the_base() -> None:
    """Every distribution a campaign may name is a Distribution, found by name."""
    assert set(DISTRIBUTIONS) == {
        "Beta",
        "LogNormal",
        "Normal",
        "StudentT",
        "Triangular",
        "Uniform",
        "VonMises",
        "Weibull",
    }
    assert all(issubclass(cls, Distribution) for cls in DISTRIBUTIONS.values())


def test_the_distribution_base_cannot_be_used() -> None:
    """The base has abstract methods, so it cannot be instantiated."""
    with pytest.raises(TypeError):
        Distribution()  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated


def test_a_distribution_must_define_its_class_variables() -> None:
    """Defining a subclass without REQUIRED, OPTIONAL or USAGE fails, unregistered."""
    with pytest.raises(TypeError, match="Lacking must define REQUIRED"):
        type("Lacking", (Distribution,), {})

    assert "Lacking" not in DISTRIBUTIONS


@pytest.mark.parametrize(
    ("required", "optional", "usage", "message"),
    [
        (["sigma"], (), "takes sigma", "REQUIRED must be a tuple"),
        (("sigma",), (3,), "takes sigma", "OPTIONAL must be a tuple"),
        (("sigma",), (), "", "USAGE must be text"),
        (("sigma",), ("sigma",), "takes sigma", "both required and optional"),
    ],
)
def test_a_distributions_class_variables_are_checked(
    required: object, optional: object, usage: str, message: str
) -> None:
    """Class variables of the wrong kind are refused when the class is defined."""
    with pytest.raises(TypeError, match=message):
        type(
            "Badly",
            (Distribution,),
            {"REQUIRED": required, "OPTIONAL": optional, "USAGE": usage},
        )

    assert "Badly" not in DISTRIBUTIONS


def test_distributions_describe_themselves() -> None:
    """Each distribution says what it is, in the nominal value's units."""
    assert Normal(0.25, low=16.0).describe("kg") == "normal, sigma 0.25 kg, low 16 kg"
    assert Uniform(0.0, 0.2).describe("") == "uniform from 0 to 0.2"
    assert StudentT(1.0, 4.0).describe("m") == "Student's t, scale 1 m, df 4"
    assert Triangular(0.0, 2.0).describe("m") == "triangular from 0 to 2 m"
    assert Weibull(2.0).describe("m/s") == "Weibull, shape 2"
    assert VonMises(400.0).describe("deg") == "von Mises, kappa 400"
    assert LogNormal(0.1).describe("kg") == "log-normal, sigma 0.1"
    assert Beta(2.0, 5.0, 0.0, 1.0).describe("m") == "beta 2, 5 from 0 to 1 m"


_VARIABLES = {"REQUIRED": ("a",), "OPTIONAL": (), "USAGE": "takes a"}


def test_a_distribution_must_define_its_abstract_methods() -> None:
    """A subclass without from_parameters is refused when it is defined."""
    methods = {"ppf": lambda self, p, n: p, "describe": lambda self, units: ""}

    with pytest.raises(TypeError, match="Hollow must define from_parameters"):
        type("Hollow", (Distribution,), {**_VARIABLES, **methods})

    assert "Hollow" not in DISTRIBUTIONS


def test_two_distributions_cannot_share_a_name() -> None:
    """A second class named Normal is refused, leaving the first registered."""
    methods = {
        "from_parameters": classmethod(lambda cls, parameters, nominal: cls()),
        "ppf": lambda self, p, n: p,
        "describe": lambda self, units: "",
    }

    with pytest.raises(TypeError, match="two distributions are named Normal"):
        type("Normal", (Distribution,), {**_VARIABLES, **methods})

    assert DISTRIBUTIONS["Normal"] is Normal


def test_messages_list_distributions_defined_after_the_module_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The valid names in an error are those registered when it is raised."""
    monkeypatch.setitem(DISTRIBUTIONS, "Gamma", Normal)

    with pytest.raises(EdithConfigError, match='"Beta" or "Gamma"'):
        _distribution_name({}, "dispersions.mass")


_NOMINAL = Nominal(10.0, "kg", '"10 kg"')

# For each distribution, a base set of parameters and a change to each key, which
# must give a different distribution. Every key a class lists needs a change here
_KEY_CHANGES: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = {
    "Normal": [
        ({"sigma": 1.0, "low": 8.0, "high": 12.0}, {"sigma": 2.0}),
        ({"sigma": 1.0, "low": 8.0, "high": 12.0}, {"low": 7.0}),
        ({"sigma": 1.0, "low": 8.0, "high": 12.0}, {"high": 13.0}),
    ],
    "Uniform": [
        ({"half_width": 1.0}, {"half_width": 2.0}),
        ({"low": 8.0, "high": 12.0}, {"low": 7.0}),
        ({"low": 8.0, "high": 12.0}, {"high": 13.0}),
    ],
    "StudentT": [
        ({"scale": 1.0, "df": 4.0}, {"scale": 2.0}),
        ({"scale": 1.0, "df": 4.0}, {"df": 5.0}),
    ],
    "Triangular": [
        ({"low": 8.0, "high": 12.0}, {"low": 7.0}),
        ({"low": 8.0, "high": 12.0}, {"high": 13.0}),
    ],
    "Weibull": [({"shape": 2.0}, {"shape": 3.0})],
    "VonMises": [({"kappa": 4.0}, {"kappa": 5.0})],
    "LogNormal": [({"sigma": 0.1}, {"sigma": 0.2})],
    "Beta": [
        ({"alpha": 2.0, "beta": 3.0, "low": 8.0, "high": 12.0}, {"alpha": 3.0}),
        ({"alpha": 2.0, "beta": 3.0, "low": 8.0, "high": 12.0}, {"beta": 4.0}),
        ({"alpha": 2.0, "beta": 3.0, "low": 8.0, "high": 12.0}, {"low": 7.0}),
        ({"alpha": 2.0, "beta": 3.0, "low": 8.0, "high": 12.0}, {"high": 13.0}),
    ],
}

# The nominal value of a distribution that cannot take the mass above
_NOMINALS = {"VonMises": Nominal(10.0, "deg", '"10 deg"')}


def _parameters(values: dict[str, float]) -> dict[str, Parameter]:
    """Return parameters as the dispersion passes them, in the nominal's units."""
    return {key: Parameter(value, f"{value:g} kg") for key, value in values.items()}


@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_every_key_a_distribution_lists_changes_its_result(name: str) -> None:
    """A listed key that from_parameters ignores would do nothing in a campaign."""
    cls = DISTRIBUTIONS[name]
    changes = _KEY_CHANGES[name]

    tested = {key for _, change in changes for key in change}
    assert tested == {*cls.REQUIRED, *cls.OPTIONAL}, "every listed key needs a change"
    nominal = _NOMINALS.get(name, _NOMINAL)
    for base, change in changes:
        moved = cls.from_parameters(_parameters({**base, **change}), nominal)
        assert moved != cls.from_parameters(_parameters(base), nominal)


def _built(name: str) -> Distribution:
    """Return a distribution of the first parameters ``_KEY_CHANGES`` lists for it."""
    base, _ = _KEY_CHANGES[name][0]
    return DISTRIBUTIONS[name].from_parameters(
        _parameters(base), _NOMINALS.get(name, _NOMINAL)
    )


@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_every_distribution_gives_finite_increasing_values(name: str) -> None:
    """The inverse CDF is finite at percentiles 0 and 1 and never decreases."""
    nominal = _NOMINALS.get(name, _NOMINAL).value
    values = _built(name).ppf(np.linspace(0.0, 1.0, 201), nominal)

    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) >= 0.0)


@pytest.mark.parametrize(
    ("distribution", "percentile", "expected"),
    [
        (StudentT(2.0, 4.0), 0.5, 10.0),  # Centred on the nominal value
        (LogNormal(0.3), 0.5, 10.0),  # The nominal value is the median
        (Weibull(2.0), 1.0 - math.exp(-1.0), 10.0),  # And the scale here
        (Triangular(8.0, 14.0), 1.0 / 3.0, 10.0),  # The mode: a third lies below
        (Beta(2.0, 2.0, 8.0, 12.0), 0.5, 10.0),  # Symmetric about the middle
    ],
)
def test_the_nominal_value_sits_where_each_distribution_says(
    distribution: Distribution, percentile: float, expected: float
) -> None:
    """Each distribution places the nominal value as its docstring describes."""
    value = distribution.ppf(np.array([percentile]), 10.0)[0]

    assert value == pytest.approx(expected)


def test_bounded_distributions_stay_inside_their_bounds() -> None:
    """Triangular and beta values lie from low to high, von Mises within a half turn."""
    percentiles = np.linspace(0.0, 1.0, 101)

    for distribution in (Triangular(8.0, 12.0), Beta(0.5, 0.5, 8.0, 12.0)):
        values = distribution.ppf(percentiles, 10.0)
        assert values.min() == pytest.approx(8.0)
        assert values.max() == pytest.approx(12.0)
    angles = VonMises(0.5, math.pi / 180.0).ppf(percentiles, 10.0)
    assert angles.min() >= 10.0 - 180.0 - 1e-6
    assert angles.max() <= 10.0 + 180.0 + 1e-6


def test_a_von_mises_spread_is_in_radians_whatever_the_angles_units() -> None:
    """A large kappa gives a near-normal spread of 1 / sqrt(kappa) radians."""
    distribution = VonMises.from_parameters(
        _parameters({"kappa": 400.0}), Nominal(10.0, "deg", '"10 deg"')
    )
    quartiles = distribution.ppf(np.array([0.25, 0.75]), 10.0)

    sigma_deg = math.degrees(1.0 / math.sqrt(400.0))
    assert quartiles[1] - quartiles[0] == pytest.approx(1.349 * sigma_deg, rel=0.01)


@pytest.mark.parametrize(
    ("name", "values", "nominal", "message"),
    [
        ("StudentT", {"scale": -1.0, "df": 4.0}, 10.0, "expected 0 or more"),
        ("StudentT", {"scale": 1.0, "df": 0.0}, 10.0, "expected above 0"),
        ("Triangular", {"low": 11.0, "high": 12.0}, 10.0, "is above the nominal"),
        ("Triangular", {"low": 12.0, "high": 11.0}, 10.0, "must be below high"),
        ("Weibull", {"shape": 2.0}, 0.0, "needs a nominal value above 0"),
        ("Weibull", {"shape": -2.0}, 10.0, "expected above 0"),
        ("VonMises", {"kappa": 0.0}, 10.0, "expected above 0"),
        ("LogNormal", {"sigma": 0.1}, -1.0, "needs a nominal value above 0"),
        ("LogNormal", {"sigma": -0.1}, 10.0, "expected 0 or more"),
        (
            "Beta",
            {"alpha": 0.0, "beta": 3.0, "low": 8.0, "high": 12.0},
            10.0,
            "expected above 0",
        ),
        (
            "Beta",
            {"alpha": 2.0, "beta": 3.0, "low": 8.0, "high": 9.0},
            10.0,
            "is below the nominal",
        ),
    ],
)
def test_bad_parameters_of_the_new_distributions_are_refused(
    name: str, values: dict[str, float], nominal: float, message: str
) -> None:
    """Each distribution checks its own parameters against the nominal value."""
    with pytest.raises(ParameterError, match=message):
        DISTRIBUTIONS[name].from_parameters(
            _parameters(values), Nominal(nominal, "", f"{nominal:g}")
        )


def test_a_von_mises_needs_an_angle() -> None:
    """A von Mises distribution of a mass is refused."""
    with pytest.raises(ParameterError, match="is not an angle"):
        VonMises.from_parameters(_parameters({"kappa": 4.0}), _NOMINAL)


def test_a_plain_key_must_be_one_the_distribution_takes() -> None:
    """PLAIN naming a key the class does not take is refused at definition."""
    with pytest.raises(TypeError, match="as plain but does not take it"):
        type("Stray", (Distribution,), {**_VARIABLES, "PLAIN": ("b",)})

    assert "Stray" not in DISTRIBUTIONS
