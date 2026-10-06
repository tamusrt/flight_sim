"""Tests of the EDITH Monte Carlo system: statistics, inputs, checks, a small batch."""

# ruff: noqa: E501

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from flight_sim.edith import failures
from flight_sim.edith import stats as ci
from flight_sim.edith.batch import (
    TARGET_HALF_WIDTH,
    WATCHED,
    Settings,
    _close_to_a_limit,
    _Pool,
    _reco_check,
    _run_seed,
    _sobol,
    _worst_headline,
    run_batch,
    summarize,
)
from flight_sim.edith.inputs import (
    DIMENSIONS,
    INPUTS,
    Nominal,
    SiteConfig,
    describe,
    draw,
)
from flight_sim.edith.report import format_report
from flight_sim.edith.run import Rocket, RocketSpec
from flight_sim.edith.surrogate import FEATURES, TARGETS, Surrogate
from flight_sim.environment.wind import LayeredWind

_NOMINAL = Nominal(
    wind_speed_m_s=5.0,
    wind_azimuth_deg=270.0,
    rail_elevation_deg=85.0,
    rail_azimuth_deg=90.0,
    pad_temperature_k=300.0,
    pad_pressure_pa=90000.0,
    pad_elevation_m=1400.0,
)
_MIDDLE = np.full(DIMENSIONS, 0.5)


# ----- statistics ---------------------------------------------------------------


def test_wilson_is_centred_and_inside_zero_and_one() -> None:
    """A 90% Wilson interval holds the estimate and stays a probability."""
    interval = ci.wilson(30, 100)
    assert interval.estimate == pytest.approx(0.3)
    assert interval.low < 0.3 < interval.high
    assert interval.high - interval.low < 0.2
    assert 0.0 <= ci.wilson(0, 10).low <= ci.wilson(0, 10).high <= 1.0
    assert math.isnan(ci.wilson(0, 0).estimate)


def test_an_event_never_seen_still_has_an_upper_bound() -> None:
    """Zero failures in n runs: the exact bound is about 3/n at 95%, 2.3/n at 90%."""
    exact = ci.clopper_pearson(0, 100)
    assert exact.low == 0.0
    assert exact.high == pytest.approx(0.0295, abs=0.002)
    shown = ci.proportion(np.zeros(100, dtype=bool))
    assert shown.low == 0.0
    assert shown.high >= exact.high - 1e-9
    everything = ci.proportion(np.ones(50, dtype=bool))
    assert everything.high == 1.0 and everything.low < 1.0


def test_independent_sets_can_widen_a_proportion() -> None:
    """Sets that disagree with each other widen the interval beyond the pooled one."""
    flags = np.repeat([True, False, True, False, True, False, True, False], 32)
    sets = np.repeat(np.arange(8), 32)
    pooled = ci.proportion(flags)
    with_sets = ci.proportion(flags, sets)
    assert with_sets.low <= pooled.low and with_sets.high >= pooled.high
    assert with_sets.half_width > pooled.half_width


def test_widen_stays_inside_zero_and_one() -> None:
    """Added uncertainty never leaves the 0 to 1 range."""
    wide = ci.widen(ci.Interval(0.02, 0.0, 0.05), 0.1)
    assert wide.low == 0.0 and wide.high == pytest.approx(0.15)
    assert ci.widen(ci.Interval(0.99, 0.95, 1.0), 0.2).high == 1.0


def test_a_90_percent_mean_interval_covers_the_true_mean_about_90_percent_of_the_time() -> (
    None
):
    """Over many samples from a known curve, the interval holds the true mean in most of them."""
    rng = np.random.default_rng(21)
    hits = 0
    trials = 150
    for _ in range(trials):
        sample = rng.normal(10.0, 2.0, 60)
        interval = ci.bootstrap(sample, np.mean, rng, resamples=200)
        hits += interval.low <= 10.0 <= interval.high
    assert 0.80 <= hits / trials <= 0.97  # nominal 90%, with room for the 150 trials


def test_a_wilson_interval_covers_the_true_chance_about_90_percent_of_the_time() -> (
    None
):
    """The same check for a yes/no chance of 30% from 100 flights."""
    rng = np.random.default_rng(22)
    hits = 0
    trials = 400
    for _ in range(trials):
        events = int(rng.binomial(100, 0.3))
        interval = ci.wilson(events, 100)
        hits += interval.low <= 0.3 <= interval.high
    assert 0.86 <= hits / trials <= 0.95


def test_the_high_end_of_an_interval_is_a_one_sided_bound_at_a_higher_confidence() -> (
    None
):
    """A two-sided 90% interval's high end is "95% sure it is below"."""
    assert ci.one_sided(0.90) == pytest.approx(0.95)
    assert ci.one_sided(0.80) == pytest.approx(0.90)
    # the exact bound for zero events in 100 flights: 95% sure it is below about 2.95%
    assert ci.clopper_pearson(0, 100).high == pytest.approx(
        1.0 - 0.05 ** (1.0 / 100.0), abs=0.002
    )


def test_a_watched_chance_can_reach_the_default_target_with_512_flights() -> None:
    """With 512 flights a 90% interval is narrower than plus or minus 4% for any chance
    (about 3.6% at 50%), which plus or minus 3% never was between 30% and 70%."""
    assert TARGET_HALF_WIDTH == 0.04 and Settings().target_half_width == 0.04
    for p in (0.3, 0.5, 0.7):
        half = ci.wilson(round(512 * p), 512).half_width
        assert 0.03 < half <= 0.04
    assert set(WATCHED) == {
        "probability_any_failure",
        "probability_apogee_at_least_target",
    }


def test_the_stop_rule_looks_at_the_widest_watched_chance() -> None:
    """Only the two watched chances decide; the widest one counts."""
    summary = {
        "probability_any_failure": {"low": 0.30, "high": 0.38},  # half-width 0.04
        "probability_apogee_at_least_target": {"low": 0.45, "high": 0.50},
        "probability_any_warning": {"low": 0.0, "high": 1.0},  # not watched
    }
    assert _worst_headline(summary) == pytest.approx(0.04)


def test_bootstrap_intervals_cover_the_truth_and_repeat() -> None:
    """The bootstrap mean interval holds the sample mean and is reproducible."""
    values = np.random.default_rng(1).normal(10.0, 2.0, 400)
    first = ci.mean_interval(values, np.random.default_rng(5))
    again = ci.mean_interval(values, np.random.default_rng(5))
    assert first == again
    assert first.low < values.mean() < first.high
    assert first.half_width == pytest.approx(1.645 * 2.0 / 20.0, rel=0.3)
    median = ci.percentile_interval(values, 50, np.random.default_rng(5))
    assert median.low < median.estimate < median.high
    assert ci.mean_interval(np.array([3.0]), np.random.default_rng(0)).low == 3.0
    assert math.isnan(ci.mean_interval(np.array([]), np.random.default_rng(0)).low)


def test_interval_dict_is_plain_numbers() -> None:
    """The report file stores intervals as plain numbers."""
    as_dict = ci.Interval(0.123456, 0.1, 0.2).as_dict(3)
    assert as_dict == {
        "estimate": 0.123,
        "low": 0.1,
        "high": 0.2,
        "confidence": 0.9,
    }
    json.dumps(as_dict)


# ----- the inputs ---------------------------------------------------------------


def test_the_site_file_round_trips_and_rejects_unknown_keys(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Edited settings come back as written; a misspelt key is an error."""
    site = SiteConfig(thrust_sd=0.08, wind_launch_limit_m_s=9.0)
    path = tmp_path / "site.json"
    site.save(path)
    assert SiteConfig.load(path) == site
    assert "placeholder" in site.note
    data = json.loads(path.read_text(encoding="utf-8"))
    data["thrust_sigma"] = 0.1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="thrust_sigma"):
        SiteConfig.load(path)


def test_the_middle_of_the_cube_is_the_nominal_rocket() -> None:
    """Unit 0.5 is the middle of every bell curve."""
    v = draw(_MIDDLE, SiteConfig(), _NOMINAL, seed=1)
    assert v.thrust == pytest.approx(1.0)
    assert v.mass == pytest.approx(1.0)
    assert v.drag == pytest.approx(1.0)
    assert v.rail_elevation_deg == pytest.approx(85.0)
    assert v.temperature_k == pytest.approx(300.0)
    assert v.pressure_pa == pytest.approx(90000.0)
    assert v.wind_azimuth_deg == pytest.approx(270.0)
    assert v.canopy_cd == pytest.approx(1.0)
    assert v.seed == 1


def test_draws_follow_the_bell_curves() -> None:
    """Draws have the asked-for spread; the wind is Weibull, cut at the limit."""
    site = SiteConfig(wind_launch_limit_m_s=11.0)
    units = np.random.default_rng(3).random((4000, DIMENSIONS))
    drawn = [draw(u, site, _NOMINAL, seed=0) for u in units]
    thrust = np.array([v.thrust for v in drawn])
    temperature = np.array([v.temperature_k for v in drawn])
    wind = np.array([v.wind_speed_m_s for v in drawn])
    assert thrust.std() == pytest.approx(site.thrust_sd, rel=0.1)
    assert temperature.mean() == pytest.approx(300.0, abs=0.3)
    assert temperature.std() == pytest.approx(site.temperature_sd_k, rel=0.1)
    assert wind.max() <= 11.0
    # the stated mean is the mean of the days that are flown (the cut curve)
    assert wind.mean() == pytest.approx(_NOMINAL.wind_speed_m_s, rel=0.05)
    # a Weibull is skewed: the mean is above the median
    assert wind.mean() > np.median(wind)


def _flown_wind_mean(site: SiteConfig, nominal: Nominal = _NOMINAL) -> float:
    """The mean of the wind over the whole flown curve (an even grid on its quantiles)."""
    n = 1500
    speeds = []
    for u in (np.arange(n) + 0.5) / n:
        point = _MIDDLE.copy()
        point[INPUTS.index("wind_speed")] = u
        speeds.append(draw(point, site, nominal, 0).wind_speed_m_s)
    return float(np.mean(speeds))


@pytest.mark.parametrize("mean", [3.0, 5.0, 7.15])
def test_the_mean_wind_of_the_flown_days_is_the_stated_mean(mean: float) -> None:
    """Winds over the limit are not flown, but the stated mean is still the mean of the days
    that are flown (within 1%); SRT14's 'average' 7.15 m/s used to come out as 5.98."""
    site = SiteConfig(wind_mean_m_s=mean, wind_launch_limit_m_s=11.0)
    assert _flown_wind_mean(site) == pytest.approx(mean, rel=0.01)


def test_the_wind_mean_comes_from_the_rocket_profile_when_the_site_has_none() -> None:
    """With no mean in the site file, the profile's wind (5 m/s here) is the flown mean."""
    assert _flown_wind_mean(SiteConfig()) == pytest.approx(5.0, rel=0.01)


def test_the_wind_description_states_the_flown_mean_honestly() -> None:
    """The report says the mean is of the days flown, and says so when the mean is out of reach."""
    text = describe(SiteConfig(wind_mean_m_s=7.15), _NOMINAL)[0]["numbers"]
    assert "mean 7.15 m/s on the days it flies" in text and "not flown" in text
    # a Rayleigh curve cut at 11 m/s cannot have a mean over 7.33 m/s (two thirds of 11)
    text = describe(SiteConfig(wind_mean_m_s=8.05), _NOMINAL)[0]["numbers"]
    assert "mean 7.33" in text and "out of reach" in text


def test_a_vertical_rail_is_not_a_pile_of_runs_exactly_on_vertical() -> None:
    """The tilt is folded at vertical: no run sits on the limit, and about half lean the other way."""
    vertical = Nominal(5.0, 270.0, 90.0, 90.0, 300.0, 90000.0, 1400.0)
    site = SiteConfig()
    units = np.random.default_rng(4).random((4000, DIMENSIONS))
    drawn = [draw(u, site, vertical, seed=0) for u in units]
    elevation = np.array([v.rail_elevation_deg for v in drawn])
    heading = np.array([v.rail_azimuth_deg for v in drawn])
    assert (elevation <= 90.0).all()
    assert (elevation == 90.0).mean() < 0.001, (
        "clipping would put about half exactly here"
    )
    tilt = 90.0 - elevation
    # a half-normal tilt: mean sd * sqrt(2 / pi)
    assert tilt.mean() == pytest.approx(
        site.rail_elevation_sd_deg * math.sqrt(2 / math.pi), rel=0.1
    )
    leaning_back = (heading > 180.0).mean()  # heading 90 +- 1, or 270 +- 1 when flipped
    assert leaning_back == pytest.approx(0.5, abs=0.05)


def test_each_run_gets_its_own_wind_layers() -> None:
    """The layer phases differ between runs, so the wind profile does too."""
    a = draw(np.random.default_rng(1).random(DIMENSIONS), SiteConfig(), _NOMINAL, 0)
    b = draw(np.random.default_rng(2).random(DIMENSIONS), SiteConfig(), _NOMINAL, 0)
    assert isinstance(a.wind(1400.0), LayeredWind)
    assert a.speed_phases != b.speed_phases
    profile_a = a.mean_wind(1400.0, 9000.0)
    assert profile_a.shape == (2,)
    assert not np.allclose(profile_a, b.mean_wind(1400.0, 9000.0))


def test_the_inputs_are_all_described() -> None:
    """The report lists what was varied and how."""
    rows = describe(SiteConfig(), _NOMINAL)
    assert rows and all("input" in r and "numbers" in r for r in rows)
    assert len(INPUTS) == DIMENSIONS


# ----- the failure checks ---------------------------------------------------------


def _good() -> dict[str, object]:
    return {
        "apogee_m": 9000.0,
        "rail_v": 40.0,
        "rail_margin": 2.5,
        "margin_lo": 2.0,
        "margin_hi": 4.0,
        "drogue_v": 30.0,
        "main_alt": 400.0,
        "land_v_vert": 6.0,
        "load_ratio": 0.3,
        "separated": True,
        "all_open": True,
        "apogee_ok": True,
        "landed": True,
        "sim_ok": True,
    }


def test_a_good_run_triggers_nothing() -> None:
    """The nominal numbers pass every check."""
    good = _good()
    assert not failures.any_failure(good)
    assert not failures.any_warning(good)
    assert not any(failures.statuses(good).values())


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("rail_v", 14.0, "rail_exit_floor"),
        ("apogee_m", 6300.0, "apogee_window"),
        ("apogee_m", 11950.0, "apogee_window"),
        ("rail_margin", 1.2, "stability_rail"),
        ("margin_lo", 1.4, "stability_lowest"),
        ("load_ratio", 1.2, "canopy_overload"),
        ("separated", False, "separation"),
        ("all_open", False, "canopies_open"),
        ("apogee_ok", False, "reached_apogee"),
        ("landed", False, "landed"),
        ("sim_ok", False, "simulation"),
    ],
)
def test_each_red_check_fails_a_run(key: str, value: object, check: str) -> None:
    """Every IREC red check, and the other ways to lose a flight, fail the run."""
    result = _good()
    result[key] = value
    assert failures.any_failure(result)
    assert failures.statuses(result)[check]


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("rail_v", 25.0, "rail_exit_recommended"),
        ("rail_margin", 4.5, "stability_static_max"),
        ("margin_hi", 6.5, "stability_dynamic_max"),
        ("drogue_v", 45.0, "drogue_rate"),
        ("drogue_v", 15.0, "drogue_rate"),
        ("main_alt", 470.0, "main_altitude"),
    ],
)
def test_each_amber_check_warns_without_failing(
    key: str, value: object, check: str
) -> None:
    """The recommended-limit rows warn but do not fail the run."""
    result = _good()
    result[key] = value
    assert failures.any_warning(result)
    assert not failures.any_failure(result)
    assert failures.statuses(result)[check]


def test_a_missing_optional_number_is_not_applicable() -> None:
    """A rocket with no drogue has no drogue rate to check (and none deployed has no main altitude)."""
    result = _good()
    result["drogue_v"] = None
    assert not failures.statuses(result)["drogue_rate"]
    result["main_alt"] = None
    assert not failures.statuses(result)["main_altitude"]
    del result["drogue_v"]
    assert not failures.any_failure(result) and not failures.any_warning(result)
    assert failures.missing_values(result) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "gone"])
@pytest.mark.parametrize(
    ("key", "check"),
    [
        ("rail_v", "rail_exit_floor"),
        ("rail_margin", "stability_rail"),
        ("margin_lo", "stability_lowest"),
        ("load_ratio", "canopy_overload"),
        ("apogee_m", "apogee_window"),
    ],
)
def test_a_required_number_that_is_missing_or_not_a_number_fails_its_check(
    key: str, check: str, bad: object
) -> None:
    """NaN, infinity or a missing number on a flight that finished is a failed check, never a pass."""
    result = _good()
    if bad == "gone":
        del result[key]
    else:
        result[key] = bad
    assert failures.statuses(result)[check]
    assert failures.any_failure(result)
    assert f"result missing: {key}" in failures.reasons(result)


@pytest.mark.parametrize(
    "key", ["separated", "all_open", "apogee_ok", "landed", "sim_ok"]
)
def test_a_missing_yes_or_no_value_fails_too(key: str) -> None:
    """A True/False result that is not there is not a pass."""
    result = _good()
    result[key] = None
    assert failures.any_failure(result)


def test_a_run_that_never_climbed_or_errored_is_not_asked_for_its_descent() -> None:
    """No apogee: only the climb's numbers are expected, and the apogee window and
    'reached apogee' fail. An error run fails only the simulation check."""
    never = {
        "sim_ok": True,
        "apogee_ok": False,
        "rail_v": 40.0,
        "rail_margin": 2.5,
        "margin_lo": 2.0,
        "margin_hi": 4.0,
    }
    assert failures.missing_values(never) == []
    triggered = {k for k, v in failures.statuses(never).items() if v}
    assert triggered == {"apogee_window", "reached_apogee"}
    errored = {"sim_ok": False, "error": "RuntimeError: boom"}
    assert failures.missing_values(errored) == []
    assert {k for k, v in failures.statuses(errored).items() if v} == {"simulation"}


def test_closeness_counts_in_tolerances() -> None:
    """Closeness is the distance to the nearest limit in units of its tolerance."""
    result = _good()
    far = failures.closeness(result)
    assert far > 1.0
    result["main_alt"] = 455.7  # 1.5 m under the limit, tolerance 3
    assert failures.closeness(result) == pytest.approx(0.5, abs=0.01)
    wide = failures.closeness(result, {"main_alt": 15.0})
    assert wide == pytest.approx(0.1, abs=0.01)
    extra = failures.closeness(_good(), None, (("margin_hi", 4.0, 1.0),))
    assert extra == pytest.approx(0.0, abs=1e-9)


# ----- the climb surrogate ----------------------------------------------------------


def _synthetic(n: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, FEATURES))
    payloads = []
    for row in x:
        payloads.append(
            {
                "position": [
                    # curved, not just a plane: a product and a square of the inputs
                    9000.0
                    + 300.0 * row[0]
                    - 200.0 * row[1]
                    + 50.0 * row[7]
                    + 40.0 * row[0] * row[1]
                    + 30.0 * row[0] ** 2,
                    100.0 * row[7],
                    100.0 * row[8],
                ],
                "time": 30.0 + 2.0 * row[0],
                "velocity": [0.0, 1.0 * row[7], 1.0 * row[8]],
                "rail_v": 40.0 + 3.0 * row[0] - row[1],
                "rail_margin": 2.5 + 0.1 * row[2],
                "margin_lo": 2.0 + 0.1 * row[2],
                "margin_hi": 4.0 + 0.2 * row[3],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "rates": [0.0, 0.0, 0.0],
            }
        )
    return x, payloads


def _curved_truth(probe: np.ndarray) -> float:
    return float(
        9000.0
        + 300.0 * probe[0]
        - 200.0 * probe[1]
        + 50.0 * probe[7]
        + 40.0 * probe[0] * probe[1]
        + 30.0 * probe[0] ** 2
    )


def test_the_surrogate_learns_a_curved_climb_and_knows_its_own_error() -> None:
    """On a climb that is not a plane (a product and a square), the surrogate beats a
    straight-line fit on new points, and its reported held-out error matches the error
    seen on them."""
    x, payloads = _synthetic(120)
    surrogate = Surrogate(x, payloads)  # type: ignore[arg-type]
    assert set(surrogate.error) == set(TARGETS)
    rng = np.random.default_rng(8)
    probes = np.clip(rng.normal(size=(60, FEATURES)), -2.0, 2.0)
    errors = np.array(
        [surrogate.predict(p)["position"][0] - _curved_truth(p) for p in probes]
    )
    rms = float(np.sqrt(np.mean(errors**2)))
    # a least-squares plane through the same pilot points is the baseline to beat
    heights = np.array([q["position"][0] for q in payloads])  # type: ignore[index]
    design = np.c_[np.ones(len(x)), x]
    plane = np.linalg.lstsq(design, heights, rcond=None)[0]
    baseline = float(
        np.sqrt(
            np.mean(
                (
                    np.c_[np.ones(len(probes)), probes] @ plane
                    - [_curved_truth(p) for p in probes]
                )
                ** 2
            )
        )
    )
    assert rms < baseline
    assert surrogate.error["apogee_m"] == pytest.approx(rms, rel=0.5)
    guess = surrogate.predict(probes[0])
    assert guess["velocity"][0] == 0.0
    assert set(surrogate.check_tolerances()) == {
        "rail_v",
        "rail_margin",
        "margin_lo",
        "margin_hi",
    }


def test_the_surrogate_needs_enough_climbs() -> None:
    """Too few climbs is an error, not a poor fit."""
    x, payloads = _synthetic(FEATURES)
    with pytest.raises(ValueError, match="Too few"):
        Surrogate(x, payloads)  # type: ignore[arg-type]


# ----- sampling ----------------------------------------------------------------------


def test_sobol_rounds_are_balanced_and_independent() -> None:
    """Each round is a balanced set; different rounds are different scrambles."""
    first = _sobol(1, 0, 64)
    again = _sobol(1, 0, 64)
    other = _sobol(1, 1, 64)
    assert first.shape == (64, DIMENSIONS)
    assert np.array_equal(first, again)
    assert not np.allclose(first, other)
    assert np.allclose(first.mean(axis=0), 0.5, atol=0.03)
    assert _run_seed(1, 0, 3) == _run_seed(1, 0, 3)
    assert _run_seed(1, 0, 3) != _run_seed(1, 0, 4)


# ----- running -----------------------------------------------------------------------


_SPEC = RocketSpec("morpheus")


def test_one_or_two_workers_give_the_same_runs() -> None:
    """Results do not depend on how many cores share the work."""
    site = SiteConfig()
    unit = _sobol(11, 0, 8)
    jobs = [
        ("ascent", i, unit[i].tolist(), _run_seed(11, 0, i), None) for i in range(3)
    ]
    one = _Pool(_SPEC, site, 1)
    two = _Pool(_SPEC, site, 2)
    try:
        a, b = one.run(jobs), two.run(jobs)
    finally:
        one.close()
        two.close()
    assert [r[0] for r in a] == [r[0] for r in b] == [0, 1, 2]
    for (_, _, x), (_, _, y) in zip(a, b, strict=True):
        for key in ("apogee_m", "rail_v", "land_v_vert", "main_alt", "drift"):
            assert key in x and x[key] is not None, f"{key} is a key of a flown run"
            assert x[key] == y[key]


@pytest.fixture(scope="module")
def tiny_report() -> dict[str, object]:
    """A real tiny batch with the surrogate: 16 full climbs, then 16 estimated ones."""
    return run_batch(
        _SPEC,
        SiteConfig(),
        Settings(
            round_size=16,
            min_rounds=2,
            max_rounds=2,
            workers=1,
            reco_checks=1,
            reco_close_checks=1,
            audit_fraction=0.25,
        ),
    )


def test_a_small_batch_reports_every_probability_with_an_interval(
    tiny_report: dict[str, object],
) -> None:
    """End to end: a tiny batch gives the full report, all intervals at 90%."""
    report = tiny_report
    json.dumps(report)  # the report is plain JSON
    assert report["runs"] == 32
    for key in (
        "probability_any_failure",
        "probability_any_warning",
        "probability_apogee_at_least_target",
        "reached_apogee",
    ):
        interval = report[key]
        assert 0.0 <= interval["low"] <= interval["estimate"] <= interval["high"] <= 1.0
        assert interval["confidence"] == 0.9
    assert set(report["checks"]) >= {"rail_exit_floor", "canopy_overload", "separation"}
    assert (
        report["apogee_m"]["p10"]["estimate"] <= report["apogee_m"]["p90"]["estimate"]
    )
    assert report["apogee_m"]["mean"]["low"] <= report["apogee_m"]["mean"]["high"]
    assert report["footprint"]["n"] > 0
    check = report["fast_reco_check"]
    assert check["pairs"] == 2, "one typical flight and one close to a limit"
    assert check["runs"] + check["skipped"] == check["pairs"]
    assert check["gusts_off"] is True
    assert report["full_climbs"] >= 16 and report["surrogate_climbs"] == 16
    assert "placeholder" in report["assumptions"].lower()
    assert report["missing_numbers"]["flights"] == 0
    assert report["simulation_errors"] == 0 and report["first_simulation_error"] is None


def test_the_text_report_of_a_batch_with_the_surrogate_does_not_crash(
    tiny_report: dict[str, object],
) -> None:
    """``format_report`` reads only keys that exist (it used to read a missing landing-speed one)."""
    text = format_report(tiny_report)
    assert "EDITH Monte Carlo" in text and "Intervals are 90% confidence" in text
    scale_line = text.split("Surrogate error scale")[1].split("\n")[0]
    assert "landing speed" not in scale_line, "the landing speed is not judged"
    assert "two chances it watches (any IREC failure" in text and "+/-4.0%" in text
    assert "FastRECO against the full RECO (a spot check of 2 flights" in text


def test_the_text_report_of_a_batch_without_the_surrogate_does_not_crash(
    fake_flights: None,
) -> None:
    """The same with every climb flown in full (no surrogate lines)."""
    text = format_report(_run(max_rounds=1))
    assert "all 16 flown with the full 6-DOF climb" in text
    assert "Surrogate" not in text


def test_the_report_uses_the_configured_confidence_and_names_the_never_seen_bound(
    tiny_report: dict[str, object],
) -> None:
    """The header follows the confidence; the 'never triggered' line gives the real bound."""
    report = json.loads(json.dumps(tiny_report))
    report["settings"]["confidence"] = 0.8
    for check in report["checks"].values():
        check["probability"].update(estimate=0.0, low=0.0, high=0.0054)
    text = format_report(report)
    assert "Intervals are 80% confidence" in text
    assert "90% sure each is below 0.54%" in text  # 80% two-sided is 90% one-sided
    assert "0.05%" not in text


# ----- the batch with a fast stand-in for the flights --------------------------------


def _flight(**over: object) -> dict[str, object]:
    """One flown run as the batch records it: a good flight with some changes."""
    flight: dict[str, object] = {
        **_good(),
        "apogee_east": 100.0,
        "apogee_north": 50.0,
        "land_east": 800.0,
        "land_north": 300.0,
        "drift": 855.0,
        "peak_force_n": 900.0,
    }
    flight.update(over)
    return flight


class _FakePool:
    """Stands in for the pool: flights are instant and fail by a rule of the test."""

    broken = staticmethod(lambda index, kind: False)  # which flights stop with an error

    def __init__(self, _spec: object, _site: object, workers: int) -> None:
        self.workers = workers

    def run(self, jobs: list[tuple]) -> list[tuple]:  # type: ignore[type-arg]
        out = []
        for kind, index, unit, _seed, _predicted in jobs:
            if kind == "check":
                good = _flight()
                result: dict[str, object] = {
                    "fast": good,
                    "full": dict(good),
                    "fast_s": 0.1,
                    "full_s": 1.0,
                }
            elif self.broken(index, kind):
                result = {"sim_ok": False, "error": "RuntimeError: boom"}
            else:
                # about half the flights break a rule, so no chance is quickly pinned down
                result = _flight(
                    separated=unit[1] > 0.5, apogee_m=9144.0 + 800.0 * (unit[0] - 0.5)
                )
            out.append((index, kind, result))
        return out

    def close(self) -> None:
        """Nothing to shut down."""


@pytest.fixture
def fake_flights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flight_sim.edith.batch._Pool", _FakePool)
    monkeypatch.setattr(_FakePool, "broken", staticmethod(lambda index, kind: False))


def _run(**settings: object) -> dict[str, object]:
    base: dict[str, object] = {
        "round_size": 16,
        "min_rounds": 1,
        "max_rounds": 3,
        "workers": 1,
        "surrogate": False,
        "reco_checks": 0,
        "reco_close_checks": 0,
    }
    base.update(settings)
    return run_batch(_SPEC, SiteConfig(), Settings(**base))  # type: ignore[arg-type]


def test_all_rounds_flown_is_reported_as_the_largest_number_of_rounds(
    fake_flights: None,
) -> None:
    """With one round planned and the time already over, the batch is complete: not 'time limit'."""
    report = _run(max_rounds=1, time_limit_s=1e-9, target_half_width=1e-6)
    assert report["rounds"] == 1
    assert report["stopped_because"] == "reached the largest number of rounds"
    assert report["stopped_for"] == "max_rounds"


def test_a_batch_cut_short_by_the_time_limit_is_labelled_so(fake_flights: None) -> None:
    """Rounds left over and no time left: that is the time limit."""
    report = _run(max_rounds=3, time_limit_s=1e-9, target_half_width=1e-6)
    assert report["rounds"] == 1
    assert report["stopped_because"] == "reached the time limit"
    assert report["stopped_for"] == "time_limit"


def test_the_batch_stops_when_the_two_watched_chances_reach_the_target(
    fake_flights: None,
) -> None:
    """Stops at the target and says which chances were watched, and the true width."""
    report = _run(min_rounds=2, max_rounds=5, target_half_width=0.25)
    assert report["stopped_for"] == "target" and report["rounds"] == 2
    assert "any IREC failure" in report["stopped_because"]
    assert "apogee at least the target" in report["stopped_because"]
    assert "+/-25.0%" in report["stopped_because"]
    assert "every headline" not in report["stopped_because"]
    assert report["watched_chances"] == [
        "any IREC failure",
        "apogee at least the target",
    ]
    narrow = _run(min_rounds=2, max_rounds=3, target_half_width=0.001)
    assert narrow["stopped_for"] == "max_rounds" and narrow["rounds"] == 3


def test_the_default_stop_target_is_plus_or_minus_4_percent_on_the_command_line_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings and the command line both default to 0.04, and the command line can change it."""
    import flight_sim.edith.__main__ as cli

    seen: list[Settings] = []

    class _StopError(Exception):
        pass

    def capture(_spec: object, _site: object, settings: Settings) -> None:
        seen.append(settings)
        raise _StopError

    monkeypatch.setattr(cli, "run_batch", capture)
    for argv in (
        ["--rocket", "morpheus"],
        ["--rocket", "morpheus", "--target", "0.02"],
    ):
        with pytest.raises(_StopError):
            cli.main(argv)
    assert [x.target_half_width for x in seen] == [0.04, 0.02]
    assert Settings().target_half_width == 0.04


def test_one_bad_flight_does_not_end_the_batch(
    fake_flights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flights that stop with an error are recorded and counted; the batch carries on."""
    monkeypatch.setattr(
        _FakePool, "broken", staticmethod(lambda index, kind: index % 4 == 0)
    )
    report = _run(max_rounds=2, min_rounds=2, target_half_width=0.001)
    assert report["runs"] == 32 and report["simulation_errors"] == 8
    assert report["first_simulation_error"] == "RuntimeError: boom"
    text = format_report(report)
    assert "8 runs failed to simulate" in text and "RuntimeError: boom" in text
    # errored flights are left out of the apogee numbers, not counted as 'did not reach'
    assert report["reached_apogee"]["estimate"] == 1.0
    assert report["checks"]["simulation"]["probability"]["estimate"] == pytest.approx(
        0.25
    )


def test_a_flight_with_no_apogee_counts_as_too_low_and_an_errored_one_is_left_out() -> (
    None
):
    """below + within + above cover every flight that ran; none is in no group."""
    site = SiteConfig()
    inside = site.target_apogee_m
    flights = [
        _flight(apogee_m=inside),
        _flight(apogee_m=inside * 1.3),
        {
            "sim_ok": True,
            "apogee_ok": False,
            "rail_v": 40.0,
            "rail_margin": 2.5,
            "margin_lo": 2.0,
            "margin_hi": 4.0,
        },
        _flight(apogee_m=float("nan")),
        {"sim_ok": False, "error": "RuntimeError: boom"},
    ]
    samples = [
        {"round": 0, "index": i, "source": "ascent", "final": f}
        for i, f in enumerate(flights)
    ]
    report = summarize(samples, [], [], None, site, Settings(workers=1))
    rng = report["apogee_range"]
    # four flights ran: one inside, one above, two with no usable apogee (below)
    assert rng["within"]["estimate"] == pytest.approx(0.25)
    assert rng["above"]["estimate"] == pytest.approx(0.25)
    assert rng["below"]["estimate"] == pytest.approx(0.5)
    assert report["reached_apogee"]["estimate"] == pytest.approx(
        0.75
    )  # 3 of the 4 that ran
    assert report["probability_apogee_at_least_target"]["estimate"] == pytest.approx(
        0.5
    )
    assert report["checks"]["simulation"]["probability"]["estimate"] == pytest.approx(
        0.2
    )
    cloud = report["cloud"]
    assert cloud["range"].count(-1) >= 1, "the NaN apogee is coloured as below range"
    # only the NaN apogee is a missing number; a climb that never reached apogee has none to give
    assert report["missing_numbers"] == {"flights": 1, "by_value": {"apogee_m": 1}}


def test_missing_and_nan_numbers_are_counted_for_the_report() -> None:
    """Flights that finished with a missing or NaN number are counted, by number."""
    flights = [
        _flight(),
        _flight(rail_v=float("nan")),
        _flight(main_alt=float("nan")),
        _flight(rail_v=None),
    ]
    samples = [
        {"round": 0, "index": i, "source": "ascent", "final": f}
        for i, f in enumerate(flights)
    ]
    report = summarize(samples, [], [], None, SiteConfig(), Settings(workers=1))
    assert report["missing_numbers"] == {
        "flights": 3,
        "by_value": {"main_alt": 1, "rail_v": 2},
    }
    # the NaN rail speed is a failed check, so it is in the chance of a failure
    assert report["checks"]["rail_exit_floor"]["probability"][
        "estimate"
    ] == pytest.approx(0.5)
    text = format_report(
        {
            **report,
            "rocket": "R",
            "site": "S",
            "rounds": 1,
            "seconds": 1.0,
            "workers": 1,
            "stopped_because": "x",
            "settings": {"target_half_width": 0.04, "confidence": 0.9},
            "target_apogee_m": 9144.0,
            "surrogate_climbs": 0,
            "full_climbs": 4,
            "reflown_for_closeness_or_audit": 0,
            "borderline_not_reflown_over_cap": 0,
            "surrogate_error": None,
            "simulation_errors": 0,
            "inputs": [],
        }
    )
    assert "3 flights had missing numbers (main_alt: 1, rail_v: 2)" in text


def test_a_worker_catches_any_error_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error that is not a maths error ends only that flight (it used to end the batch)."""
    from flight_sim.edith import batch

    class _Broken:
        site = SiteConfig()
        nominal = _NOMINAL

        def fly(self, *_a: object, **_k: object) -> dict[str, object]:
            raise KeyError("no such thing")

    monkeypatch.setitem(batch._WORKER, "rocket", _Broken())
    unit = [0.5] * DIMENSIONS
    index, kind, result = batch._job(("ascent", 7, unit, 1, None))
    assert (index, kind) == (7, "ascent")
    assert result["sim_ok"] is False and "KeyError" in result["error"]
    _, _, both = batch._job(("check", 8, unit, 1, None))
    assert both["fast"]["sim_ok"] is False and both["full"]["sim_ok"] is False


def test_a_rocket_flight_that_raises_is_returned_as_an_errored_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Rocket.fly`` returns the error text for any exception raised inside the flight."""
    import flight_sim.edith.run as run_module

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_module, "fly_ascent", boom)
    rocket = Rocket(_SPEC, SiteConfig())
    v = draw(_MIDDLE, rocket.site, rocket.nominal, 1)
    result = rocket.fly(v)
    assert result == {"sim_ok": False, "error": "RuntimeError: boom"}
    assert failures.any_failure(result) and failures.statuses(result)["simulation"]


# ----- the FastRECO check -----------------------------------------------------------


def test_the_reco_check_compares_pass_or_fail_including_warnings_and_counts_skips() -> (
    None
):
    """A pair that differs only in a warning disagrees; a pair that errored is counted, not dropped."""
    agree = {
        "fast": _flight(),
        "full": _flight(land_east=805.0),
        "fast_s": 0.1,
        "full_s": 1.0,
    }
    check = _reco_check([agree])
    assert (
        check["same_pass_fail"] is True
        and check["disagreements"] == 0
        and check["skipped"] == 0
    )
    warned = {
        "fast": _flight(rail_v=25.0),
        "full": _flight(),
        "fast_s": 0.1,
        "full_s": 1.0,
    }
    check = _reco_check([agree, warned])
    assert (
        check["same_pass_fail"] is False
        and check["disagreements"] == 1
        and check["pairs"] == 2
    )
    errored = {
        "fast": {"sim_ok": False, "error": "x"},
        "full": _flight(),
        "fast_s": 0.0,
        "full_s": 0.0,
    }
    check = _reco_check([agree, errored])
    assert check["disagreements"] == 1 and check["skipped"] == 1 and check["runs"] == 1
    only_errors = _reco_check([errored])
    assert (
        only_errors["runs"] == 0
        and only_errors["skipped"] == 1
        and not only_errors["same_pass_fail"]
    )
    assert _reco_check([]) is None


def test_the_close_calls_chosen_for_the_reco_check_are_the_nearest_to_a_descent_limit() -> (
    None
):
    """The runs nearest a descent limit (not the first ones) are picked, nearest first;
    a run close only to a climb limit (rail exit) is not, since both models fly the same climb."""
    results = [
        _flight(),
        _flight(drogue_v=39.0),
        _flight(rail_v=15.3),  # close to the rail limit only: no use to the descent check
        _flight(main_alt=456.0),
        _flight(landed=False),
        _flight(load_ratio=0.9),
    ]
    assert _close_to_a_limit(results, 1, 2) == [3, 1]
    assert _close_to_a_limit(results, 4, 2) == [5], (
        "the first ones are skipped; a flight that did not land is not chosen"
    )


def test_the_reco_check_compares_check_by_check() -> None:
    """Both models warning, but on different checks, is a disagreement."""
    pair = {
        "fast": _flight(main_alt=500.0),  # the main-altitude warning
        "full": _flight(drogue_v=45.0),  # the drogue-rate warning
        "fast_s": 0.1,
        "full_s": 1.0,
    }
    assert _reco_check([pair])["disagreements"] == 1


def test_a_nan_drogue_rate_or_main_altitude_is_missing_not_a_pass() -> None:
    """Absent is 'no drogue'; there but NaN is a broken result, and fails its check."""
    for key, check in (("drogue_v", "drogue_rate"), ("main_alt", "main_altitude")):
        result = _good()
        result[key] = float("nan")
        assert key in failures.missing_values(result)
        assert failures.statuses(result)[check]


def test_a_landing_that_is_not_a_number_does_not_spoil_the_footprint() -> None:
    """One NaN landing is left out of the footprint (and counted), not made NaN of it all."""
    samples = [
        {"round": 0, "index": i, "source": "ascent", "final": _flight(land_east=800.0 + 10 * i)}
        for i in range(6)
    ]
    samples[2]["final"]["land_east"] = float("nan")
    report = summarize(samples, [], [], None, SiteConfig(), Settings(workers=1))
    assert report["footprint"]["n"] == 5
    assert math.isfinite(report["footprint"]["ellipse90_semi_major_m"])
    assert report["missing_numbers"]["by_value"].get("land_east") == 1
    assert None in report["cloud"]["landing"]


def test_site_settings_that_cannot_be_right_are_refused_by_name() -> None:
    """A negative spread, a zero target or a mean wind over the launch limit stop the batch at once."""
    with pytest.raises(ValueError, match="thrust_sd"):
        SiteConfig(thrust_sd=-0.1)
    with pytest.raises(ValueError, match="target_apogee_m"):
        SiteConfig(target_apogee_m=0.0)
    with pytest.raises(ValueError, match="wind_mean_m_s"):
        SiteConfig(wind_mean_m_s=12.0, wind_launch_limit_m_s=11.0)
    with pytest.raises(ValueError, match="rated_load_g"):
        SiteConfig(rated_load_g=float("nan"))
    SiteConfig(thrust_sd=0.0)  # no spread at all is allowed


def test_calm_flights_of_both_reco_versions_land_close_together() -> None:
    """With the gusts off, FastRECO and the full RECO of one draw agree to about 1% of the drift."""
    rocket = Rocket(_SPEC, SiteConfig())
    v = draw(_MIDDLE, rocket.site, rocket.nominal, 3)
    fast = rocket.fly(v, reco="fast", calm=True)
    full = rocket.fly(v, reco="full", calm=True)
    assert fast["sim_ok"] and full["sim_ok"]
    assert (
        math.hypot(
            fast["land_east"] - full["land_east"],
            fast["land_north"] - full["land_north"],
        )
        < 0.03 * full["drift"]
    )
    assert failures.any_failure(fast) == failures.any_failure(full)


def test_a_descent_fast_reco_refuses_is_flown_with_the_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastRECO's 'cannot fly this one' sends that flight to the full RECO, and says so."""
    from flight_sim import fast_reco  # pylint: disable=import-outside-toplevel

    def refuse(self, request):  # type: ignore[no-untyped-def]
        raise fast_reco.Unsupported("a main set at or above the apogee")

    monkeypatch.setattr(fast_reco.FastRECO, "descend", refuse)
    rocket = Rocket(_SPEC, SiteConfig())
    result = rocket.fly(draw(_MIDDLE, rocket.site, rocket.nominal, 3), calm=True)
    assert result["sim_ok"] and result["landed"], result.get("error")
    assert "apogee" in result["reco_fallback"]


def test_a_batch_where_every_flight_errored_still_reports_without_nan() -> None:
    """Nothing ran: the apogee chances are 'unknown' (0 to 100%), not NaN, and the report is valid JSON."""
    samples = [
        {
            "round": 0,
            "index": i,
            "source": "ascent",
            "final": {"sim_ok": False, "error": "E"},
        }
        for i in range(4)
    ]
    report = summarize(samples, [], [], None, SiteConfig(), Settings(workers=1))
    assert report["apogee_range"]["within"]["high"] == 1.0
    assert report["probability_any_failure"]["estimate"] == 1.0
    assert "NaN" not in json.dumps(report["apogee_range"])
