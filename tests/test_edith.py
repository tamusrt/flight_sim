"""Tests of the EDITH Monte Carlo system: statistics, inputs, checks, a small batch."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from flight_sim.edith import failures
from flight_sim.edith import stats as ci
from flight_sim.edith.batch import Settings, _Pool, _run_seed, _sobol, run_batch
from flight_sim.edith.inputs import (
    DIMENSIONS,
    INPUTS,
    Nominal,
    SiteConfig,
    describe,
    draw,
)
from flight_sim.edith.run import RocketSpec
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
    assert wind.mean() < _NOMINAL.wind_speed_m_s  # the cut removes the windiest days
    assert wind.mean() > 0.7 * _NOMINAL.wind_speed_m_s
    # a Weibull is skewed: the mean is above the median
    assert wind.mean() > np.median(wind)


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


def test_a_missing_number_is_not_applicable() -> None:
    """A rocket with no drogue has no drogue rate to check."""
    result = _good()
    result["drogue_v"] = None
    assert not failures.statuses(result)["drogue_rate"]


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
                    9000.0 + 300.0 * row[0] - 200.0 * row[1] + 50.0 * row[7],
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


def test_the_surrogate_learns_a_smooth_climb() -> None:
    """On a smooth synthetic climb the held-out error is small and predictions agree."""
    x, payloads = _synthetic(120)
    surrogate = Surrogate(x, payloads)  # type: ignore[arg-type]
    assert set(surrogate.error) == set(TARGETS)
    assert surrogate.error["apogee_m"] < 5.0
    probe = np.random.default_rng(8).normal(size=FEATURES)
    guess = surrogate.predict(probe)
    truth = 9000.0 + 300.0 * probe[0] - 200.0 * probe[1] + 50.0 * probe[7]
    assert guess["position"][0] == pytest.approx(truth, abs=15.0)
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
        for key in ("apogee_m", "rail_v", "land_v_vert", "main_alt", "drift_m"):
            assert x.get(key) == y.get(key)


def test_a_small_batch_reports_every_probability_with_an_interval() -> None:
    """End to end: a tiny batch gives the full report, all intervals at 90%."""
    report = run_batch(
        _SPEC,
        SiteConfig(),
        Settings(round_size=16, min_rounds=2, max_rounds=2, workers=1, reco_checks=1),
    )
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
    assert report["fast_reco_check"]["runs"] == 1
    assert report["full_climbs"] >= 16 and report["surrogate_climbs"] == 16
    assert "placeholder" in report["assumptions"].lower()
