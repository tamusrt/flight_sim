"""Tests of the EDITH alerts and of what EDITH adds to the site's pages."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from flight_sim.edith import alerts
from flight_sim.edith.batch import Settings, run_batch
from flight_sim.edith.inputs import SiteConfig
from flight_sim.edith.run import RocketSpec
from flight_sim.whatif import edith_site


def _interval(
    estimate: float, low: float | None = None, high: float | None = None
) -> dict[str, float]:
    return {
        "estimate": estimate,
        "low": estimate if low is None else low,
        "high": estimate if high is None else high,
        "confidence": 0.9,
    }


def _report(**over: Any) -> dict[str, Any]:
    """A report with nothing wrong, as the batch writes it (only what the alerts read)."""
    report: dict[str, Any] = {
        "runs": 512,
        "probability_any_failure": _interval(0.0, 0.0, 0.004),
        "probability_any_warning": _interval(0.0, 0.0, 0.004),
        "apogee_range": {
            "target_m": 9144.0,
            "tolerance": 0.05,
            "low_m": 8687.0,
            "high_m": 9601.0,
            "within": _interval(0.9, 0.88, 0.92),
        },
        "checks": {
            "stability_rail": {
                "label": "Stability at rail exit at least 1.5 cal",
                "severity": "fail",
                "probability": _interval(0.0, 0.0, 0.004),
            },
            "main_altitude": {
                "label": "Main deployed at most 457 m above the pad",
                "severity": "warn",
                "probability": _interval(0.0, 0.0, 0.004),
            },
        },
        "settings": {"target_half_width": 0.04, "confidence": 0.9},
        "stopped_because": "the watched chances (any IREC failure and apogee at least the target) are within +/-4.0%",
        "stopped_for": "target",
        "missing_numbers": {"flights": 0, "by_value": {}},
        "simulation_errors": 0,
        "borderline_not_reflown_over_cap": 0,
        "fast_reco_check": {"same_pass_fail": True},
    }
    report.update(over)
    return report


def _levels(report: dict[str, Any]) -> dict[str, str]:
    return {a["id"]: a["level"] for a in alerts.build(report)["alerts"]}


# ----- alert levels ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimate", "low", "high", "level"),
    [
        (0.0, 0.0, 0.004, "green"),  # none seen, and not likely to be common
        (0.0, 0.0, 0.06, "amber"),  # none seen, but too few flights to rule out a red
        (0.012, 0.005, 0.03, "amber"),
        (0.05, 0.03, 0.08, "red"),
        (0.4, 0.3, 0.5, "red"),
    ],
)
def test_failure_levels(estimate: float, low: float, high: float, level: str) -> None:
    assert alerts.failure_level(_interval(estimate, low, high)) == level


@pytest.mark.parametrize(
    ("estimate", "level"),
    [(0.0, "green"), (0.049, "green"), (0.05, "amber"), (1.0, "amber")],
)
def test_a_warning_is_never_red(estimate: float, level: str) -> None:
    assert alerts.warning_level(_interval(estimate)) == level


@pytest.mark.parametrize(
    ("estimate", "level"),
    [
        (0.95, "green"),
        (0.70, "green"),
        (0.69, "amber"),
        (0.40, "amber"),
        (0.39, "red"),
        (0.0, "red"),
    ],
)
def test_apogee_levels(estimate: float, level: str) -> None:
    assert alerts.apogee_level(_interval(estimate)) == level


def test_a_clean_report_has_only_green_alerts() -> None:
    built = alerts.build(_report())
    assert {a["level"] for a in built["alerts"]} == {"green"}
    assert built["headline"]["level"] == "green"
    assert set(built["levels"]) == {
        "failure_red",
        "failure_amber",
        "warning_amber",
        "apogee_green",
        "apogee_amber",
    }


def test_every_check_has_a_plain_explanation_and_the_alerts_are_sorted() -> None:
    report = _report()
    report["checks"]["stability_rail"]["probability"] = _interval(0.2, 0.15, 0.25)
    report["probability_any_failure"] = _interval(0.2, 0.15, 0.25)
    report["checks"]["main_altitude"]["probability"] = _interval(0.5, 0.45, 0.55)
    built = alerts.build(report)
    order = {"red": 0, "amber": 1, "green": 2}
    ranks = [order[a["level"]] for a in built["alerts"]]
    assert ranks == sorted(ranks), "red first, then amber, then green"
    levels = _levels(report)
    assert levels["stability_rail"] == "red" and levels["main_altitude"] == "amber"
    assert (
        built["headline"]["level"] == "red" and "red alert" in built["headline"]["text"]
    )
    for alert in built["alerts"]:
        if alert["kind"] in ("failure", "warning") and alert["id"] in alerts.WHY:
            assert alert["text"] == alerts.WHY[alert["id"]]
    assert all(len(text) > 20 for text in alerts.WHY.values()), (
        "each reason is a sentence"
    )


def test_a_missed_apogee_target_is_red() -> None:
    report = _report()
    report["apogee_range"]["within"] = _interval(0.2, 0.15, 0.25)
    assert _levels(report)["apogee_range"] == "red"


def test_the_quality_alerts_say_when_not_to_trust_the_numbers() -> None:
    assert not [a for a in alerts.build(_report())["alerts"] if a["kind"] == "quality"]
    report = _report(
        stopped_because="reached the time limit",
        stopped_for="time_limit",
        simulation_errors=3,
        first_simulation_error="KeyError: 'x'",
        missing_numbers={"flights": 2, "by_value": {"rail_v": 2}},
        borderline_not_reflown_over_cap=7,
        fast_reco_check={"same_pass_fail": False, "disagreements": 1, "pairs": 6},
    )
    quality = {
        a["id"]: a for a in alerts.build(report)["alerts"] if a["kind"] == "quality"
    }
    assert set(quality) == {"rough", "errors", "missing", "fast_reco", "borderline"}
    assert (
        quality["errors"]["level"] == "red" and "3 flights" in quality["errors"]["text"]
    )
    assert "KeyError: 'x'" in quality["errors"]["text"], "the first error is shown"
    assert quality["rough"]["level"] == "amber"
    assert (
        "two watched chances" in quality["rough"]["text"]
        and "4%" in quality["rough"]["text"]
    )
    assert (
        quality["missing"]["level"] == "amber"
        and "2 flights" in quality["missing"]["text"]
    )
    model = quality["fast_reco"]
    assert model["level"] == "amber" and model["title"].startswith("Model check")
    assert "1 of the 6 check flights" in model["text"]


def test_two_amber_checks_that_add_up_to_a_red_failure_chance_make_a_red_headline() -> (
    None
):
    """Two fail-checks at 3.5% each are amber alone, but 7% of flights break a rule: red, said plainly."""
    report = _report(probability_any_failure=_interval(0.07, 0.05, 0.09))
    report["checks"]["stability_rail"]["probability"] = _interval(0.035, 0.02, 0.05)
    report["checks"]["stability_lowest"] = {
        "label": "Lowest stability to apogee at least 1.5 cal",
        "severity": "fail",
        "probability": _interval(0.035, 0.02, 0.05),
    }
    built = alerts.build(report)
    levels = {a["id"]: a["level"] for a in built["alerts"]}
    assert levels["stability_rail"] == levels["stability_lowest"] == "amber"
    assert levels["any_failure"] == "red"
    assert built["headline"]["level"] == "red"
    assert "IREC rules: 7.0% of flights break a rule" in built["headline"]["text"]
    assert built["headline"]["text"].startswith("1 red alert:")
    assert built["headline"]["text"] != "No red alerts."


def test_the_headline_counts_other_red_alerts_with_the_failure_chance() -> None:
    """A red failure chance and a red apogee together are two alerts, both named."""
    report = _report(probability_any_failure=_interval(0.2, 0.15, 0.25))
    report["apogee_range"]["within"] = _interval(0.2, 0.15, 0.25)
    text = alerts.build(report)["headline"]["text"]
    assert (
        text.startswith("2 red alerts:")
        and "IREC rules: 20% of flights break a rule" in text
    )
    assert "Reaching the wanted apogee" in text


def test_the_page_script_counts_the_failure_chance_in_its_verdict_too() -> None:
    """The script shares the Python rule: no 'any_failure' left out of the red list."""
    page = (Path(edith_site.__file__).parent / "edith_page.html").read_text(
        encoding="utf-8"
    )
    assert "const allReds = A.alerts.filter((a) => a.level === 'red');" in page
    assert "const lvl = allReds.length ? 'red' : 'green';" in page
    assert "IREC rules: ' + pct(fail.estimate) + ' of flights break a rule" in page
    assert "pctUp" in page and "sure it is below" in page


# ----- what EDITH adds to the pages --------------------------------------------------


@pytest.fixture(scope="module")
def small_report() -> dict[str, Any]:
    """A real tiny batch (32 flights of the built-in Morpheus rocket)."""
    return run_batch(
        RocketSpec("morpheus"),
        SiteConfig(),
        Settings(
            round_size=16,
            min_rounds=2,
            max_rounds=2,
            workers=1,
            reco_checks=1,
            surrogate=False,
        ),
    )


def test_without_the_surrogate_every_climb_is_flown(
    small_report: dict[str, Any],
) -> None:
    assert small_report["full_climbs"] == small_report["runs"] == 32
    assert (
        small_report["surrogate_climbs"] == 0
        and small_report["surrogate_error"] is None
    )
    assert small_report["audit"]["share_flown_from_surrogate"] == 0


def test_the_climb_over_time_has_bands_for_each_chart(
    small_report: dict[str, Any],
) -> None:
    series = small_report["series"]
    assert set(series) == {"altitude_m", "mach", "stability_cal"}
    for band in series.values():
        assert band["n"] == 32 and len(band["t"]) > 20
        assert (
            len(band["t"])
            == len(band["mean"])
            == len(band["sd"])
            == len(band["min"])
            == len(band["max"])
        )
        assert all(
            a <= b + 1e-6 for a, b in zip(band["t"], band["t"][1:], strict=False)
        )
        for lo, mid, hi, sd in zip(
            band["min"], band["mean"], band["max"], band["sd"], strict=True
        ):
            assert lo - 1e-6 <= mid <= hi + 1e-6 and sd >= 0
    altitude = series["altitude_m"]
    assert altitude["mean"][0] == pytest.approx(0.0, abs=1.0), "it starts on the pad"
    peak = max(altitude["mean"])
    assert peak == pytest.approx(small_report["apogee_m"]["mean"]["estimate"], rel=0.05)
    mach = series["mach"]["mean"]
    top = mach.index(max(mach))
    assert 0 < top < len(mach) - 1, "Mach rises to burnout, then falls"
    assert mach[0] < 0.1 and mach[-1] < max(mach)
    assert 0 < min(series["stability_cal"]["mean"]) < 20


def test_every_flown_flight_has_a_path_from_the_pad_to_the_ground(
    small_report: dict[str, Any],
) -> None:
    cloud = small_report["cloud"]
    assert len(cloud["paths"]) == cloud["n"]
    for path, landing in zip(cloud["paths"], cloud["landing"], strict=True):
        assert path and path[0] == [0, 0, 0]
        assert max(p[2] for p in path) > 1000
        if landing:
            assert path[-1][2] == 0
            assert (
                abs(path[-1][0] - landing[0]) < 2 and abs(path[-1][1] - landing[1]) < 2
            )
    geometry = small_report["geometry"]
    assert geometry is None or geometry["length_m"] > geometry["diameter_m"] > 0


def test_the_band_keeps_only_times_most_flights_reach() -> None:
    import numpy as np

    from flight_sim.edith.batch import _band

    curves = [
        (np.array([0.0, 10.0]), np.array([0.0, 10.0 * k])) for k in (1.0, 2.0, 3.0)
    ]
    curves.append((np.array([0.0, 2.0]), np.array([0.0, 2.0])))  # a short one
    band = _band(curves, 10.0)
    assert (
        band["n"] == 4 and band["t"][0] == 0.0 and band["t"][-1] == pytest.approx(10.0)
    )
    assert band["mean"][-1] == pytest.approx(20.0) and band["min"][-1] == pytest.approx(
        10.0
    )
    assert band["max"][-1] == pytest.approx(30.0) and band["sd"][-1] == pytest.approx(
        10.0
    )
    assert _band([], 5.0)["t"] == []


def test_the_landing_circles_hold_their_share_of_the_landings() -> None:
    landing = [
        [float(i), 0.0] for i in range(-50, 51)
    ]  # 101 landings on a line, centred on 0
    circles = edith_site.landing_circles({"landing": [*landing, None]})
    assert circles is not None and circles["landings"] == 101
    assert circles["centre"] == [0.0, 0.0]
    assert [r["share"] for r in circles["rings"]] == [0.25, 0.5, 0.75, 0.9]
    radii = [r["radius_m"] for r in circles["rings"]]
    assert radii == sorted(radii)
    for ring in circles["rings"]:
        inside = sum(abs(p[0]) <= ring["radius_m"] + 1e-9 for p in landing) / len(
            landing
        )
        assert inside == pytest.approx(ring["share"], abs=0.02)
    assert edith_site.landing_circles({"landing": [[0.0, 0.0], None]}) is None


def test_the_batch_reports_the_apogee_range_and_a_cloud(
    small_report: dict[str, Any],
) -> None:
    rng = small_report["apogee_range"]
    assert rng["low_m"] < rng["target_m"] < rng["high_m"]
    assert rng["high_m"] / rng["target_m"] == pytest.approx(1.05)
    cloud = small_report["cloud"]
    assert (
        cloud["n"]
        == len(cloud["apogee"])
        == len(cloud["landing"])
        == len(cloud["status"])
        == len(cloud["range"])
    )
    assert set(cloud["range"]) <= {-1, 0, 1}


def test_the_page_data_is_plain_json_and_the_ellipse_closes_round_the_mean(
    small_report: dict[str, Any],
) -> None:
    data = edith_site.page_data(small_report, "Morpheus", "average", cached=False)
    json.dumps(data)
    assert data["name"] == "Morpheus" and data["cached"] is False
    assert "points_east_north_m" not in (data["report"]["footprint"] or {}), (
        "the points are in the cloud"
    )
    points = data["ellipse"]
    assert len(points) == 72
    foot = small_report["footprint"]
    mid_e = sum(p[0] for p in points) / len(points)
    mid_n = sum(p[1] for p in points) / len(points)
    assert mid_e == pytest.approx(foot["mean_east_m"]["estimate"], abs=1.0)
    assert mid_n == pytest.approx(foot["mean_north_m"]["estimate"], abs=1.0)
    assert edith_site.ellipse({"footprint": None}) == []


def _site(tmp_path: Path) -> Path:
    """A predictions folder as flight_sim.whatif.build leaves it, with the VISION markers."""
    out = tmp_path / "predictions"
    (out / "viewer" / "average").mkdir(parents=True)
    (out / "index.html").write_text("<p>JARVIS</p>", encoding="utf-8")
    (out / "viewer" / "index.html").write_text(
        "<body><!--EDITH--></body>", encoding="utf-8"
    )
    (out / "viewer" / "average" / "index.html").write_text(
        "<body><!--EDITH--></body>", encoding="utf-8"
    )
    (out / "viewer" / "other").mkdir()
    (out / "viewer" / "other" / "index.html").write_text(
        "<body><!--EDITH--></body>", encoding="utf-8"
    )
    (out / "viewer" / "sims.json").write_text(
        json.dumps(
            {
                "default": "average",
                "sims": {"average": "average/index.html", "other": "other/index.html"},
            }
        ),
        encoding="utf-8",
    )
    return out


def test_only_the_default_launch_condition_gets_the_flights_in_vision(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    out = _site(tmp_path)
    data = edith_site.page_data(small_report, "Morpheus", "average", False)
    assert edith_site.patch_viewers(out, edith_site.cloud_data(data)) == 2
    top = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    default = (out / "viewer" / "average" / "index.html").read_text(encoding="utf-8")
    other = (out / "viewer" / "other" / "index.html").read_text(encoding="utf-8")
    assert "window.EDITH_CLOUD=" in top and '"link":"../edith/index.html"' in top
    assert '"circles":{"centre"' in top, "VISION gets the landing circles"
    assert (
        "window.EDITH_CLOUD=" in default
        and '"link":"../../edith/index.html"' in default
    )
    assert "<!--EDITH-->" in other, (
        "the other launch conditions are not centred on EDITH's flights"
    )


def test_patching_the_viewers_again_replaces_the_cloud(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    out = _site(tmp_path)
    data = edith_site.page_data(small_report, "Morpheus", "average", False)
    first = edith_site.cloud_data(data)
    assert edith_site.patch_viewers(out, first) == 2
    second = {**first, "n": first["n"] + 1000}
    assert edith_site.patch_viewers(out, second) == 2
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    assert page.count("window.EDITH_CLOUD=") == 1, "no stale cloud is left behind"
    assert f'"n":{first["n"] + 1000},' in page
    assert page.startswith("<body><!--EDITH-->") and "<!--/EDITH-->" in page, (
        "the marker stays so a later run can find the cloud"
    )
    # a third run with the same data changes nothing
    before = page
    edith_site.patch_viewers(out, second)
    assert (out / "viewer" / "index.html").read_text(encoding="utf-8") == before


def test_a_cloud_script_without_markers_is_replaced_too(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    """Pages made by an earlier version have the script but no marker."""
    out = _site(tmp_path)
    old = '<body><script>window.EDITH_CLOUD={"n":1};</script></body>'
    (out / "viewer" / "index.html").write_text(old, encoding="utf-8")
    data = edith_site.page_data(small_report, "Morpheus", "average", False)
    edith_site.patch_viewers(out, edith_site.cloud_data(data))
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    assert page.count("window.EDITH_CLOUD=") == 1 and '"n":1}' not in page


def test_the_edith_page_has_its_data_and_says_what_edith_is(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    data = edith_site.page_data(small_report, "Morpheus", "average", False)
    page = edith_site.write_page(data, tmp_path / "predictions")
    assert page == tmp_path / "predictions" / "edith" / "index.html"
    text = page.read_text(encoding="utf-8")
    assert "/*DATA*/null" not in text and '"name":"Morpheus"' in text
    assert "Monte Carlo" in text and "EDITH" in text
    assert '"series":{' in text and '"paths":[' in text, (
        "the charts and the picture have their data"
    )


# ----- the result cache --------------------------------------------------------------


def _args(tmp_path: Path, **over: Any) -> argparse.Namespace:
    for name in ("a.ork", "a.csv", "a.eng"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    values: dict[str, Any] = {
        "ork": str(tmp_path / "a.ork"),
        "aero": str(tmp_path / "a.csv"),
        "motor": str(tmp_path / "a.eng"),
        "sim": "average",
        "name": "R",
        "site": None,
        "cache": str(tmp_path / "cache"),
        "minutes": 10.0,
        "round_size": 64,
        "max_rounds": 8,
        "workers": None,
    }
    values.update(over)
    return argparse.Namespace(**values)


def test_the_cache_key_follows_what_the_result_depends_on(tmp_path: Path) -> None:
    args = _args(tmp_path)
    site, settings = SiteConfig(), Settings(round_size=64, max_rounds=8)
    key = edith_site.cache_key(args, site, settings)
    assert key == edith_site.cache_key(args, site, settings)
    assert key == edith_site.cache_key(
        args, site, Settings(round_size=64, max_rounds=8, time_limit_s=5.0, workers=3)
    ), "how long or how wide it ran does not change the answer"
    assert key != edith_site.cache_key(
        args, site, Settings(round_size=32, max_rounds=8)
    )
    assert key != edith_site.cache_key(_args(tmp_path, sim="other"), site, settings)
    (tmp_path / "a.ork").write_text("a changed design", encoding="utf-8")
    assert key != edith_site.cache_key(args, site, settings), (
        "a changed design reruns EDITH"
    )
    assert key != edith_site.cache_key(
        args, SiteConfig(target_apogee_m=9000.0), settings
    ), "so do new site settings"


def test_a_finished_result_is_reused_and_one_cut_short_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    outcome = {
        "stopped_for": "target",
        "stopped_because": "the watched chances are within +/-4.0%",
    }

    def fake(_spec: object, _site: object, _settings: object) -> dict[str, Any]:
        calls.append(1)
        return {"runs": 64, **outcome}

    monkeypatch.setattr(edith_site, "run_batch", fake)
    args = _args(tmp_path)
    first = edith_site.run(args)
    second = edith_site.run(args)
    assert (first["cached"], second["cached"]) == (False, True) and len(calls) == 1
    assert second["report"]["runs"] == 64

    # a result stopped by the time limit is shown but not kept
    other = _args(tmp_path, round_size=32)
    kept = {p.name for p in (tmp_path / "cache").glob("edith-*.json")}
    assert len(kept) == 1
    outcome.update(stopped_for="time_limit", stopped_because="reached the time limit")
    edith_site.run(other)
    edith_site.run(other)
    assert len(calls) == 3, "the cut-short result was not reused"
    assert {p.name for p in (tmp_path / "cache").glob("edith-*.json")} == kept, (
        "the cut-short result was not written, and the finished one is still there"
    )
    # a finished result for other inputs replaces the old one (only one result is kept)
    outcome.update(
        stopped_for="max_rounds", stopped_because="reached the largest number of rounds"
    )
    edith_site.run(other)
    now = {p.name for p in (tmp_path / "cache").glob("edith-*.json")}
    assert len(now) == 1 and now != kept, "and the old result was cleared"
    assert not list((tmp_path / "cache").glob("*.tmp")) and not list(
        (tmp_path / "cache").glob(".*")
    )


def test_a_batch_that_flew_all_its_rounds_is_kept_even_when_the_time_was_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason 'largest number of rounds' is complete and cacheable."""
    monkeypatch.setattr(
        edith_site,
        "run_batch",
        lambda *_: {
            "runs": 8,
            "stopped_for": "max_rounds",
            "stopped_because": "reached the largest number of rounds",
        },
    )
    args = _args(tmp_path)
    assert edith_site.run(args)["cached"] is False
    assert edith_site.run(args)["cached"] is True


def _fake_batch(
    monkeypatch: pytest.MonkeyPatch, calls: list[int], rocket: str = "Old"
) -> None:
    def fake(_spec: object, _site: object, _settings: object) -> dict[str, Any]:
        calls.append(1)
        return {
            "runs": 8,
            "rocket": rocket,
            "stopped_for": "target",
            "stopped_because": "x",
        }

    monkeypatch.setattr(edith_site, "run_batch", fake)


@pytest.mark.parametrize(
    "damage", ["", '{"runs": 64, "stopped_be', "[1, 2]", "{}", "\x00\xff"]
)
def test_an_unreadable_kept_result_is_a_miss_that_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """A file cut short (or garbage) in the cache folder is deleted and the batch is run again."""
    calls: list[int] = []
    _fake_batch(monkeypatch, calls)
    args = _args(tmp_path)
    edith_site.run(args)
    (cached,) = (tmp_path / "cache").glob("edith-*.json")
    cached.write_bytes(damage.encode("latin-1"))
    result = edith_site.run(args)
    assert result["cached"] is False and len(calls) == 2
    assert json.loads(cached.read_text(encoding="utf-8"))["runs"] == 8, (
        "a good file is kept again"
    )
    assert edith_site.run(args)["cached"] is True


def test_the_kept_result_is_written_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while writing leaves no half file behind and keeps the earlier result."""
    calls: list[int] = []
    _fake_batch(monkeypatch, calls)
    args = _args(tmp_path)
    edith_site.run(args)
    (cached,) = (tmp_path / "cache").glob("edith-*.json")
    before = cached.read_text(encoding="utf-8")

    def broken(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(edith_site.json, "dump", broken)
    with pytest.raises(OSError, match="disk full"):
        edith_site._write_cache(cached, {"runs": 9})
    assert cached.read_text(encoding="utf-8") == before
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [cached.name]


def test_a_reused_result_shows_the_current_rocket_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is not part of the cache key, so a kept report is given the current name."""
    calls: list[int] = []
    _fake_batch(monkeypatch, calls, rocket="Old name")
    edith_site.run(_args(tmp_path, name="Old name"))
    again = edith_site.run(_args(tmp_path, name="New name"))
    assert again["cached"] is True and len(calls) == 1
    assert again["report"]["rocket"] == "New name"


def test_the_cache_key_follows_the_library_and_python_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new numpy, scipy, pandas or Python can change the numbers, so it reruns EDITH."""
    args = _args(tmp_path)
    site, settings = SiteConfig(), Settings(round_size=64, max_rounds=8)
    key = edith_site.cache_key(args, site, settings)
    monkeypatch.setattr(edith_site.numpy, "__version__", "0.0.1")
    assert edith_site.cache_key(args, site, settings) != key
    monkeypatch.undo()
    monkeypatch.setattr(edith_site.scipy, "__version__", "0.0.1")
    assert edith_site.cache_key(args, site, settings) != key
    monkeypatch.undo()
    monkeypatch.setattr(edith_site.pandas, "__version__", "0.0.1")
    assert edith_site.cache_key(args, site, settings) != key
    monkeypatch.undo()
    monkeypatch.setattr(edith_site.platform, "python_version", lambda: "0.0.1")
    assert edith_site.cache_key(args, site, settings) != key


def test_without_a_cache_folder_nothing_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        edith_site,
        "run_batch",
        lambda *_: {"runs": 1, "stopped_because": "x", "stopped_for": "target"},
    )
    args = _args(tmp_path, cache=None)
    assert (
        edith_site.run(args)["cached"] is False
        and edith_site.run(args)["cached"] is False
    )
    assert not (tmp_path / "cache").exists()


def test_main_stops_before_doing_anything_when_the_page_is_not_built(
    tmp_path: Path,
) -> None:
    for name in ("a.ork", "a.csv", "a.eng"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        edith_site.main(
            [
                "--ork",
                str(tmp_path / "a.ork"),
                "--aero",
                str(tmp_path / "a.csv"),
                "--motor",
                str(tmp_path / "a.eng"),
                "--out",
                str(tmp_path / "nothing"),
            ]
        )
    assert "build the JARVIS predictions page first" in str(stop.value)


def test_the_kept_result_depends_only_on_the_code_edith_runs() -> None:
    """Pages and VISION's viewer are not in the fingerprint; the descent EDITH flies is."""
    names = {
        p.relative_to(Path(edith_site.__file__).parents[1]).as_posix()
        for p in edith_site.physics_files()
    }
    assert {"fast_reco.py", "edith/batch.py", "edith/failures.py"} <= names
    assert not any(n.startswith("whatif/") for n in names)
    assert "visual_run.py" not in names


# ----- the page's own script ---------------------------------------------------------

# Runs the page's script against a stub of the browser's document and prints what it
# wrote at the top of the page.
_NODE_HARNESS = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const els = {};
const make = (id) => new Proxy({ id, innerHTML: '', textContent: '', className: '', value: '', dataset: {}, style: {},
  classList: { toggle() {}, add() {}, remove() {} }, querySelectorAll: () => [], querySelector: () => null },
  { get: (t, k) => (k in t ? t[k] : () => ({})), set: (t, k, v) => { t[k] = v; return true; } });
global.document = { getElementById: (id) => (els[id] = els[id] || make(id)), querySelectorAll: () => [], body: make('body'), addEventListener() {} };
global.window = { addEventListener() {}, location: { hash: '', pathname: '/x' } };
global.window.top = global.window.self = global.window;
global.location = { hash: '', pathname: '/x', protocol: 'http:' };
global.history = { replaceState() {} };
(0, eval)(script.replace('"use strict";', ''));
const strip = (h) => String(h).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
console.log(JSON.stringify({ verdict: strip(els.verdict.innerHTML), level: els.verdict.className, alerts: strip(els.alerts.innerHTML) }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_script_says_red_for_two_amber_checks_that_add_up(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    """Run the real page script: the verdict is red and names the failure chance, as the Python headline does."""
    report = copy.deepcopy(small_report)
    report["probability_any_failure"] = _interval(0.07, 0.05, 0.09)
    for name in ("rail_exit_floor", "stability_rail"):
        report["checks"][name]["probability"] = _interval(0.035, 0.02, 0.05)
    data = edith_site.page_data(report, "Morpheus", "average", cached=False)
    assert data["alerts"]["headline"]["level"] == "red"
    page = edith_site.write_page(data, tmp_path)
    harness = tmp_path / "harness.js"
    harness.write_text(_NODE_HARNESS, encoding="utf-8")
    done = subprocess.run(
        ["node", str(harness), str(page)], capture_output=True, text=True, check=True
    )
    shown = json.loads(done.stdout)
    assert shown["level"] == "verdict red"
    assert "IREC rules: 7.0% of flights break a rule" in shown["verdict"]
    assert (
        "IREC rules: 7.0% of flights break a rule" in data["alerts"]["headline"]["text"]
    )
