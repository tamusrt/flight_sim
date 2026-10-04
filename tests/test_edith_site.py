"""Tests of the EDITH alerts and of what EDITH adds to the site's pages."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
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
            "landing_speed": {
                "label": "Landing speed under the main below 11 m/s",
                "severity": "fail",
                "probability": _interval(0.0, 0.0, 0.004),
            },
            "main_altitude": {
                "label": "Main deployed at most 457 m above the pad",
                "severity": "warn",
                "probability": _interval(0.0, 0.0, 0.004),
            },
        },
        "settings": {"target_half_width": 0.03},
        "stopped_because": "every headline probability is within +/-3.0%",
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
    report["checks"]["landing_speed"]["probability"] = _interval(0.2, 0.15, 0.25)
    report["probability_any_failure"] = _interval(0.2, 0.15, 0.25)
    report["checks"]["main_altitude"]["probability"] = _interval(0.5, 0.45, 0.55)
    built = alerts.build(report)
    order = {"red": 0, "amber": 1, "green": 2}
    ranks = [order[a["level"]] for a in built["alerts"]]
    assert ranks == sorted(ranks), "red first, then amber, then green"
    levels = _levels(report)
    assert levels["landing_speed"] == "red" and levels["main_altitude"] == "amber"
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
        simulation_errors=3,
        borderline_not_reflown_over_cap=7,
        fast_reco_check={"same_pass_fail": False},
    )
    quality = {
        a["id"]: a for a in alerts.build(report)["alerts"] if a["kind"] == "quality"
    }
    assert set(quality) == {"rough", "errors", "fast_reco", "borderline"}
    assert (
        quality["errors"]["level"] == "red" and "3 flights" in quality["errors"]["text"]
    )
    assert quality["rough"]["level"] == "amber"


# ----- what EDITH adds to the pages --------------------------------------------------


@pytest.fixture(scope="module")
def small_report() -> dict[str, Any]:
    """A real tiny batch (32 flights of the built-in Morpheus rocket)."""
    return run_batch(
        RocketSpec("morpheus"),
        SiteConfig(),
        Settings(round_size=16, min_rounds=2, max_rounds=2, workers=1, reco_checks=1),
    )


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


def test_the_summary_has_what_the_card_shows(small_report: dict[str, Any]) -> None:
    data = edith_site.page_data(small_report, "Morpheus", "average", cached=True)
    card = edith_site.summary(data)
    assert card["link"] == "edith/index.html" and card["runs"] == small_report["runs"]
    assert (
        card["any_failure"]["id"] == "any_failure"
        and card["apogee"]["id"] == "apogee_range"
    )
    assert len(card["top"]) <= 4 and all(a["level"] != "green" for a in card["top"])
    assert card["headline"]["level"] in ("red", "amber", "green")


def _site(tmp_path: Path) -> Path:
    """A predictions folder as flight_sim.whatif.build leaves it, with the two markers."""
    out = tmp_path / "predictions"
    (out / "viewer" / "average").mkdir(parents=True)
    (out / "index.html").write_text(
        "<script>const EDITH = /*EDITHSUMMARY*/null;</script>", encoding="utf-8"
    )
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


def test_the_jarvis_page_gets_the_summary_card(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    out = _site(tmp_path)
    card = edith_site.summary(
        edith_site.page_data(small_report, "Morpheus", "average", False)
    )
    assert edith_site.patch_jarvis(out, card) is True
    text = (out / "index.html").read_text(encoding="utf-8")
    assert (
        "/*EDITHSUMMARY*/" not in text
        and json.dumps(card, separators=(",", ":")) in text
    )
    assert edith_site.patch_jarvis(out, card) is False, (
        "a page already patched is left alone"
    )
    assert edith_site.patch_jarvis(tmp_path / "nowhere", card) is False


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
    assert (
        "window.EDITH_CLOUD=" in default
        and '"link":"../../edith/index.html"' in default
    )
    assert "<!--EDITH-->" in other, (
        "the other launch conditions are not centred on EDITH's flights"
    )


def test_the_edith_page_has_its_data_and_says_what_edith_is(
    tmp_path: Path, small_report: dict[str, Any]
) -> None:
    data = edith_site.page_data(small_report, "Morpheus", "average", False)
    page = edith_site.write_page(data, tmp_path / "predictions")
    assert page == tmp_path / "predictions" / "edith" / "index.html"
    text = page.read_text(encoding="utf-8")
    assert "/*DATA*/null" not in text and '"name":"Morpheus"' in text
    assert "Monte Carlo" in text and "EDITH" in text


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
    outcome = {"stopped_because": "every headline probability is within +/-3.0%"}

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
    outcome["stopped_because"] = "reached the time limit"
    edith_site.run(other)
    edith_site.run(other)
    assert len(calls) == 3, "the cut-short result was not reused"
    assert len(list((tmp_path / "cache").glob("edith-*.json"))) == 1, (
        "and the old result was cleared"
    )


def test_without_a_cache_folder_nothing_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        edith_site, "run_batch", lambda *_: {"runs": 1, "stopped_because": "x"}
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
