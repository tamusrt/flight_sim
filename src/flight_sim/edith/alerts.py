"""Turn a batch's chances into the alerts the EDITH page shows.

Every alert has a colour:

* ``red``: a chance of failing the IREC rules (or of missing the target
  apogee) that the team should act on;
* ``amber``: worth a look: a smaller chance, a warning against the IREC
  recommendations, or too few flights to rule out a red;
* ``green``: nothing to do.

The cut-offs are the team's to change; they are the numbers at the top of this
file. They are judgement calls, not IREC rules, and the page says so.
"""

# ruff: noqa: E501  (the plain-language sentences are kept whole)
# pylint: disable=line-too-long

from __future__ import annotations

from typing import Any

# A chance of failing an IREC rule: red at or above 5%, amber at or above 1%
FAILURE_RED = 0.05
FAILURE_AMBER = 0.01
# A chance of a warning against an IREC recommendation: amber at or above 5%
WARNING_AMBER = 0.05
# The chance of coming within range of the target apogee: green at or above
# 70%, amber from 40%, red below 40%
APOGEE_GREEN = 0.70
APOGEE_AMBER = 0.40

# What each check means, in plain words (shown under the check's name)
WHY: dict[str, str] = {
    "apogee_window": (
        "The rocket peaks below 21,000 ft or above 39,000 ft above the pad, "
        "which disqualifies the flight."
    ),
    "rail_exit_floor": (
        "The rocket leaves the launch rail too slowly to fly straight "
        "(IREC minimum 15.24 m/s, 50 ft/s)."
    ),
    "rail_exit_recommended": (
        "The rocket leaves the rail slower than the IREC recommended speed "
        "(30.48 m/s, 100 ft/s)."
    ),
    "stability_rail": (
        "The rocket is not stable enough as it leaves the rail "
        "(IREC minimum 1.5 calibers: the centre of pressure is at least 1.5 "
        "body widths behind the centre of gravity)."
    ),
    "stability_lowest": (
        "Somewhere between the rail and apogee the rocket is less stable than "
        "the IREC minimum of 1.5 calibers."
    ),
    "stability_static_max": (
        "The rocket is more stable than IREC recommends (over 4 calibers) at "
        "the rail, so it will turn into the wind strongly."
    ),
    "stability_dynamic_max": (
        "The rocket is more stable than IREC recommends (over 6 calibers) "
        "during the flight."
    ),
    "drogue_rate": (
        "The rocket falls under the drogue parachute faster or slower than "
        "IREC expects (20 to 40 m/s)."
    ),
    "main_altitude": (
        "The main parachute opens higher than IREC recommends (457 m, 1,500 "
        "ft, above the pad), so the wind carries the rocket further."
    ),
    "landing_speed": (
        "The rocket lands faster than the IREC limit under the main "
        "parachute (11 m/s, 36 ft/s)."
    ),
    "canopy_overload": (
        "A parachute opens harder than it is assumed to be built for, so it could tear."
    ),
    "separation": "The ejection charge does not separate the rocket, so no parachute comes out.",
    "canopies_open": "A parachute never opens, so the rocket falls without it.",
    "reached_apogee": "The rocket does not reach its highest point.",
    "landed": "The rocket does not come down in the time allowed.",
    "simulation": "The simulation itself stopped with an error on that flight.",
}


def _chance(interval: dict[str, float]) -> dict[str, float]:
    return {k: interval[k] for k in ("estimate", "low", "high")}


def failure_level(chance: dict[str, float]) -> str:
    """Colour for the chance of failing an IREC rule."""
    if chance["estimate"] >= FAILURE_RED:
        return "red"
    if chance["estimate"] >= FAILURE_AMBER or chance["high"] >= FAILURE_RED:
        return "amber"
    return "green"


def warning_level(chance: dict[str, float]) -> str:
    """Colour for the chance of a warning against an IREC recommendation."""
    return "amber" if chance["estimate"] >= WARNING_AMBER else "green"


def apogee_level(chance: dict[str, float]) -> str:
    """Colour for the chance of coming within range of the target apogee."""
    if chance["estimate"] >= APOGEE_GREEN:
        return "green"
    return "amber" if chance["estimate"] >= APOGEE_AMBER else "red"


def build(report: dict[str, Any]) -> dict[str, Any]:
    """The alerts of a report: a headline, the list, and how the colours are set."""
    alerts: list[dict[str, Any]] = []
    checks = report["checks"]
    rng = report["apogee_range"]

    chance = _chance(report["probability_any_failure"])
    alerts.append(
        {
            "id": "any_failure",
            "level": failure_level(chance),
            "kind": "failure",
            "title": "Any IREC failure",
            "text": "At least one flight in the batch breaks an IREC rule or loses the "
            "rocket (the red checks below).",
            "chance": chance,
        }
    )
    chance = _chance(rng["within"])
    alerts.append(
        {
            "id": "apogee_range",
            "level": apogee_level(chance),
            "kind": "apogee",
            "title": "Reaching the wanted apogee",
            "text": f"Apogee within {rng['tolerance']:.0%} of the target, "
            f"{rng['low_m']:,.0f} to {rng['high_m']:,.0f} m above the pad.",
            "chance": chance,
        }
    )
    for name, check in checks.items():
        chance = _chance(check["probability"])
        if check["severity"] == "fail":
            level = failure_level(chance)
            kind = "failure"
        else:
            level = warning_level(chance)
            kind = "warning"
        alerts.append(
            {
                "id": name,
                "level": level,
                "kind": kind,
                "title": check["label"],
                "text": WHY.get(name, ""),
                "chance": chance,
            }
        )
    chance = _chance(report["probability_any_warning"])
    alerts.append(
        {
            "id": "any_warning",
            "level": warning_level(chance),
            "kind": "warning",
            "title": "Any IREC warning",
            "text": "At least one flight misses an IREC recommendation (the amber checks below).",
            "chance": chance,
        }
    )
    alerts.extend(_quality(report))
    order = {"red": 0, "amber": 1, "green": 2}
    alerts.sort(key=lambda a: order[a["level"]])
    return {
        "headline": _headline(alerts),
        "alerts": alerts,
        "levels": {
            "failure_red": FAILURE_RED,
            "failure_amber": FAILURE_AMBER,
            "warning_amber": WARNING_AMBER,
            "apogee_green": APOGEE_GREEN,
            "apogee_amber": APOGEE_AMBER,
        },
    }


def _quality(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Alerts about how much to trust the batch itself."""
    out: list[dict[str, Any]] = []
    target = report["settings"]["target_half_width"]
    if not report["stopped_because"].startswith("every headline"):
        out.append(
            {
                "id": "rough",
                "level": "amber",
                "kind": "quality",
                "title": "These chances are rougher than planned",
                "text": f"The batch stopped ({report['stopped_because']}) before the main "
                f"chances were pinned down to plus or minus {target:.0%}. Read the "
                "likely ranges, not just the middle numbers.",
                "chance": None,
            }
        )
    if report["simulation_errors"]:
        out.append(
            {
                "id": "errors",
                "level": "red",
                "kind": "quality",
                "title": "Some flights could not be simulated",
                "text": f"{report['simulation_errors']} flights stopped with an error. They count "
                "as failures above.",
                "chance": None,
            }
        )
    check = report.get("fast_reco_check")
    if check is not None and not check["same_pass_fail"]:
        out.append(
            {
                "id": "fast_reco",
                "level": "amber",
                "kind": "quality",
                "title": "The quick descent model disagreed with the full one",
                "text": "On at least one check flight the quick model and the full model did not "
                "agree on pass or fail. Treat the landing chances with extra care.",
                "chance": None,
            }
        )
    if report["borderline_not_reflown_over_cap"]:
        out.append(
            {
                "id": "borderline",
                "level": "amber",
                "kind": "quality",
                "title": "Some close calls were not double-checked",
                "text": f"{report['borderline_not_reflown_over_cap']} flights were close to a limit "
                "but were not re-flown with the full climb simulation, to save time.",
                "chance": None,
            }
        )
    return out


def _headline(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """One sentence for the top of the page."""
    red = [a for a in alerts if a["level"] == "red"]
    amber = [a for a in alerts if a["level"] == "amber"]
    if red:
        level, text = (
            "red",
            f"{len(red)} red alert{'s' if len(red) != 1 else ''}: "
            + "; ".join(a["title"] for a in red[:3]),
        )
    elif amber:
        level, text = (
            "amber",
            f"No red alerts. {len(amber)} to look at: "
            + "; ".join(a["title"] for a in amber[:3]),
        )
    else:
        level, text = "green", "No alerts. Every chance is inside the levels below."
    return {"level": level, "text": text}
