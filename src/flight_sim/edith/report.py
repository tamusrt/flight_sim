"""The batch report as text, for a person to read."""

# ruff: noqa: E501  (long sentences of the report are kept whole)
# pylint: disable=line-too-long

from __future__ import annotations

from typing import Any

from flight_sim.edith import stats as ci


def _pct(share: float) -> str:
    """A share of 1 as a percentage that keeps a small one readable (0.54%, not 0.5%)."""
    value = 100.0 * share
    return f"{value:.2f}%" if value < 1.0 else f"{value:.1f}%"


def _percent(item: dict[str, Any]) -> str:
    return (
        f"{100 * item['estimate']:.1f}% "
        f"({100 * item['confidence']:.0f}% interval {100 * item['low']:.1f}"
        f"-{100 * item['high']:.1f}%)"
    )


def _f(values: dict[str, Any] | None, key: str, digits: int) -> str:
    """A number from a dict of numbers, or "n/a" when it is not there (never an error)."""
    value = (values or {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{digits}f}"


def _number(item: dict[str, Any], unit: str, digits: int = 1) -> str:
    return (
        f"{item['estimate']:.{digits}f} {unit} "
        f"({item['low']:.{digits}f} to {item['high']:.{digits}f})"
    )


def _spread(
    name: str, spread: dict[str, Any] | None, unit: str, digits: int = 1
) -> list[str]:
    if not spread:
        return [f"  {name}: no runs"]
    return [
        f"  {name}: mean {_number(spread['mean'], unit, digits)}",
        f"    10th / 50th / 90th percentile: "
        f"{spread['p10']['estimate']:.{digits}f} / {spread['p50']['estimate']:.{digits}f}"
        f" / {spread['p90']['estimate']:.{digits}f} {unit}",
    ]


def format_report(r: dict[str, Any]) -> str:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """The report, as lines of text. Every interval has the batch's confidence (90% by default)."""
    confidence = r.get("settings", {}).get("confidence", ci.CONFIDENCE)
    sure_below = ci.one_sided(confidence)
    lines = [
        f"EDITH Monte Carlo: {r['rocket']} at {r['site']}",
        f"{r['runs']} flights in {r['rounds']} rounds, {r['seconds']:.0f} s on "
        f"{r['workers']} core(s); stopped because {r['stopped_because']}.",
        "The batch stops when the two chances it watches (any IREC failure, and apogee at "
        f"least the target) are each within +/-{100 * r.get('settings', {}).get('target_half_width', 0.04):.1f}%; "
        "the other chances are not watched and can be wider.",
        f"Intervals are {100 * confidence:.0f}% confidence, under the assumed input distributions "
        "(they do not cover errors in the assumptions themselves).",
        "",
        "Headline probabilities",
        f"  Any IREC failure: {_percent(r['probability_any_failure'])}",
        f"  Any IREC warning: {_percent(r['probability_any_warning'])}",
        f"  Apogee at least {r['target_apogee_m']:.0f} m: "
        f"{_percent(r['probability_apogee_at_least_target'])}",
        "",
        "By check (probability the check is failed or warned)",
    ]
    for check in r["checks"].values():
        p = check["probability"]
        if p["estimate"] > 0:
            lines.append(f"  [{check['severity']}] {check['label']}: {_percent(p)}")
    quiet = [c for c in r["checks"].values() if c["probability"]["estimate"] == 0]
    if quiet:
        worst = max(c["probability"]["high"] for c in quiet)
        lines.append(
            f"  Never triggered in {r['runs']} flights: {len(quiet)} checks "
            f"({100 * sure_below:.0f}% sure each is below {_pct(worst)})"
        )
    lines += ["", "Apogee and flight"]
    lines += _spread("Apogee above the pad", r["apogee_m"], "m", 0)
    lines += _spread("Rail exit speed", r["rail_exit_speed_m_s"], "m/s")
    lines += ["", "Descent"]
    lines += _spread("Drogue rate", r["drogue_rate_m_s"], "m/s")
    lines += _spread("Main deployment altitude", r["main_deploy_altitude_m"], "m", 0)
    lines += _spread("Peak opening load", r["peak_opening_load_g"], "g", 2)
    lines += _spread("Time from apogee to landing", r["descent_time_s"], "s", 0)
    foot = r.get("footprint")
    lines += ["", "Landing footprint (east, north of the pad)"]
    if foot:
        d = foot["distance_from_pad_m"]
        lines += [
            f"  Mean landing point: east {_number(foot['mean_east_m'], 'm', 0)}, "
            f"north {_number(foot['mean_north_m'], 'm', 0)}",
            f"  90% ellipse: {foot['ellipse90_semi_major_m']:.0f} m by "
            f"{foot['ellipse90_semi_minor_m']:.0f} m, long axis along "
            f"{foot['ellipse90_major_axis_bearing_deg']:.0f} deg",
            f"  Distance from the pad: 50% within {_number(d['p50'], 'm', 0)}, "
            f"90% within {_number(d['p90'], 'm', 0)}, "
            f"95% within {_number(d['p95'], 'm', 0)}; farthest {d['max']:.0f} m",
        ]
    else:
        lines.append("  Too few landings to describe.")
    audit = r["audit"]
    lines += ["", "How much to trust it"]
    if r["surrogate_climbs"]:
        lines.append(
            f"  Climbs: {r['full_climbs']} full 6-DOF, {r['surrogate_climbs']} from the "
            f"surrogate; {r['reflown_for_closeness_or_audit']} re-flown with the full climb "
            f"(close to a limit, or as an audit); {100 * audit['share_flown_from_surrogate']:.0f}% "
            f"of the final runs come from the surrogate."
        )
    else:
        lines.append(
            f"  Climbs: all {r['full_climbs']} flown with the full 6-DOF climb."
        )
    if audit["re_flown_pairs"]:
        scale = audit["error_scale"]
        lines.append(
            f"  Surrogate error scale (from {audit['re_flown_pairs']} re-flown runs): "
            f"apogee {_f(scale, 'apogee_m', 1)} m, rail exit speed {_f(scale, 'rail_v', 2)} m/s, "
            f"main altitude {_f(scale, 'main_alt', 1)} m; "
            f"this adds {100 * audit['extra_width_any_failure']:.2f} points of width to "
            f"the failure probability."
        )
        lines.append(
            f"  Random audit: {audit['random_audit_disagreements']} of "
            f"{audit['random_audit_runs']} runs got a different pass/fail from the surrogate."
        )
    if r["borderline_not_reflown_over_cap"]:
        lines.append(
            f"  {r['borderline_not_reflown_over_cap']} close-to-limit runs were not "
            f"re-flown because of the per-round cap."
        )
    err = r["surrogate_error"]
    if err:
        lines.append(
            f"  Surrogate held-out error: apogee {_f(err, 'apogee_m', 1)} m, "
            f"rail exit speed {_f(err, 'rail_v', 2)} m/s, stability {_f(err, 'rail_margin', 3)} cal."
        )
    check = r.get("fast_reco_check")
    if check and check.get("runs"):
        lines.append(
            f"  FastRECO against the full RECO (a spot check of {check['pairs']} flights, "
            f"typical ones and the closest to a limit, gusts off in both; not a guarantee "
            f"for every flight): landing within "
            f"{check['largest_landing_offset_m']:.0f} m "
            f"({100 * check['largest_offset_share_of_drift']:.1f}% of the drift), "
            f"landing speed within {check['largest_landing_speed_diff_m_s']:.2f} m/s, "
            f"peak load within {check['largest_peak_load_diff_percent']:.1f}%, "
            f"main altitude within {check['largest_main_altitude_diff_m']:.0f} m; "
            f"same pass/fail (rules and recommendations): "
            f"{'yes' if check['same_pass_fail'] else 'NO, on ' + str(check['disagreements']) + ' flights'}; "
            f"{check['fast_seconds_per_run']:.1f} s against {check['full_seconds_per_run']:.1f} s per run."
        )
    elif check:
        lines.append(
            f"  FastRECO against the full RECO: none of the {check['pairs']} check flights "
            f"could be compared in detail; same pass/fail: "
            f"{'yes' if check['same_pass_fail'] else 'NO, on ' + str(check['disagreements']) + ' flights'}."
        )
    if check and check.get("skipped") and check.get("runs"):
        lines.append(
            f"  {check['skipped']} of the {check['pairs']} FastRECO check flights could "
            f"not be compared in detail (a flight did not land or stopped with an error)."
        )
    if r["simulation_errors"]:
        first = r.get("first_simulation_error")
        lines.append(
            f"  {r['simulation_errors']} runs failed to simulate (counted as failures)"
            + (f"; the first error: {first}." if first else ".")
        )
    missing = r.get("missing_numbers") or {}
    if missing.get("flights"):
        names = ", ".join(f"{k}: {v}" for k, v in missing["by_value"].items())
        lines.append(
            f"  {missing['flights']} flights had missing numbers ({names}); a missing "
            f"number a check needs counts as a failed check."
        )
    lines += ["", "Inputs (assumed unless you replaced them)"]
    for row in r["inputs"]:
        lines.append(f"  {row['input']}: {row['curve']}, {row['numbers']}")
    return "\n".join(lines)
