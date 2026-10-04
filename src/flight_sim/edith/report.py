"""The batch report as text, for a person to read."""

# ruff: noqa: E501  (long sentences of the report are kept whole)
# pylint: disable=line-too-long

from __future__ import annotations

from typing import Any


def _percent(item: dict[str, Any]) -> str:
    return (
        f"{100 * item['estimate']:.1f}% "
        f"({100 * item['confidence']:.0f}% interval {100 * item['low']:.1f}"
        f"-{100 * item['high']:.1f}%)"
    )


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


def format_report(r: dict[str, Any]) -> str:  # pylint: disable=too-many-locals,too-many-branches
    """The report, as lines of text. Every interval is 90% unless it says otherwise."""
    lines = [
        f"EDITH Monte Carlo: {r['rocket']} at {r['site']}",
        f"{r['runs']} flights in {r['rounds']} rounds, {r['seconds']:.0f} s on "
        f"{r['workers']} core(s); stopped because {r['stopped_because']}.",
        "Intervals are 90% confidence, under the assumed input distributions "
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
        if p["high"] > 0.0005 or p["estimate"] > 0:
            lines.append(f"  [{check['severity']}] {check['label']}: {_percent(p)}")
    quiet = [
        c["label"] for c in r["checks"].values() if c["probability"]["high"] <= 0.0005
    ]
    if quiet:
        lines.append(
            f"  Never triggered (upper bound under 0.05%): {len(quiet)} checks"
        )
    lines += ["", "Apogee and flight"]
    lines += _spread("Apogee above the pad", r["apogee_m"], "m", 0)
    lines += _spread("Rail exit speed", r["rail_exit_speed_m_s"], "m/s")
    lines += ["", "Descent"]
    lines += _spread("Drogue rate", r["drogue_rate_m_s"], "m/s")
    lines += _spread("Main deployment altitude", r["main_deploy_altitude_m"], "m", 0)
    lines += _spread("Landing speed (vertical)", r["landing_speed_m_s"], "m/s", 2)
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
            f"apogee {scale['apogee_m']:.1f} m, rail exit speed {scale['rail_v']:.2f} m/s, "
            f"main altitude {scale['main_alt']:.1f} m, landing speed {scale['land_v_vert']:.2f} m/s; "
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
            f"  Surrogate held-out error: apogee {err['apogee_m']:.1f} m, "
            f"rail exit speed {err['rail_v']:.2f} m/s, stability {err['rail_margin']:.3f} cal."
        )
    check = r.get("fast_reco_check")
    if check:
        lines.append(
            f"  FastRECO against the full RECO on {check['runs']} climbs: landing within "
            f"{check['largest_landing_offset_m']:.0f} m "
            f"({100 * check['largest_offset_share_of_drift']:.1f}% of the drift), "
            f"landing speed within {check['largest_landing_speed_diff_m_s']:.2f} m/s, "
            f"peak load within {check['largest_peak_load_diff_percent']:.1f}%, "
            f"main altitude within {check['largest_main_altitude_diff_m']:.0f} m; "
            f"same pass/fail: {'yes' if check['same_pass_fail'] else 'NO'}; "
            f"{check['fast_seconds_per_run']:.1f} s against {check['full_seconds_per_run']:.1f} s per run."
        )
    if r["simulation_errors"]:
        lines.append(
            f"  {r['simulation_errors']} runs failed to simulate (counted as failures)."
        )
    lines += ["", "Inputs (assumed unless you replaced them)"]
    for row in r["inputs"]:
        lines.append(f"  {row['input']}: {row['curve']}, {row['numbers']}")
    return "\n".join(lines)
