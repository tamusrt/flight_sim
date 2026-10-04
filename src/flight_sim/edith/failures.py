"""What counts as a failure, or as a warning, in one EDITH run.

The numbers are the ones on the JARVIS predictions page's IREC checks (the
IREC Design, Test & Evaluation Guide): the page marks the stability and
landing-speed rows and the rail exit floor red, and the rest amber. EDITH uses
the same split: a *failure* is red, a *warning* is amber. On top of those, a
run also fails for the things that would end an IREC flight or lose major
points whatever the guide's numbers say:

* the apogee is outside 21,000 to 39,000 ft above the pad (the flight is
  disqualified);
* the apogee charge did not separate the rocket (nothing comes out);
* a canopy never opened (the descent is ballistic);
* a canopy was opened harder than it is rated for (it tears);
* the rocket did not reach apogee, or did not come down within the time;
* the simulation itself failed on that run.

There is no landing zone here: where the rocket comes down is reported as a
footprint, not judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# IREC thresholds, as on the predictions page
RAIL_RECOMMENDED_M_S = 30.48
RAIL_MINIMUM_M_S = 15.24
STABILITY_MIN_CAL = 1.5
STABILITY_STATIC_MAX_CAL = 4.0
STABILITY_DYNAMIC_MAX_CAL = 6.0
DROGUE_RANGE_M_S = (20.0, 40.0)
MAIN_ALTITUDE_MAX_M = 457.2
MAIN_LANDING_MAX_M_S = 11.0
# Apogee above the pad outside this range disqualifies the flight (21,000 to 39,000 ft)
APOGEE_WINDOW_M = (21000 * 0.3048, 39000 * 0.3048)


@dataclass(frozen=True)
class Check:
    """One check of a run.

    Attributes:
        name (str): Short name used in the report.
        label (str): What the check says, in words.
        severity (str): "fail" or "warn".
        key (str): The run's number the check reads (a True/False key for a
            check with no limits).
        low (float | None): Lowest allowed value.
        high (float | None): Highest allowed value.
        strict_high (bool): The high limit is exclusive (the value must be under it).
        tolerance (float): How close to a limit counts as borderline.
    """

    name: str
    label: str
    severity: str
    key: str
    low: float | None = None
    high: float | None = None
    strict_high: bool = False
    tolerance: float = 0.0


CHECKS: tuple[Check, ...] = (
    Check(
        "apogee_window",
        "Apogee 21,000 to 39,000 ft above the pad",
        "fail",
        "apogee_m",
        low=APOGEE_WINDOW_M[0],
        high=APOGEE_WINDOW_M[1],
        tolerance=30.0,
    ),
    Check(
        "rail_exit_floor",
        f"Rail exit speed at least {RAIL_MINIMUM_M_S:g} m/s",
        "fail",
        "rail_v",
        low=RAIL_MINIMUM_M_S,
        tolerance=0.3,
    ),
    Check(
        "rail_exit_recommended",
        f"Rail exit speed at least {RAIL_RECOMMENDED_M_S:g} m/s",
        "warn",
        "rail_v",
        low=RAIL_RECOMMENDED_M_S,
        tolerance=0.3,
    ),
    Check(
        "stability_rail",
        "Stability at rail exit at least 1.5 cal",
        "fail",
        "rail_margin",
        low=STABILITY_MIN_CAL,
        tolerance=0.03,
    ),
    Check(
        "stability_lowest",
        "Lowest stability to apogee at least 1.5 cal",
        "fail",
        "margin_lo",
        low=STABILITY_MIN_CAL,
        tolerance=0.03,
    ),
    Check(
        "stability_static_max",
        "Static stability at rail exit at most 4 cal",
        "warn",
        "rail_margin",
        high=STABILITY_STATIC_MAX_CAL,
        tolerance=0.03,
    ),
    Check(
        "stability_dynamic_max",
        "Highest stability in flight at most 6 cal",
        "warn",
        "margin_hi",
        high=STABILITY_DYNAMIC_MAX_CAL,
        tolerance=0.03,
    ),
    Check(
        "drogue_rate",
        "Drogue descent rate 20 to 40 m/s",
        "warn",
        "drogue_v",
        low=DROGUE_RANGE_M_S[0],
        high=DROGUE_RANGE_M_S[1],
        tolerance=0.5,
    ),
    Check(
        "main_altitude",
        "Main deployed at most 457 m above the pad",
        "warn",
        "main_alt",
        high=MAIN_ALTITUDE_MAX_M,
        tolerance=3.0,
    ),
    Check(
        "landing_speed",
        "Landing speed under the main below 11 m/s",
        "fail",
        "land_v_vert",
        high=MAIN_LANDING_MAX_M_S,
        strict_high=True,
        tolerance=0.2,
    ),
    Check(
        "canopy_overload",
        "Canopy opening load within its rating",
        "fail",
        "load_ratio",
        high=1.0,
        tolerance=0.03,
    ),
    Check("separation", "Apogee charge separated the rocket", "fail", "separated"),
    Check("canopies_open", "Every canopy opened", "fail", "all_open"),
    Check("reached_apogee", "Reached apogee", "fail", "apogee_ok"),
    Check("landed", "Came down within the time limit", "fail", "landed"),
    Check("simulation", "Simulation ran without error", "fail", "sim_ok"),
)


def _triggered(check: Check, result: dict[str, Any]) -> bool:
    value = result.get(check.key)
    if value is None:  # not applicable to this rocket (no drogue, say)
        return False
    if check.low is None and check.high is None:
        return not bool(value)
    number = float(value)
    if check.low is not None and number < check.low:
        return True
    if check.high is not None:
        return number >= check.high if check.strict_high else number > check.high
    return False


def statuses(result: dict[str, Any]) -> dict[str, bool]:
    """Which checks the run triggered (True means the check was failed or warned)."""
    return {check.name: _triggered(check, result) for check in CHECKS}


def any_failure(result: dict[str, Any]) -> bool:
    """Whether the run failed at least one red check."""
    return any(_triggered(c, result) for c in CHECKS if c.severity == "fail")


def any_warning(result: dict[str, Any]) -> bool:
    """Whether the run triggered at least one amber check."""
    return any(_triggered(c, result) for c in CHECKS if c.severity == "warn")


def limits(check: Check) -> list[float]:
    """The numbers a check compares its value with."""
    return [x for x in (check.low, check.high) if x is not None]


def closeness(
    result: dict[str, Any],
    tolerances: dict[str, float] | None = None,
    extra: tuple[tuple[str, float, float], ...] = (),
) -> float:
    """How close the run is to any limit, in units of that number's tolerance.

    Under 1 means within tolerance of a limit, so an error of that size in the
    run could change its outcome.

    Args:
        result (dict): The run's numbers.
        tolerances (dict | None): Tolerance by the run's number (``Check.key``);
            the check's own floor where there is none.
        extra (tuple): More limits to watch, as (key, limit, tolerance).
    """
    best = float("inf")
    for check in CHECKS:
        value = result.get(check.key)
        if value is None or check.tolerance <= 0.0 or isinstance(value, bool):
            continue
        tolerance = max((tolerances or {}).get(check.key, 0.0), check.tolerance)
        for limit in limits(check):
            best = min(best, abs(float(value) - limit) / tolerance)
    for key, limit, tolerance in extra:
        value = result.get(key)
        if isinstance(value, (int, float)) and tolerance > 0.0:
            best = min(best, abs(float(value) - limit) / tolerance)
    return best
