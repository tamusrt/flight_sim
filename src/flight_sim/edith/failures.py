"""What counts as a failure, or as a warning, in one EDITH run.

The numbers are the ones on the JARVIS predictions page's IREC checks (the
IREC Design, Test & Evaluation Guide): the stability rows and the rail exit
floor are red, and the rest amber. EDITH uses
the same split: a *failure* is red, a *warning* is amber. On top of those, a
run also fails for the things that would end an IREC flight or lose major
points whatever the guide's numbers say:

* the apogee is outside 21,000 to 39,000 ft above the pad (the flight is
  disqualified);
* the apogee charge did not separate the rocket (nothing comes out);
* a canopy never opened (the descent is ballistic);
* a canopy was opened harder than it is rated for (it tears);
* the rocket did not reach apogee, or did not come down within the time;
* the simulation itself failed on that run;
* a number a check needs is missing or is not a finite number (NaN or infinity)
  on a run that otherwise finished: that counts as a failed check ("result
  missing: rail_v"), never as a pass. Only two numbers may be absent, because
  some rockets genuinely do not have them (see ``OPTIONAL_KEYS``): the drogue's
  descent rate (no drogue) and the main's deployment altitude (no deployment;
  the "every canopy opened" check catches that case). Absent means absent: if
  either is there but NaN or infinite, it is missing like any other number.

There is no landing zone here: where the rocket comes down is reported as a
footprint, not judged. Nor is the landing speed, and nor are the descent rates of the
main parachute: EDITH's quick descent model gave a different landing speed from the
full model, so the JARVIS page's number is the one to use. (The drogue's descent rate
and the main's deployment altitude are checked.)
"""

from __future__ import annotations

import math
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


# Numbers a rocket may genuinely not have: None (or absent) means "not applicable"
OPTIONAL_KEYS = frozenset({"drogue_v", "main_alt"})
# Numbers that only exist once the rocket has reached apogee and come down
_AFTER_APOGEE = frozenset(
    {"drogue_v", "main_alt", "load_ratio", "separated", "all_open", "landed"}
)


def finite(value: Any) -> bool:
    """Whether a value is a real (not boolean) number that is not NaN or infinite."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _usable(check: Check, value: Any) -> bool:
    """Whether a value can be judged: present, and a finite number if it is a number."""
    if value is None:
        return False
    if check.low is None and check.high is None:
        return True  # a yes/no value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
    return math.isfinite(float(value))


def missing_values(result: dict[str, Any]) -> list[str]:
    """The numbers a run that finished should have but lacks (absent, None, NaN, inf).

    A run that stopped with an error (``sim_ok`` false) is not asked for numbers: it
    already fails the "simulation" check. A run that never reached apogee is not asked
    for the descent's numbers, and its missing apogee is not "missing" but too low.
    """
    if not result.get("sim_ok"):
        return []
    climbed = result.get("apogee_ok") is not False
    out: list[str] = []
    for check in CHECKS:
        key = check.key
        if key == "sim_ok" or key in out:
            continue
        if key in OPTIONAL_KEYS:
            # absent or None: not applicable; but a value that is there and is not a
            # finite number (NaN, inf) is a broken result, never a pass
            if climbed and result.get(key) is not None and not _usable(check, result[key]):
                out.append(key)
            continue
        if not climbed and (key in _AFTER_APOGEE or key == "apogee_m"):
            continue
        if not _usable(check, result.get(key)):
            out.append(key)
    return out


def reasons(result: dict[str, Any]) -> list[str]:
    """Reasons for a run's missing numbers, e.g. "result missing: rail_v"."""
    return [f"result missing: {key}" for key in missing_values(result)]


def _triggered(  # pylint: disable=too-many-return-statements
    check: Check, result: dict[str, Any], missing: list[str] | None = None
) -> bool:
    if check.key == "sim_ok":
        return not result.get("sim_ok")  # an absent value is a failed simulation too
    if not result.get("sim_ok"):
        return False  # the run stopped with an error: only "simulation" shows it
    if missing is None:
        missing = missing_values(result)
    if check.key in missing:
        return True  # a number the check needs is not there: failed, never passed
    value = result.get(check.key)
    if value is None:
        # never reached apogee: below the window; else not applicable (no drogue, say)
        return check.key == "apogee_m" and result.get("apogee_ok") is False
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
    missing = missing_values(result)
    return {check.name: _triggered(check, result, missing) for check in CHECKS}


def any_failure(result: dict[str, Any]) -> bool:
    """Whether the run failed at least one red check."""
    missing = missing_values(result)
    return any(_triggered(c, result, missing) for c in CHECKS if c.severity == "fail")


def any_warning(result: dict[str, Any]) -> bool:
    """Whether the run triggered at least one amber check."""
    missing = missing_values(result)
    return any(_triggered(c, result, missing) for c in CHECKS if c.severity == "warn")


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
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or check.tolerance <= 0.0
        ):
            continue
        tolerance = max((tolerances or {}).get(check.key, 0.0), check.tolerance)
        for limit in limits(check):
            best = min(best, abs(float(value) - limit) / tolerance)
    for key, limit, tolerance in extra:
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and tolerance > 0.0:
            best = min(best, abs(float(value) - limit) / tolerance)
    return best
