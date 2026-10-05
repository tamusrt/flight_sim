"""The EDITH batch: many flights, drawn from the site's distributions, in parallel.

How a batch goes, in rounds of ``round_size`` runs (a power of two, one
independent scrambled Sobol set each, so the points fill the space of inputs
evenly):

1. **Pilot.** The first round flies the full 6-DOF climb for every run. These
   climbs train the climb surrogate (``surrogate``). A few of them (three typical
   runs, and the three closest to an IREC limit) are also flown with the full RECO as
   well as FastRECO, with the gusts switched off in both so that only the model
   difference shows, to measure how far FastRECO is from it. This is a spot check
   on a handful of flights, not a guarantee for every flight or every rocket.
2. **Later rounds.** The surrogate predicts each run's climb, and FastRECO flies
   the descent from the predicted apogee. Runs whose numbers lie close to an IREC
   limit (nearer than the surrogate's or FastRECO's likely error) are re-flown
   with the full climb, up to ``borderline_fraction`` of the round, closest first.
   A random ``audit_fraction`` of the rest is re-flown too. The differences between
   the surrogate's numbers and the full climbs' measure its error, and for every run
   not re-flown, the chance that error carries a number across an IREC limit is added
   as width to the probabilities, so a poor surrogate widens the answer instead of
   hiding in it.
   With ``surrogate`` off, every round flies the full climb like the pilot
   (no estimates, no re-flies); the site's runs do that, since on a rocket like
   SRT14 the full climb costs about a second.
3. **Stop** when the 90% intervals of the two chances the batch watches (any IREC
   failure, and apogee at least the target) are both narrower than
   ``target_half_width`` (plus or minus 4% unless changed), at ``max_rounds``, or at
   the time limit. The other chances (each check, any warning, the apogee range) are
   not watched and can be wider. A 90% interval on a chance near 50% is about plus or
   minus 3.6% with 512 flights, so a target much below 4% needs more flights.

Every run's inputs depend only on the batch seed and its index, so the answer is
the same on any number of cores.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import qmc

from flight_sim.edith import failures
from flight_sim.edith import stats as ci
from flight_sim.edith.inputs import (
    DIMENSIONS,
    Nominal,
    SiteConfig,
    describe,
    draw,
)
from flight_sim.edith.run import Rocket, RocketSpec
from flight_sim.edith.surrogate import Surrogate, features

# The default width the watched chances are pinned down to (plus or minus 4%)
TARGET_HALF_WIDTH = 0.04


@dataclass(frozen=True)
class Settings:  # pylint: disable=too-many-instance-attributes
    """How a batch is run.

    Attributes:
        round_size (int): Runs per round; rounded up to a power of two, at least 16
            (the climb model needs that many climbs to fit).
        min_rounds (int): Rounds always flown (the pilot and one more).
        max_rounds (int): Most rounds flown.
        target_half_width (float): Stop when the interval of each watched chance
            (any IREC failure, apogee at least the target) is this narrow
            (half-width, as a probability; 0.04 is plus or minus 4%).
        time_limit_s (float | None): Stop starting new rounds after this long.
        workers (int | None): Processes to use; all the cores when None.
        audit_fraction (float): Share of confidently classified runs re-flown.
        borderline_fraction (float): Most runs per round re-flown for being
            close to a limit.
        reco_checks (int): Typical pilot runs (the first of the round) also flown
            with the full RECO.
        reco_close_checks (int): More pilot runs, the ones closest to an IREC
            limit, also flown with the full RECO.
        seed (int): Seed of the whole batch.
        confidence (float): Confidence level of every interval.
        surrogate (bool): Estimate the climbs of later rounds from the pilot's
            (faster on big batches); off, every climb is flown in full.
    """

    round_size: int = 128
    min_rounds: int = 2
    max_rounds: int = 8
    target_half_width: float = TARGET_HALF_WIDTH
    time_limit_s: float | None = None
    workers: int | None = None
    audit_fraction: float = 0.05
    borderline_fraction: float = 0.15
    reco_checks: int = 3
    reco_close_checks: int = 3
    seed: int = 2027
    confidence: float = ci.CONFIDENCE
    surrogate: bool = True


# ----- the worker side -------------------------------------------------------

_WORKER: dict[str, Rocket] = {}


def _init(spec: RocketSpec, site: SiteConfig) -> None:
    _WORKER["rocket"] = Rocket(spec, site)


def _error_result(error: BaseException) -> dict[str, Any]:
    """The result of a flight that stopped with an error: it fails the run's checks."""
    return {"sim_ok": False, "error": f"{type(error).__name__}: {error}"}


def _job(job: tuple[Any, ...]) -> tuple[int, str, dict[str, Any]]:
    """Fly one job in a worker; the result carries the job's index back.

    Whatever goes wrong inside one flight is recorded as that flight's error and the
    batch carries on, so one bad draw does not throw away minutes of other flights.
    """
    kind, index = job[0], job[1]
    try:
        return _fly_job(job)
    except Exception as error:  # pylint: disable=broad-exception-caught
        if kind == "check":
            failed = _error_result(error)
            return (
                index,
                kind,
                {
                    "fast": failed,
                    "full": dict(failed),
                    "fast_s": 0.0,
                    "full_s": 0.0,
                },
            )
        return index, kind, _error_result(error)


def _fly_job(job: tuple[Any, ...]) -> tuple[int, str, dict[str, Any]]:
    kind, index, unit, seed, predicted = job
    rocket = _WORKER["rocket"]
    v = draw(np.asarray(unit), rocket.site, rocket.nominal, seed)
    if kind == "check":
        # gusts off in both models: otherwise each model flies different random gusts
        # (they cannot be given the same ones) and the offsets mix gust noise with
        # the model difference
        start = time.perf_counter()
        fast = rocket.fly(v, reco="fast", calm=True)
        middle = time.perf_counter()
        full = rocket.fly(v, reco="full", calm=True)
        end = time.perf_counter()
        return (
            index,
            kind,
            {
                "fast": fast,
                "full": full,
                "fast_s": middle - start,
                "full_s": end - middle,
            },
        )
    return index, kind, rocket.fly(v, predicted=predicted)


class _Pool:
    """Runs jobs on several processes, or in this one when there is just one."""

    def __init__(self, spec: RocketSpec, site: SiteConfig, workers: int) -> None:
        self.workers = workers
        if workers > 1:
            self.executor: ProcessPoolExecutor | None = ProcessPoolExecutor(
                max_workers=workers, initializer=_init, initargs=(spec, site)
            )
        else:
            self.executor = None
            _init(spec, site)

    def run(self, jobs: list[tuple[Any, ...]]) -> list[tuple[int, str, dict[str, Any]]]:
        """The results, in the order of the jobs."""
        if not jobs:
            return []
        if self.executor is None:
            return [_job(j) for j in jobs]
        chunk = max(1, len(jobs) // (self.workers * 4))
        return list(self.executor.map(_job, jobs, chunksize=chunk))

    def close(self) -> None:
        """Shut the worker processes down."""
        if self.executor is not None:
            self.executor.shutdown()


# ----- sampling --------------------------------------------------------------


def _sobol(seed: int, round_index: int, size: int) -> np.ndarray:
    """One independent scrambled Sobol set of ``size`` (a power of two) points."""
    entropy = np.random.SeedSequence([seed, 1, round_index]).generate_state(1)[0]
    sampler = qmc.Sobol(DIMENSIONS, scramble=True, seed=int(entropy))
    return np.asarray(sampler.random_base2(int(math.log2(size))))


def _run_seed(seed: int, round_index: int, index: int) -> int:
    """The seed of one run's sensor noise."""
    return int(
        np.random.SeedSequence([seed, 2, round_index, index]).generate_state(1)[0]
    )


# ----- the estimates ---------------------------------------------------------


def _flag(samples: list[dict[str, Any]], name: str, severity: str) -> np.ndarray:
    """Whether each run triggered a named check (or any check of a severity)."""
    if name == "any":
        pick = failures.any_failure if severity == "fail" else failures.any_warning
        return np.array([pick(s["final"]) for s in samples])
    return np.array([failures.statuses(s["final"])[name] for s in samples])


def _numbers(samples: list[dict[str, Any]], key: str) -> np.ndarray:
    """A number of every flight that has it (missing and NaN ones are left out;
    ``_missing_numbers`` counts them so the report can say so)."""
    values = [s["final"].get(key) for s in samples]
    return np.array(
        [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    )


_SPREAD_KEYS = (
    "apogee_m",
    "drogue_v",
    "main_alt",
    "rail_v",
    "load_ratio",
    "descent_s",
)


def _missing_numbers(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Flights that finished but have a missing or NaN number, and which numbers.

    A missing number a check needs counts as a failed check (``failures``); the
    spreads of the report leave the flight's NaN out. Either way the report says how
    many flights were affected, so it is not silent.
    """
    by_value: dict[str, int] = {}
    flights = 0
    for sample in samples:
        result = sample["final"]
        if not result.get("sim_ok"):
            continue
        keys = set(failures.missing_values(result))
        keys.update(
            key
            for key in _SPREAD_KEYS
            if isinstance(result.get(key), (int, float))
            and not math.isfinite(result[key])
        )
        flights += bool(keys)
        for key in keys:
            by_value[key] = by_value.get(key, 0) + 1
    return {"flights": flights, "by_value": dict(sorted(by_value.items()))}


def _spread(
    values: np.ndarray, rng: np.random.Generator, confidence: float
) -> dict[str, Any] | None:
    """Mean and the 10th, 50th and 90th percentiles, each with its interval."""
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "mean": ci.mean_interval(values, rng, confidence).as_dict(3),
        "p10": ci.percentile_interval(values, 10, rng, confidence).as_dict(3),
        "p50": ci.percentile_interval(values, 50, rng, confidence).as_dict(3),
        "p90": ci.percentile_interval(values, 90, rng, confidence).as_dict(3),
    }


def _footprint(
    samples: list[dict[str, Any]], rng: np.random.Generator, confidence: float
) -> dict[str, Any] | None:
    """Where the rocket comes down: mean point, 90% ellipse, distances."""
    points = np.array(
        [
            (s["final"]["land_east"], s["final"]["land_north"])
            for s in samples
            if s["final"].get("landed") and "land_east" in s["final"]
        ]
    )
    if len(points) < 3:
        return None
    mean = points.mean(axis=0)
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    scale = float(stats.chi2.ppf(confidence, 2))
    axes = np.sqrt(np.maximum(values, 0.0) * scale)
    major = vectors[:, 1]
    from_mean = np.hypot(*(points - mean).T)
    from_pad = np.hypot(points[:, 0], points[:, 1])

    def radius(values_: np.ndarray, percent: float) -> dict[str, float]:
        return ci.percentile_interval(values_, percent, rng, confidence).as_dict(1)

    return {
        "n": len(points),
        "mean_east_m": ci.mean_interval(points[:, 0], rng, confidence).as_dict(1),
        "mean_north_m": ci.mean_interval(points[:, 1], rng, confidence).as_dict(1),
        "ellipse90_semi_major_m": round(float(axes[1]), 1),
        "ellipse90_semi_minor_m": round(float(axes[0]), 1),
        "ellipse90_major_axis_bearing_deg": round(
            float(np.degrees(np.arctan2(major[0], major[1])) % 180.0), 1
        ),
        "distance_from_pad_m": {
            "p50": radius(from_pad, 50),
            "p90": radius(from_pad, 90),
            "p95": radius(from_pad, 95),
            "max": round(float(from_pad.max()), 1),
        },
        "radius_about_mean_m": {
            "p50": radius(from_mean, 50),
            "p90": radius(from_mean, 90),
        },
        "points_east_north_m": [
            [round(float(x), 1), round(float(y), 1)]
            for x, y in points[:: max(1, len(points) // 500)]
        ],
    }


_NUMERIC = sorted({c.key for c in failures.CHECKS if c.tolerance > 0.0})
_APOGEE_FLOOR_M = 5.0


def _apogee_range(site: SiteConfig) -> tuple[float, float]:
    """The lowest and highest apogee that counts as within range of the target."""
    margin = site.target_apogee_m * site.target_apogee_tolerance
    return site.target_apogee_m - margin, site.target_apogee_m + margin


_CLOUD_POINTS = 800


def _status(result: dict[str, Any]) -> int:
    """0 for a flight with no alert, 1 with a warning, 2 with a failure."""
    if failures.any_failure(result):
        return 2
    return 1 if failures.any_warning(result) else 0


def _cloud(samples: list[dict[str, Any]], site: SiteConfig) -> dict[str, Any]:
    """Where each flight peaked and came down, for the page's pictures and VISION.

    At most ``_CLOUD_POINTS`` flights, spread evenly through the batch. East and
    north are metres from the pad; ``status`` is 0 (none), 1 (warning) or 2
    (failure); ``range`` is -1 below the apogee range, 0 inside it, 1 above it.
    """
    step = max(1, math.ceil(len(samples) / _CLOUD_POINTS))
    low_m, high_m = _apogee_range(site)
    rows = []
    for sample in samples[::step]:
        r = sample["final"]
        if not isinstance(r.get("apogee_m"), (int, float)) or "apogee_east" not in r:
            continue
        place = [r["apogee_east"], r["apogee_north"]]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in place):
            continue
        height = float(r["apogee_m"])
        finite = math.isfinite(height)
        landed = bool(r.get("landed")) and "land_east" in r
        rows.append(
            {
                "apogee": [
                    round(float(r["apogee_east"]), 1),
                    round(float(r["apogee_north"]), 1),
                    round(height if finite else 0.0, 1),
                ],
                "landing": (
                    [round(float(r["land_east"]), 1), round(float(r["land_north"]), 1)]
                    if landed
                    else None
                ),
                "status": _status(r),
                "path": (
                    [p[1:] for p in r["history"]["path"]] if "history" in r else None
                ),
                # no usable apogee counts as below the range
                "range": -1
                if not finite or height < low_m
                else 1
                if height > high_m
                else 0,
            }
        )
    return {
        "n": len(rows),
        "apogee": [r["apogee"] for r in rows],
        "landing": [r["landing"] for r in rows],
        "status": [r["status"] for r in rows],
        "range": [r["range"] for r in rows],
        "paths": [r["path"] for r in rows],
    }


_GRID_POINTS = 160  # points along each chart's time axis
_MIN_SHARE = 0.5  # a time is charted while at least this share of flights has it


def _band(curves: list[tuple[np.ndarray, np.ndarray]], end_s: float) -> dict[str, Any]:
    """Mean, standard deviation, lowest and highest of many curves over time.

    Each curve is put on one time grid from 0 to ``end_s``; a time is kept while at
    least half the flights have a value there (a curve has none outside its own
    times), so the ends are not drawn from a few flights.
    """
    grid = np.linspace(0.0, end_s, _GRID_POINTS)
    values = np.full((len(curves), len(grid)), np.nan)
    for i, (t, y) in enumerate(curves):
        if len(t) < 2:
            continue
        inside = (grid >= t[0]) & (grid <= t[-1])
        values[i, inside] = np.interp(grid[inside], t, y)
    count = np.sum(~np.isnan(values), axis=0)
    keep = count >= max(3, _MIN_SHARE * len(curves))
    if not keep.any():
        return {"n": len(curves), "t": [], "mean": [], "sd": [], "min": [], "max": []}
    kept = values[:, keep]

    def r(a: np.ndarray, d: int = 3) -> list[float]:
        return [round(float(x), d) for x in a]

    return {
        "n": len(curves),
        "t": r(grid[keep], 3),
        "mean": r(np.nanmean(kept, axis=0)),
        "sd": r(np.nanstd(kept, axis=0, ddof=1)),
        "min": r(np.nanmin(kept, axis=0)),
        "max": r(np.nanmax(kept, axis=0)),
    }


def flight_series(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Altitude, Mach number and stability over time across the flown climbs.

    Altitude (metres above the pad) runs from launch to a little after the
    latest apogee, Mach to apogee, stability (calibres) while the rocket is
    faster than 30 m/s. Only flights whose climb was flown (not estimated) have
    these.
    """

    flown = [s["final"]["history"] for s in samples if "history" in s["final"]]
    if not flown:
        return {}
    altitude = []
    for h in flown:
        t = list(h["t"]) + [p[0] for p in h["path"] if p[0] > h["t"][-1]]
        a = list(h["alt"]) + [p[3] for p in h["path"] if p[0] > h["t"][-1]]
        altitude.append((np.asarray(t), np.asarray(a)))
    apogee_end = max(h["t"][-1] for h in flown)
    stability = [(np.asarray(h["mt"]), np.asarray(h["m"])) for h in flown]
    mach = [(np.asarray(h["t"]), np.asarray(h["mach"])) for h in flown]
    margin_end = max((h["mt"][-1] for h in flown if h["mt"]), default=0.0)
    return {
        "altitude_m": _band(altitude, apogee_end * 1.08),
        "stability_cal": _band(stability, margin_end),
        "mach": _band(mach, apogee_end),
    }


def error_sigmas(
    pairs: list[dict[str, Any]], surrogate: Surrogate | None, confidence: float
) -> dict[str, float]:
    """How far a surrogate-based number is likely off, by number.

    Each re-flown run gives the surrogate's number and the full climb's; the
    spread of those differences, with the upper end of its own uncertainty
    (chi-squared), is the size of the error to expect on the runs not re-flown.
    Until there are four such pairs, the surrogate's held-out error is used.
    """
    floors = {c.key: c.tolerance / 3.0 for c in failures.CHECKS if c.tolerance > 0}
    floors["apogee_m"] = _APOGEE_FLOOR_M
    held_out = surrogate.error if surrogate is not None else {}
    sigmas: dict[str, float] = {}
    for key in (*_NUMERIC, "apogee_m"):
        diffs = [
            exact["final"][key] - exact["predicted"][key]
            for exact in pairs
            if failures.finite(exact["final"].get(key))
            and failures.finite(exact["predicted"].get(key))
        ]
        if len(diffs) >= 4:
            rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
            sigma = rms * math.sqrt(
                len(diffs) / stats.chi2.ppf(1 - confidence, len(diffs))
            )
        else:
            sigma = held_out.get(key, 0.0)
        sigmas[key] = max(sigma, floors[key])
    return sigmas


def _flip(
    result: dict[str, Any], check: failures.Check, sigmas: dict[str, float]
) -> float:
    """Chance an error in a run's number carries it across a limit of the check."""
    value = result.get(check.key)
    if (
        check.tolerance <= 0.0
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        return 0.0
    sigma = sigmas.get(check.key, check.tolerance / 3.0)
    return sum(
        float(stats.norm.sf(abs(float(value) - limit) / sigma))
        for limit in failures.limits(check)
    )


def _error_share(
    samples: list[dict[str, Any]], sigmas: dict[str, float], site: SiteConfig
) -> dict[str, float]:
    """Extra width each probability needs for the runs flown from the surrogate.

    For a run whose climb was predicted, the chance its number crosses a limit
    through the surrogate's error is the normal tail beyond the distance to
    that limit. The mean of those chances over all runs is the share of the
    answer that could be wrong, and it is added to both ends of the interval.
    """
    shares: dict[str, float] = {c.name: 0.0 for c in failures.CHECKS}
    shares.update(
        {"any_fail": 0.0, "any_warn": 0.0, "apogee": 0.0, "apogee_range": 0.0}
    )
    low_m, high_m = _apogee_range(site)
    for sample in samples:
        if sample["source"] != "predicted":
            continue
        result = sample["final"]
        flips = {c.name: _flip(result, c, sigmas) for c in failures.CHECKS}
        for name, value in flips.items():
            shares[name] += value
        for severity, name in (("fail", "any_fail"), ("warn", "any_warn")):
            shares[name] += min(
                1.0,
                sum(
                    v
                    for c in failures.CHECKS
                    if c.severity == severity
                    for v in [flips[c.name]]
                ),
            )
        apogee = result.get("apogee_m")
        if failures.finite(apogee):
            shares["apogee"] += float(
                stats.norm.sf(abs(apogee - site.target_apogee_m) / sigmas["apogee_m"])
            )
            shares["apogee_range"] += sum(
                float(stats.norm.sf(abs(apogee - edge) / sigmas["apogee_m"]))
                for edge in (low_m, high_m)
            )
    return {name: value / max(len(samples), 1) for name, value in shares.items()}


def summarize(  # pylint: disable=too-many-locals,too-many-positional-arguments
    samples: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    surrogate: Surrogate | None,
    site: SiteConfig,
    settings: Settings,
) -> dict[str, Any]:
    """Every estimate of the batch so far, each with its interval."""
    rng = np.random.default_rng(
        np.random.SeedSequence([settings.seed, 3, len(samples)])
    )
    confidence = settings.confidence
    sets = np.array([s["round"] for s in samples])
    sigmas = error_sigmas(pairs, surrogate, confidence)
    share = _error_share(samples, sigmas, site)
    wrong = sum(
        failures.any_failure(a["predicted"]) != failures.any_failure(a["final"])
        or failures.any_warning(a["predicted"]) != failures.any_warning(a["final"])
        for a in audits
    )

    def probability(
        flags: np.ndarray, extra: float, only: np.ndarray | None = None
    ) -> dict[str, float]:
        """A chance with its interval; ``only`` keeps just those flights."""
        if only is not None:
            flags, flight_sets = flags[only], sets[only]
        else:
            flight_sets = sets
        if flags.size == 0:  # nothing to count (every flight stopped with an error)
            return {"estimate": 0.0, "low": 0.0, "high": 1.0, "confidence": confidence}
        return ci.widen(ci.proportion(flags, flight_sets, confidence), extra).as_dict(4)

    apogee = _numbers(samples, "apogee_m")
    # Flights that stopped with an error are reported on their own
    # (``simulation_errors``)
    # and left out of the apogee numbers. Every other flight is in exactly one of
    # below / within / above the range; one with no usable apogee (it never climbed,
    # or the number is NaN) counts as below.
    ran = np.array([bool(s["final"].get("sim_ok")) for s in samples])
    reached = np.array([bool(s["final"].get("apogee_ok")) for s in samples])
    low_m, high_m = _apogee_range(site)
    apogees = np.array(
        [
            s["final"]["apogee_m"]
            if failures.finite(s["final"].get("apogee_m"))
            else np.nan
            for s in samples
        ],
        dtype=float,
    )
    usable = ~np.isnan(apogees)
    with np.errstate(invalid="ignore"):
        over = usable & (apogees >= site.target_apogee_m)
        above = usable & (apogees > high_m)
        within = usable & (apogees >= low_m) & (apogees <= high_m)
    below = ~within & ~above
    checks = {}
    for check in failures.CHECKS:
        flags = _flag(samples, check.name, check.severity)
        checks[check.name] = {
            "label": check.label,
            "severity": check.severity,
            "probability": probability(flags, share[check.name]),
        }
    return {
        "runs": len(samples),
        "probability_any_failure": probability(
            _flag(samples, "any", "fail"), share["any_fail"]
        ),
        "probability_any_warning": probability(
            _flag(samples, "any", "warn"), share["any_warn"]
        ),
        "probability_apogee_at_least_target": probability(over, share["apogee"], ran),
        "target_apogee_m": site.target_apogee_m,
        "apogee_range": {
            "target_m": site.target_apogee_m,
            "tolerance": site.target_apogee_tolerance,
            "low_m": round(low_m, 1),
            "high_m": round(high_m, 1),
            "within": probability(within, share["apogee_range"], ran),
            "below": probability(below, share["apogee_range"] / 2.0, ran),
            "above": probability(above, share["apogee_range"] / 2.0, ran),
        },
        "cloud": _cloud(samples, site),
        "rated_load_g": site.rated_load_g,
        "reached_apogee": probability(reached, 0.0, ran),
        "missing_numbers": _missing_numbers(samples),
        "checks": checks,
        "apogee_m": _spread(apogee, rng, confidence),
        "footprint": _footprint(samples, rng, confidence),
        "drogue_rate_m_s": _spread(_numbers(samples, "drogue_v"), rng, confidence),
        "main_deploy_altitude_m": _spread(
            _numbers(samples, "main_alt"), rng, confidence
        ),
        "rail_exit_speed_m_s": _spread(_numbers(samples, "rail_v"), rng, confidence),
        "peak_opening_load_g": _spread(
            _numbers(samples, "load_ratio") * site.rated_load_g, rng, confidence
        ),
        "descent_time_s": _spread(_numbers(samples, "descent_s"), rng, confidence),
        "audit": {
            "random_audit_runs": len(audits),
            "random_audit_disagreements": int(wrong),
            "re_flown_pairs": len(pairs),
            "share_flown_from_surrogate": round(
                sum(s["source"] == "predicted" for s in samples) / max(len(samples), 1),
                3,
            ),
            "error_scale": {k: round(v, 3) for k, v in sigmas.items()},
            "extra_width_any_failure": round(share["any_fail"], 4),
        },
    }


# The chances the batch watches to decide when to stop, and how the report names them
WATCHED = {
    "probability_any_failure": "any IREC failure",
    "probability_apogee_at_least_target": "apogee at least the target",
}


def _worst_headline(summary: dict[str, Any]) -> float:
    """The widest half-width among the watched chances (``WATCHED``)."""
    return float(max(0.5 * (summary[n]["high"] - summary[n]["low"]) for n in WATCHED))


# ----- the FastRECO check ----------------------------------------------------


def _reco_check(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How far FastRECO's descents are from the full RECO's, on the same climbs.

    Pass or fail (the red and the amber checks) is compared on every pair, so a flight
    that one model could simulate and the other could not counts as a disagreement. The
    landing, speed and load differences need both flights to have landed; a pair
    without them is counted in ``skipped``, not dropped silently.
    """
    if not pairs:
        return None
    rows = []
    disagreements = 0
    for pair in pairs:
        fast, full = pair["fast"], pair["full"]
        disagreements += not _same_outcome(fast, full)
        if not (
            fast.get("sim_ok")
            and full.get("sim_ok")
            and "land_east" in fast
            and "land_east" in full
        ):
            continue
        rows.append(
            {
                "landing_offset_m": math.hypot(
                    fast["land_east"] - full["land_east"],
                    fast["land_north"] - full["land_north"],
                ),
                "drift_full_m": full["drift"],
                "landing_speed_diff_m_s": abs(
                    fast["land_v_vert"] - full["land_v_vert"]
                ),
                "peak_load_diff_percent": 100.0
                * abs(fast["peak_force_n"] - full["peak_force_n"])
                / max(full["peak_force_n"], 1e-9),
                "main_altitude_diff_m": abs(
                    (fast["main_alt"] or 0.0) - (full["main_alt"] or 0.0)
                ),
                "fast_s": pair["fast_s"],
                "full_s": pair["full_s"],
            }
        )
    base = {
        "pairs": len(pairs),
        "runs": len(rows),
        "skipped": len(pairs) - len(rows),
        "disagreements": int(disagreements),
        "same_pass_fail": disagreements == 0,
        "gusts_off": True,
    }
    if not rows:
        return base
    return {
        **base,
        "largest_landing_offset_m": round(max(r["landing_offset_m"] for r in rows), 1),
        "largest_offset_share_of_drift": round(
            max(r["landing_offset_m"] / max(r["drift_full_m"], 1.0) for r in rows), 4
        ),
        "largest_landing_speed_diff_m_s": round(
            max(r["landing_speed_diff_m_s"] for r in rows), 3
        ),
        "largest_peak_load_diff_percent": round(
            max(r["peak_load_diff_percent"] for r in rows), 2
        ),
        "largest_main_altitude_diff_m": round(
            max(r["main_altitude_diff_m"] for r in rows), 1
        ),
        "fast_seconds_per_run": round(float(np.mean([r["fast_s"] for r in rows])), 2),
        "full_seconds_per_run": round(float(np.mean([r["full_s"] for r in rows])), 2),
    }


def _same_outcome(fast: dict[str, Any], full: dict[str, Any]) -> bool:
    """Whether two flights of the same draw have the same red and amber checks."""
    return failures.any_failure(fast) == failures.any_failure(full) and (
        failures.any_warning(fast) == failures.any_warning(full)
    )


def _close_to_a_limit(
    results: list[dict[str, Any]], skip: int, count: int
) -> list[int]:
    """Indexes of the ``count`` flights nearest to a check limit (leaving out the first
    ``skip``, which are already checked), nearest first."""
    nearness = [
        (failures.closeness(r), i)
        for i, r in enumerate(results)
        if i >= skip and r.get("sim_ok") and r.get("landed")
    ]
    return [i for value, i in sorted(nearness) if math.isfinite(value)][:count]


# ----- the batch -------------------------------------------------------------


def run_batch(  # pylint: disable=too-many-locals,too-many-statements,too-many-nested-blocks,too-many-branches
    spec: RocketSpec, site: SiteConfig, settings: Settings | None = None
) -> dict[str, Any]:
    """Run a batch and return the report as a dict that can be saved as JSON."""
    settings = settings or Settings()
    size = 1 << max(math.ceil(math.log2(max(settings.round_size, 16))), 4)
    workers = settings.workers or os.cpu_count() or 1
    started = time.perf_counter()
    nominal_rocket = Rocket(spec, site)
    nominal: Nominal = nominal_rocket.nominal
    pool = _Pool(spec, site, workers)
    samples: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    reco_pairs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    surrogate: Surrogate | None = None
    top_m = 0.0
    full_climbs = predicted_climbs = reflown = borderline_uncapped = 0
    summary: dict[str, Any] = {}
    stop_reason = "reached the largest number of rounds"
    stopped_for = "max_rounds"
    try:
        for round_index in range(settings.max_rounds):
            unit = _sobol(settings.seed, round_index, size)
            seeds = [_run_seed(settings.seed, round_index, i) for i in range(size)]
            if round_index == 0 or not settings.surrogate:
                jobs: list[tuple[Any, ...]] = [
                    ("ascent", i, unit[i].tolist(), seeds[i], None) for i in range(size)
                ]
                done = pool.run(jobs)
                full_climbs += size
                pilot = sorted(done, key=lambda r: r[0])
                results = [r[2] for r in pilot]
                if round_index == 0:
                    # The FastRECO check: the first runs (typical ones, as the first
                    # points of a Sobol set are spread evenly) and the runs closest to
                    # a limit, which are the ones most likely to change pass or fail.
                    typical = min(settings.reco_checks, size)
                    chosen = list(range(typical)) + _close_to_a_limit(
                        results, typical, settings.reco_close_checks
                    )
                    reco_pairs = [
                        r[2]
                        for r in pool.run(
                            [
                                ("check", size + i, unit[i].tolist(), seeds[i], None)
                                for i in chosen
                            ]
                        )
                    ]
                for i, result in enumerate(results):
                    samples.append(
                        {
                            "round": round_index,
                            "index": i,
                            "source": "ascent",
                            "final": result,
                        }
                    )
                if settings.surrogate and round_index == 0:  # train the climb model
                    good = [(i, r) for i, r in enumerate(results) if "payload" in r]
                    top_m = float(
                        np.median([r["payload"]["position"][0] for _, r in good])
                    )
                    x = np.array(
                        [
                            features(
                                draw(unit[i], site, nominal, seeds[i]), nominal, top_m
                            )
                            for i, _ in good
                        ]
                    )
                    surrogate = Surrogate(x, [r["payload"] for _, r in good])
            else:
                assert surrogate is not None
                predictions = [
                    surrogate.predict(
                        features(draw(unit[i], site, nominal, seeds[i]), nominal, top_m)
                    )
                    for i in range(size)
                ]
                jobs = [
                    ("predicted", i, unit[i].tolist(), seeds[i], predictions[i])
                    for i in range(size)
                ]
                results = [r[2] for r in sorted(pool.run(jobs), key=lambda r: r[0])]
                predicted_climbs += size
                sigmas = error_sigmas(pairs, surrogate, settings.confidence)
                tolerances = {key: 3.0 * value for key, value in sigmas.items()}
                watch = (("apogee_m", site.target_apogee_m, tolerances["apogee_m"]),)
                closeness = np.array(
                    [failures.closeness(r, tolerances, watch) for r in results]
                )
                order = np.argsort(closeness)
                near = [int(i) for i in order if closeness[i] < 1.0]
                cap = max(int(settings.borderline_fraction * size), 1)
                borderline_uncapped += max(len(near) - cap, 0)
                near = near[:cap]
                rest = [i for i in range(size) if i not in set(near)]
                rng = np.random.default_rng(
                    np.random.SeedSequence([settings.seed, 4, round_index])
                )
                audit_count = round(settings.audit_fraction * size)
                audited = (
                    [int(i) for i in rng.choice(rest, size=audit_count, replace=False)]
                    if audit_count
                    else []
                )
                redo = near + audited
                exact = {
                    r[0]: r[2]
                    for r in pool.run(
                        [("ascent", i, unit[i].tolist(), seeds[i], None) for i in redo]
                    )
                }
                full_climbs += len(redo)
                reflown += len(redo)
                for i, result in enumerate(results):
                    sample = {
                        "round": round_index,
                        "index": i,
                        "source": "predicted",
                        "final": result,
                    }
                    if i in exact:
                        sample["source"] = "exact"
                        sample["final"] = exact[i]
                        pairs.append({"predicted": result, "final": exact[i]})
                        if i in audited:
                            audits.append({"predicted": result, "final": exact[i]})
                    samples.append(sample)
            summary = summarize(samples, pairs, audits, surrogate, site, settings)
            worst = _worst_headline(summary)
            if (
                round_index + 1 >= settings.min_rounds
                and worst <= settings.target_half_width
            ):
                stop_reason = (
                    "the watched chances (" + " and ".join(WATCHED.values()) + ") are "
                    f"within +/-{settings.target_half_width:.1%}"
                )
                stopped_for = "target"
                break
            if round_index + 1 >= settings.max_rounds:
                break  # every planned round was flown: that is the reason, not the time
            elapsed = time.perf_counter() - started
            if settings.time_limit_s is not None and elapsed >= settings.time_limit_s:
                stop_reason = "reached the time limit"
                stopped_for = "time_limit"
                break
    finally:
        pool.close()
    elapsed = time.perf_counter() - started
    errors = [s["final"] for s in samples if not s["final"].get("sim_ok")]
    summary.update(
        {
            "rocket": nominal_rocket.profile.name,
            "site": site.name,
            "settings": asdict(settings),
            "inputs": describe(site, nominal),
            "assumptions": site.note,
            "stopped_because": stop_reason,
            "stopped_for": stopped_for,
            "watched_chances": list(WATCHED.values()),
            "rounds": samples[-1]["round"] + 1,
            "seconds": round(elapsed, 1),
            "workers": workers,
            "seconds_per_run": round(elapsed / max(len(samples), 1), 3),
            "full_climbs": full_climbs,
            "surrogate_climbs": predicted_climbs,
            "reflown_for_closeness_or_audit": reflown,
            "borderline_not_reflown_over_cap": borderline_uncapped,
            "surrogate_error": (
                {k: round(v, 3) for k, v in surrogate.error.items()}
                if surrogate is not None
                else None
            ),
            "series": flight_series(samples),
            "geometry": nominal_rocket.geometry(),
            "simulation_errors": len(errors),
            "first_simulation_error": (
                str(errors[0].get("error", "unknown error")) if errors else None
            ),
            "fast_reco_check": _reco_check(reco_pairs),
        }
    )
    return summary
