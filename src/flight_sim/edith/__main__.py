"""Command line for EDITH: ``python -m flight_sim.edith``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flight_sim.edith.batch import Settings, run_batch
from flight_sim.edith.inputs import SiteConfig
from flight_sim.edith.report import format_report
from flight_sim.edith.run import RocketSpec


def main(argv: list[str] | None = None) -> None:
    """Run a batch and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocket", choices=("invictus", "morpheus", "ork"), default="morpheus"
    )
    parser.add_argument("--ork", help="OpenRocket file (with --rocket ork)")
    parser.add_argument("--aero", help="RASAero CSV (with --rocket ork)")
    parser.add_argument("--motor", help="motor file (with --rocket ork)")
    parser.add_argument("--sim", help="saved OpenRocket simulation to launch as")
    parser.add_argument("--name", help="name for the report")
    parser.add_argument(
        "--site", help="site settings file (JSON); the placeholders if left out"
    )
    parser.add_argument(
        "--write-site", help="write the placeholder site settings here and stop"
    )
    parser.add_argument("--out", help="write the full report here as JSON")
    parser.add_argument("--round-size", type=int, default=128)
    parser.add_argument("--min-rounds", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--target",
        type=float,
        default=0.03,
        help="stop at this +/- on the headline probabilities",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="seconds")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)

    if args.write_site:
        SiteConfig().save(args.write_site)
        print(f"Wrote the placeholder site settings to {args.write_site}; edit them.")
        return
    site = SiteConfig.load(args.site) if args.site else SiteConfig()
    spec = RocketSpec(args.rocket, args.ork, args.aero, args.motor, args.sim, args.name)
    settings = Settings(
        round_size=args.round_size,
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        target_half_width=args.target,
        time_limit_s=args.time_limit,
        workers=args.workers,
        seed=args.seed,
    )
    result = run_batch(spec, site, settings)
    print(format_report(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
