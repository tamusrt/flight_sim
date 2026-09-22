"""Build the project's predefined local gravity caches from the command line."""

import argparse
from collections.abc import Callable
from pathlib import Path

from flight_sim.environment.gravity import (
    LaunchSite,
    build_gravity_cache,
    save_gravity_cache,
)

_SITES = {
    "hearne": LaunchSite(30.8721, -96.6222),
    "seymour": LaunchSite(33.498630, -99.337548),
}


def _progress_bar(site_name: str) -> Callable[[int, int], None]:
    """Create a terminal progress callback for a site's grid columns.

    Args:
        site_name (str): Label displayed beside the progress bar.

    Returns:
        Callable[[int, int], None]: Callback accepting completed and total columns.
    """
    width = 30

    def update(completed: int, total: int) -> None:
        """Redraw the current site's completed-column indicator."""
        filled = int(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[{site_name}] [{bar}] {completed}/{total} "
            f"({completed / total:.0%})",
            end="",
            flush=True,
        )
        if completed == total:
            print()

    return update


def main() -> None:
    """Build and save one or more named local EGM2008 gravity caches."""
    parser = argparse.ArgumentParser(description="Build local EGM2008 gravity caches")
    parser.add_argument(
        "sites",
        nargs="*",
        metavar="site",
        help="Cache names to build (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gravity"),
        help="Directory for generated .npz cache files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker threads (default: up to 32 logical CPUs)",
    )
    args = parser.parse_args()

    unknown_sites = set(args.sites).difference(_SITES)
    if unknown_sites:
        parser.error(f"unknown cache site(s): {', '.join(sorted(unknown_sites))}")
    sites = args.sites or tuple(_SITES)
    for name in sites:
        site = _SITES[name]
        print(
            f"Building {name} gravity cache at "
            f"{site.latitude_deg}, {site.longitude_deg}..."
        )
        cache = build_gravity_cache(
            site.latitude_deg,
            site.longitude_deg,
            workers=args.workers,
            progress=_progress_bar(name),
        )
        destination = args.output_dir / f"{name}.npz"
        save_gravity_cache(cache, destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
