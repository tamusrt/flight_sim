"""Tests for the local gravity-field cache."""

from pathlib import Path

import pytest

from flight_sim.environment import gravity
from flight_sim.units import scalar, vector


def test_gravity_cache_interpolates_can_be_installed_and_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cache has the requested grid and supplies the public gravity API."""
    monkeypatch.setattr(
        gravity,
        "_gravity_magnitude",
        lambda latitude, longitude, altitude, coefficients, lmax: (
            latitude + longitude + altitude / 1000.0
        ),
    )
    progress_updates: list[tuple[int, int]] = []
    cache = gravity.build_gravity_cache(
        32.39,
        -106.48,
        max_altitude_m=1000.0,
        horizontal_extent_deg=1.0,
        horizontal_points=2,
        altitude_points=2,
        coefficients=object(),
        workers=1,
        progress=lambda completed, total: progress_updates.append((completed, total)),
    )

    assert cache.latitude_axis_deg.shape == (2,)
    assert cache.longitude_axis_deg.shape == (2,)
    assert cache.altitude_axis_m.shape == (2,)
    assert progress_updates == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert cache.gravity(
        scalar(32.39, "deg"), scalar(-106.48, "deg"), scalar(500.0, "m")
    ).m_as("m/s**2") == pytest.approx(-73.59)

    path = tmp_path / "hearne.npz"
    gravity.save_gravity_cache(cache, path)
    loaded_cache = gravity.load_gravity_cache(path)
    assert loaded_cache.gravity(
        scalar(32.39, "deg"), scalar(-106.48, "deg"), scalar(500.0, "m")
    ).m_as("m/s**2") == pytest.approx(-73.59)

    gravity.set_gravity_cache(cache)
    try:
        assert gravity.get_gravity(
            scalar(32.39, "deg"), scalar(-106.48, "deg"), scalar(500.0, "m")
        ).m_as("m/s**2") == pytest.approx(-73.59)
    finally:
        gravity.set_gravity_cache(None)


def test_local_position_is_converted_from_configured_launch_site() -> None:
    """A local origin is translated to the configured geodetic launch site."""
    original_site = gravity._launch_site
    site = gravity.LaunchSite(30.8721, -96.6222, 87.0)
    gravity.set_launch_site(site)
    try:
        latitude, longitude, altitude = gravity.local_position_to_geodetic(
            vector((0.0, 0.0, 0.0), "m")
        )
    finally:
        gravity.set_launch_site(original_site)

    assert latitude.m_as("deg") == pytest.approx(30.8721)
    assert longitude.m_as("deg") == pytest.approx(-96.6222)
    assert altitude.m_as("m") == pytest.approx(87.0)


def test_gravity_cache_rejects_degenerate_axes() -> None:
    """Interpolation needs at least two samples along every axis."""
    with pytest.raises(ValueError, match="at least two"):
        gravity.build_gravity_cache(
            32.39, -106.48, horizontal_points=1, coefficients=object()
        )
