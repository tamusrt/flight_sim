"""Provide gravity models and cached local EGM2008 gravity fields."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pyshtools as pysh  # type: ignore[import-untyped]
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

from flight_sim.units import Scalar, Vector, scalar

_GM = 3.986004415e14
_EGM2008_RADIUS_M = 6378136.3
_EARTH_ROTATION_RATE = 7.292115e-5
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = 6.6943799901413165e-3
_WGS84_EQUATORIAL_GRAVITY = 9.7803253359
_WGS84_SOMIGLIANA_CONSTANT = 1.93185265241e-3
_WGS84_GM = 3.986004418e14


@dataclass(frozen=True)
class LaunchSite:
    """Represent the geodetic origin of the simulator's local ENU frame."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


def geodetic_to_spherical_r(latitude_deg: float, height_m: float) -> float:
    """Return WGS84 geocentric radius for a geodetic point, in metres."""
    semi_major_axis_m = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = 2 * flattening - flattening**2
    latitude_rad = np.radians(latitude_deg)
    prime_vertical_radius_m = semi_major_axis_m / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude_rad) ** 2
    )
    x_m = (prime_vertical_radius_m + height_m) * np.cos(latitude_rad)
    z_m = (
        prime_vertical_radius_m * (1.0 - eccentricity_squared) + height_m
    ) * np.sin(latitude_rad)
    return float(np.hypot(x_m, z_m))


def _load_egm2008_coefficients() -> Any:
    """Load EGM2008 only when an accurate field is explicitly requested."""
    return pysh.datasets.Earth.EGM2008()


def _gravity_magnitude(
    latitude_deg: float,
    longitude_deg: float,
    height_m: float,
    coefficients: Any,
    lmax: int,
) -> float:
    """Evaluate the EGM2008 acceleration magnitude at one geodetic point."""
    radius_m = geodetic_to_spherical_r(latitude_deg, height_m)
    maximum_degree = min(lmax, int(coefficients.lmax))
    vector = pysh.gravmag.MakeGravGridPoint(
        coefficients.coeffs,
        _GM,
        _EGM2008_RADIUS_M,
        radius_m,
        latitude_deg,
        longitude_deg,
        lmax=maximum_degree,
        omega=_EARTH_ROTATION_RATE,
    )
    return float(np.linalg.norm(vector))


@dataclass(frozen=True)
class GravityCache:
    """Store trilinearly interpolated gravity magnitudes for a launch corridor."""

    latitude_axis_deg: np.ndarray[Any, Any]
    longitude_axis_deg: np.ndarray[Any, Any]
    altitude_axis_m: np.ndarray[Any, Any]
    _interpolator: Any
    launch_site: LaunchSite

    def gravity(self, latitude: Scalar, longitude: Scalar, altitude: Scalar) -> Scalar:
        """Interpolate gravity at a geodetic point.

        Points outside the cache bounds are linearly extrapolated.

        Args:
            latitude (Scalar): Geodetic latitude.
            longitude (Scalar): Geodetic longitude.
            altitude (Scalar): Height above the reference ellipsoid.

        Returns:
            Scalar: Gravity acceleration magnitude.
        """
        point = np.array(
            [
                latitude.m_as("deg"),
                longitude.m_as("deg"),
                altitude.m_as("m"),
            ]
        )
        magnitude = float(np.asarray(self._interpolator(point)).item())
        return scalar(magnitude, "m/s**2")


def build_gravity_cache(
    launch_latitude_deg: float,
    launch_longitude_deg: float,
    max_altitude_m: float = 105000.0,
    horizontal_extent_deg: float = 1.5,
    horizontal_points: int = 30,
    altitude_points: int = 20,
    lmax: int = 36,
    coefficients: Any | None = None,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> GravityCache:
    """Build an EGM2008 field around a launch site.

    The default corridor is 3 degrees wide in both latitude and longitude and
    covers 0--105 km altitude.

    Args:
        launch_latitude_deg (float): Launch-site geodetic latitude.
        launch_longitude_deg (float): Launch-site geodetic longitude.
        max_altitude_m (float, optional): Cache ceiling in metres. Defaults to
            105000.0.
        horizontal_extent_deg (float, optional): Distance to each horizontal
            edge. Defaults to 1.5.
        horizontal_points (int, optional): Samples on each horizontal axis.
            Defaults to 30.
        altitude_points (int, optional): Samples on the altitude axis. Defaults to 20.
        lmax (int, optional): Maximum spherical-harmonic degree. Defaults to 36.
        coefficients (Any | None, optional): Preloaded EGM2008 coefficients.
            Defaults to None.
        workers (int | None, optional): Worker-thread count. Defaults to None.
        progress (Callable[[int, int], None] | None, optional): Callback
            receiving completed and total columns. Defaults to None.

    Raises:
        ValueError: If cache extents, grid sizes, degree, or worker count is invalid.

    Returns:
        GravityCache: Populated grid and its trilinear interpolator.
    """
    if max_altitude_m <= 0 or horizontal_extent_deg <= 0:
        raise ValueError("cache extents must be positive")
    if horizontal_points < 2 or altitude_points < 2:
        raise ValueError("each cache axis requires at least two points")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one")

    latitude_axis = np.linspace(
        launch_latitude_deg - horizontal_extent_deg,
        launch_latitude_deg + horizontal_extent_deg,
        horizontal_points,
    )
    longitude_axis = np.linspace(
        launch_longitude_deg - horizontal_extent_deg,
        launch_longitude_deg + horizontal_extent_deg,
        horizontal_points,
    )
    altitude_axis = np.linspace(0.0, max_altitude_m, altitude_points)
    model = coefficients if coefficients is not None else _load_egm2008_coefficients()
    values = np.empty(
        (horizontal_points, horizontal_points, altitude_points), dtype=float
    )

    def build_vertical_column(
        indices: tuple[int, int],
    ) -> tuple[tuple[int, int], np.ndarray[Any, Any]]:
        """Evaluate one latitude/longitude column for the shared EGM model."""
        latitude_index, longitude_index = indices
        column = np.empty(altitude_points, dtype=float)
        for altitude_index, altitude_m in enumerate(altitude_axis):
            column[altitude_index] = _gravity_magnitude(
                float(latitude_axis[latitude_index]),
                float(longitude_axis[longitude_index]),
                float(altitude_m),
                model,
                lmax,
            )
        return indices, column

    indices = [
        (latitude_index, longitude_index)
        for latitude_index in range(horizontal_points)
        for longitude_index in range(horizontal_points)
    ]
    thread_count = workers if workers is not None else min(32, cpu_count() or 1)
    columns: list[tuple[tuple[int, int], np.ndarray[Any, Any]]] = []
    if thread_count == 1:
        for index in indices:
            columns.append(build_vertical_column(index))
            if progress is not None:
                progress(len(columns), len(indices))
    else:
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(build_vertical_column, index) for index in indices
            ]
            for future in as_completed(futures):
                columns.append(future.result())
                if progress is not None:
                    progress(len(columns), len(indices))
    for (latitude_index, longitude_index), column in columns:
        values[latitude_index, longitude_index, :] = column

    interpolator = RegularGridInterpolator(
        (latitude_axis, longitude_axis, altitude_axis),
        values,
        bounds_error=False,
        fill_value=None,
    )
    return GravityCache(
        latitude_axis,
        longitude_axis,
        altitude_axis,
        interpolator,
        LaunchSite(launch_latitude_deg, launch_longitude_deg),
    )


def save_gravity_cache(cache: GravityCache, path: Path) -> None:
    """Save cache grids and metadata to a compressed NumPy archive.

    Args:
        cache (GravityCache): Cache to persist.
        path (Path): Destination archive path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        latitude_axis_deg=cache.latitude_axis_deg,
        longitude_axis_deg=cache.longitude_axis_deg,
        altitude_axis_m=cache.altitude_axis_m,
        values=cache._interpolator.values,
        launch_latitude_deg=cache.launch_site.latitude_deg,
        launch_longitude_deg=cache.launch_site.longitude_deg,
        launch_altitude_m=cache.launch_site.altitude_m,
    )


def load_gravity_cache(path: Path) -> GravityCache:
    """Load a cache from a compressed NumPy archive.

    Args:
        path (Path): Archive created by :func:`save_gravity_cache`.

    Returns:
        GravityCache: Loaded grid and reconstructed trilinear interpolator.
    """
    with np.load(path) as stored:
        latitude_axis = stored["latitude_axis_deg"]
        longitude_axis = stored["longitude_axis_deg"]
        altitude_axis = stored["altitude_axis_m"]
        values = stored["values"]
        launch_site = LaunchSite(
            float(stored["launch_latitude_deg"]),
            float(stored["launch_longitude_deg"]),
            float(stored["launch_altitude_m"]),
        )
    interpolator = RegularGridInterpolator(
        (latitude_axis, longitude_axis, altitude_axis),
        values,
        bounds_error=False,
        fill_value=None,
    )
    return GravityCache(
        latitude_axis, longitude_axis, altitude_axis, interpolator, launch_site
    )


_active_cache: GravityCache | None = None
_launch_site = LaunchSite(0.0, 0.0)


def set_launch_site(launch_site: LaunchSite) -> None:
    """Set the geodetic origin for the simulator's local ENU coordinates."""
    global _launch_site
    _launch_site = launch_site


def local_position_to_geodetic(position: Vector) -> tuple[Scalar, Scalar, Scalar]:
    """Convert local east-north-up position to latitude, longitude, altitude.

    The conversion uses WGS84 curvature at the configured launch site.

    Args:
        position (Vector): East-north-up displacement from the launch site.

    Returns:
        tuple[Scalar, Scalar, Scalar]: Latitude, longitude, and ellipsoid height.
    """
    east_m, north_m, up_m = position.m_as("m")
    semi_major_axis_m = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = 2 * flattening - flattening**2
    origin_latitude_rad = np.radians(_launch_site.latitude_deg)
    sin_latitude = np.sin(origin_latitude_rad)
    radius_denominator = 1.0 - eccentricity_squared * sin_latitude**2
    prime_vertical_radius_m = semi_major_axis_m / np.sqrt(radius_denominator)
    meridional_radius_m = (
        semi_major_axis_m * (1.0 - eccentricity_squared) / radius_denominator**1.5
    )
    latitude_deg = _launch_site.latitude_deg + np.degrees(
        north_m / (meridional_radius_m + _launch_site.altitude_m)
    )
    longitude_deg = _launch_site.longitude_deg + np.degrees(
        east_m
        / (
            (prime_vertical_radius_m + _launch_site.altitude_m)
            * np.cos(origin_latitude_rad)
        )
    )
    altitude_m = _launch_site.altitude_m + up_m
    return (
        scalar(float(latitude_deg), "deg"),
        scalar(float(longitude_deg), "deg"),
        scalar(float(altitude_m), "m"),
    )


def set_gravity_cache(cache: GravityCache | None) -> None:
    """Set the cache used by :func:`get_gravity`.

    Args:
        cache (GravityCache | None): Cache to activate, or None to clear it.
    """
    global _active_cache
    _active_cache = cache
    if cache is not None:
        set_launch_site(cache.launch_site)


def get_gravity(latitude: Scalar, longitude: Scalar, altitude: Scalar) -> Scalar:
    """Return gravity for the integrator.

    Args:
        latitude (Scalar): Geodetic latitude.
        longitude (Scalar): Geodetic longitude.
        altitude (Scalar): Height above the reference ellipsoid.

    Returns:
        Scalar: Cached gravity magnitude or WGS84 normal gravity when no cache
            is active.
    """
    if _active_cache is not None:
        return _active_cache.gravity(latitude, longitude, altitude)
    latitude_rad = float(latitude.m_as("rad"))
    height_m = float(altitude.m_as("m"))
    sin_latitude_squared = np.sin(latitude_rad) ** 2
    surface_gravity = _WGS84_EQUATORIAL_GRAVITY * (
        1.0 + _WGS84_SOMIGLIANA_CONSTANT * sin_latitude_squared
    ) / np.sqrt(1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude_squared)
    semi_minor_axis_m = _WGS84_SEMI_MAJOR_AXIS_M * (1.0 - _WGS84_FLATTENING)
    rotation_parameter = (
        _EARTH_ROTATION_RATE**2
        * _WGS84_SEMI_MAJOR_AXIS_M**2
        * semi_minor_axis_m
        / _WGS84_GM
    )
    height_factor = 1.0 - (
        2.0
        / _WGS84_SEMI_MAJOR_AXIS_M
        * (
            1.0
            + _WGS84_FLATTENING
            + rotation_parameter
            - 2.0 * _WGS84_FLATTENING * sin_latitude_squared
        )
        * height_m
    ) + 3.0 * height_m**2 / _WGS84_SEMI_MAJOR_AXIS_M**2
    return scalar(float(surface_gravity * height_factor), "m/s**2")
