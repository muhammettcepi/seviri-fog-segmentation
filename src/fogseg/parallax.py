"""Parallax: where a cloud at some height appears on the ground when seen from the satellite."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
SATELLITE_RADIUS_KM = 42164.0


def unit_vector(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit vectors from the Earth's centre to (lon, lat) in degrees, as an (..., 3) array."""
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    return np.stack(
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)], axis=-1
    )


def apparent_position(
    lon: np.ndarray, lat: np.ndarray, height_m: np.ndarray, satellite_lon: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Where a point `height_m` above the ground appears on the ground, seen from a satellite over the equator.

    Positions are in degrees on a spherical Earth. The apparent position lies behind the point,
    away from the sub-satellite point.
    """
    satellite_lon_rad = np.radians(satellite_lon)
    satellite = SATELLITE_RADIUS_KM * np.array([np.cos(satellite_lon_rad), np.sin(satellite_lon_rad), 0.0])
    point = (EARTH_RADIUS_KM + np.asarray(height_m) / 1000.0)[..., np.newaxis] * unit_vector(lon, lat)
    direction = point - satellite

    # |satellite + t * direction| = EARTH_RADIUS_KM; the smaller root is where the ray reaches the ground
    quadratic = np.sum(direction**2, axis=-1)
    linear = 2 * direction @ satellite
    constant = satellite @ satellite - EARTH_RADIUS_KM**2
    ray_length = (-linear - np.sqrt(linear**2 - 4 * quadratic * constant)) / (2 * quadratic)

    ground = satellite + ray_length[..., np.newaxis] * direction
    return np.degrees(np.arctan2(ground[..., 1], ground[..., 0])), np.degrees(
        np.arcsin(ground[..., 2] / EARTH_RADIUS_KM)
    )
