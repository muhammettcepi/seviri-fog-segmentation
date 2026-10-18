"""Helpers for exploring the BTD distributions before the label thresholds are fixed."""

from __future__ import annotations

import numpy as np

# Spherical Earth with the equatorial radius SEVIRI's geostationary projection uses.
EARTH_RADIUS_KM = 6378.169
GEOSTATIONARY_ORBIT_RADIUS_KM = 42164.0


def satellite_zenith_angle(lons: np.ndarray, lats: np.ndarray, satellite_lon: float = 0.0) -> np.ndarray:
    """Viewing zenith angle (degrees) of a geostationary satellite at `satellite_lon` for each pixel."""
    lat = np.radians(lats)
    lon_from_satellite = np.radians(lons - satellite_lon)
    cos_central_angle = np.cos(lat) * np.cos(lon_from_satellite)
    sin_central_angle = np.sqrt(1.0 - cos_central_angle**2)

    radius_ratio = EARTH_RADIUS_KM / GEOSTATIONARY_ORBIT_RADIUS_KM
    zenith = np.arctan2(sin_central_angle, cos_central_angle - radius_ratio)
    return np.degrees(zenith)
