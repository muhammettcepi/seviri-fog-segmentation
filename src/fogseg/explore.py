"""Helpers for exploring the BTD distributions before the label thresholds are fixed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fogseg.check_scenes import TRAINING_SEASON, season_of
from fogseg.dynamic_threshold import smooth_counts

# Spherical Earth with the equatorial radius SEVIRI's geostationary projection uses.
EARTH_RADIUS_KM = 6378.169
GEOSTATIONARY_ORBIT_RADIUS_KM = 42164.0

# The same light smoothing the dynamic threshold applies, so a peak found here is a peak it would see.
SMOOTHING_BINS = 3
SEASON_FIRST_MONTH = 10


def satellite_zenith_angle(lons: np.ndarray, lats: np.ndarray, satellite_lon: float = 0.0) -> np.ndarray:
    """Viewing zenith angle (degrees) of a geostationary satellite at `satellite_lon` for each pixel."""
    lat = np.radians(lats)
    lon_from_satellite = np.radians(lons - satellite_lon)
    cos_central_angle = np.cos(lat) * np.cos(lon_from_satellite)
    sin_central_angle = np.sqrt(1.0 - cos_central_angle**2)

    radius_ratio = EARTH_RADIUS_KM / GEOSTATIONARY_ORBIT_RADIUS_KM
    zenith = np.arctan2(sin_central_angle, cos_central_angle - radius_ratio)
    return np.degrees(zenith)


def training_scene_indices(times: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the scenes that belong to the training season (2023-24)."""
    is_training = np.array([season_of(time) == TRAINING_SEASON for time in times])
    return np.flatnonzero(is_training)


def months_in_season_order(times: pd.DatetimeIndex) -> list[int]:
    """The months present in `times`, October first (the fog season runs October to February)."""
    return sorted(set(times.month), key=lambda month: (month - SEASON_FIRST_MONTH) % 12)


def most_isolated_free_scene(
    candidates: pd.DatetimeIndex, month: int, slot: str, taken: Sequence[pd.Timestamp]
) -> pd.Timestamp:
    """The scene of `month` at `slot` that lies farthest in time from every scene already taken.

    Spreading the scenes in time keeps neighbouring slots of one night (nearly identical) apart.
    """
    in_month_and_slot = (candidates.month == month) & (candidates.strftime("%H:%M") == slot)
    free = candidates[in_month_and_slot].difference(pd.DatetimeIndex(taken))
    if len(free) == 0:
        raise ValueError(f"no free scene in month {month} at {slot}")
    if len(taken) == 0:
        return free[len(free) // 2]

    gaps_to_taken = np.abs(free.values[:, None] - pd.DatetimeIndex(taken).values[None, :])
    distance_to_nearest_taken = gaps_to_taken.min(axis=1)
    return free[int(np.argmax(distance_to_nearest_taken))]


def pick_exploration_scenes(
    times: pd.DatetimeIndex,
    n_scenes: int,
    slots: Sequence[str],
    must_include: Sequence[pd.Timestamp] = (),
) -> pd.DatetimeIndex:
    """`n_scenes` scenes spread evenly over the months and the night slots, sorted by time.

    Each new scene comes from the month and the slot that have been used least so far, so the
    choice depends only on the times and is the same on every run.
    """
    candidates = times.sort_values()
    months = months_in_season_order(candidates)
    picked = list(must_include)

    while len(picked) < n_scenes:
        month_counts = Counter(time.month for time in picked)
        slot_counts = Counter(time.strftime("%H:%M") for time in picked)
        month = min(months, key=lambda m: (month_counts[m], months.index(m)))
        slot = min(slots, key=lambda s: (slot_counts[s], slots.index(s)))
        picked.append(most_isolated_free_scene(candidates, month, slot, picked))

    return pd.DatetimeIndex(sorted(picked))


def hump_peak(
    btd: np.ndarray,
    window: tuple[float, float],
    bin_width: float = 0.25,
    min_pixels: int = 1000,
) -> float:
    """BTD of the highest bin of the smoothed histogram inside `window`; NaN if there is no real peak.

    There is no real peak when there are too few pixels, or when the highest bin is the first or
    last one of the window: then the hump continues outside the window (or there is none).
    """
    samples = btd[np.isfinite(btd)]
    if samples.size < min_pixels:
        return float("nan")

    # The histogram reaches a bit beyond the window, so the smoothing at the window edge still
    # sees the pixels next to it; otherwise a hump rising at the edge would look like a peak.
    margin_bins = SMOOTHING_BINS // 2
    margin = margin_bins * bin_width
    padded_window = (window[0] - margin, window[1] + margin)
    n_bins = round((padded_window[1] - padded_window[0]) / bin_width)
    counts, edges = np.histogram(samples, bins=n_bins, range=padded_window)
    smoothed = smooth_counts(counts, SMOOTHING_BINS)

    inside_window = smoothed[margin_bins : n_bins - margin_bins]
    top = int(np.argmax(inside_window))
    if top in (0, len(inside_window) - 1):
        return float("nan")
    top_bin = top + margin_bins
    return float((edges[top_bin] + edges[top_bin + 1]) / 2)


def valley_between(
    btd: np.ndarray,
    left_peak: float,
    right_peak: float,
    bin_width: float = 0.25,
    bin_range: tuple[float, float] = (-4.0, 10.0),
) -> tuple[float, float]:
    """BTD of the lowest point of the smoothed histogram between two peaks, and how deep it is.

    The depth is the height of the valley divided by the height of the smaller peak: near 0 for
    two separate humps, near 1 when the humps merge into one.
    """
    samples = btd[np.isfinite(btd)]
    n_bins = round((bin_range[1] - bin_range[0]) / bin_width)
    counts, edges = np.histogram(samples, bins=n_bins, range=bin_range)
    centres = (edges[:-1] + edges[1:]) / 2
    smoothed = smooth_counts(counts, SMOOTHING_BINS)

    between = np.flatnonzero((centres > left_peak) & (centres < right_peak))
    if between.size == 0:
        raise ValueError("there is no histogram bin between the two peaks")
    valley = between[np.argmin(smoothed[between])]

    left_height = smoothed[np.argmin(np.abs(centres - left_peak))]
    right_height = smoothed[np.argmin(np.abs(centres - right_peak))]
    depth = smoothed[valley] / min(left_height, right_height)
    return float(centres[valley]), float(depth)


def peaks_by_group(
    btd: np.ndarray,
    group: np.ndarray,
    n_groups: int,
    window: tuple[float, float],
    bin_width: float = 0.25,
    min_pixels: int = 1000,
) -> np.ndarray:
    """`hump_peak` of the pixels of each group 0 .. n_groups - 1; pixels with another code are skipped."""
    peaks = np.full(n_groups, np.nan)
    for code in range(n_groups):
        peaks[code] = hump_peak(btd[group == code], window, bin_width, min_pixels)
    return peaks
