"""Tests for the helpers of the BTD exploration notebook."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from helpers import TEST_DYNAMIC

from fogseg.dynamic_threshold import clear_ground_peak
from fogseg.explore import (
    hump_peak,
    peaks_by_group,
    pick_exploration_scenes,
    satellite_zenith_angle,
    training_scene_indices,
    valley_between,
)

SLOTS = ("21:00", "00:00", "03:00", "05:00")
FOGGY_SCENE = pd.Timestamp("2023-11-21 00:00")
CLEAR_WINDOW = (0.0, 5.0)


def test_satellite_zenith_is_zero_below_the_satellite():
    assert satellite_zenith_angle(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_satellite_zenith_at_50_north_on_the_meridian():
    # geostationary at 0°E, Earth radius 6378.169 km, orbit radius 42164.0 km
    assert satellite_zenith_angle(np.array([0.0]), np.array([50.0]))[0] == pytest.approx(57.3, abs=0.2)


def every_second_night(first_evening: str, last_evening: str) -> pd.DatetimeIndex:
    """Scene times like the real dataset: every second night, at 21, 00, 03 and 05 UTC."""
    evenings = pd.date_range(first_evening, last_evening, freq="2D")
    offsets = [pd.Timedelta(hours=hours) for hours in (21, 24, 27, 29)]
    times = [evening + offset for evening in evenings for offset in offsets]
    return pd.DatetimeIndex(times)


def test_training_scene_indices_keep_only_the_2023_24_season():
    times = every_second_night("2023-10-01", "2023-10-03").append(
        every_second_night("2024-10-01", "2024-10-03")
    )
    indices = training_scene_indices(times)
    assert list(indices) == list(range(8))
    assert indices.dtype.kind == "i"


def test_pick_exploration_scenes_spreads_over_months_and_slots():
    times = every_second_night("2023-10-01", "2024-02-28")
    picked = pick_exploration_scenes(times, 12, SLOTS, must_include=[FOGGY_SCENE])

    assert len(picked) == 12
    assert FOGGY_SCENE in picked
    assert picked.isin(times).all()
    assert set(picked.month) == {10, 11, 12, 1, 2}
    slot_counts = pd.Series(picked.strftime("%H:%M")).value_counts()
    assert all(slot_counts[slot] == 3 for slot in SLOTS)


def test_pick_exploration_scenes_avoids_scenes_of_the_same_night():
    times = every_second_night("2023-10-01", "2024-02-28")
    picked = pick_exploration_scenes(times, 12, SLOTS, must_include=[FOGGY_SCENE])
    gaps = picked[1:] - picked[:-1]
    assert gaps.min() >= pd.Timedelta(days=1)


def test_pick_exploration_scenes_is_deterministic_and_sorted():
    times = every_second_night("2023-10-01", "2024-02-28")
    first = pick_exploration_scenes(times, 12, SLOTS, must_include=[FOGGY_SCENE])
    second = pick_exploration_scenes(times[::-1], 12, SLOTS, must_include=[FOGGY_SCENE])
    assert first.equals(second)
    assert first.is_monotonic_increasing


def two_hump_btd(seed: int, clear_centre: float = 2.1, cloud_centre: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clear = rng.normal(clear_centre, 0.4, 8000)
    cloud = rng.normal(cloud_centre, 0.6, 4000)
    return np.concatenate([clear, cloud])


def test_hump_peak_finds_the_mode_of_the_bigger_hump_in_the_window():
    assert hump_peak(two_hump_btd(0), CLEAR_WINDOW) == pytest.approx(2.1, abs=0.1)


def test_hump_peak_ignores_a_bigger_hump_outside_the_window():
    rng = np.random.default_rng(1)
    small_clear_hump = rng.normal(2.6, 0.4, 2000)
    big_cloud_hump = rng.normal(7.0, 0.6, 8000)
    btd = np.concatenate([small_clear_hump, big_cloud_hump])
    assert hump_peak(btd, CLEAR_WINDOW) == pytest.approx(2.6, abs=0.1)


def test_hump_peak_is_nan_when_the_top_lies_on_the_window_edge():
    only_cloud = np.random.default_rng(2).normal(6.0, 0.6, 10000)
    assert np.isnan(hump_peak(only_cloud, CLEAR_WINDOW))


def test_hump_peak_is_nan_for_too_few_pixels():
    assert np.isnan(hump_peak(np.array([2.0, 2.1, 2.2]), CLEAR_WINDOW, min_pixels=10))


def test_hump_peak_agrees_with_the_production_peak_finder():
    params = dataclasses.replace(TEST_DYNAMIC, clear_window=CLEAR_WINDOW)
    for seed in range(5):
        btd = two_hump_btd(seed)
        assert hump_peak(btd, CLEAR_WINDOW) == clear_ground_peak(btd, params)


def test_valley_between_two_clear_humps_is_low_and_in_the_middle():
    valley_btd, depth = valley_between(two_hump_btd(6), left_peak=2.1, right_peak=6.0)
    assert 3.0 < valley_btd < 4.5
    assert depth < 0.1


def test_valley_between_overlapping_humps_is_shallow():
    rng = np.random.default_rng(7)
    btd = np.concatenate([rng.normal(3.0, 0.8, 8000), rng.normal(5.0, 0.8, 6000)])
    valley_btd, depth = valley_between(btd, left_peak=3.0, right_peak=5.0)
    assert valley_btd == pytest.approx(4.0, abs=0.5)
    assert depth > 0.6


def test_valley_between_peaks_with_no_bin_between_them_is_an_error():
    with pytest.raises(ValueError, match="no histogram bin"):
        valley_between(two_hump_btd(8), left_peak=3.0, right_peak=3.1)


def test_peaks_by_group_finds_each_groups_hump():
    low_group = two_hump_btd(3, clear_centre=1.5)
    high_group = two_hump_btd(4, clear_centre=3.0)
    btd = np.concatenate([low_group, high_group])
    group = np.concatenate([np.zeros(low_group.size, dtype=int), np.ones(high_group.size, dtype=int)])

    peaks = peaks_by_group(btd, group, n_groups=3, window=CLEAR_WINDOW)

    np.testing.assert_allclose(peaks[:2], [1.5, 3.0], atol=0.13)  # half a bin
    assert np.isnan(peaks[2])


def test_peaks_by_group_skips_negative_group_codes():
    btd = two_hump_btd(5, clear_centre=2.0)
    group = np.full(btd.size, -1)
    assert np.isnan(peaks_by_group(btd, group, n_groups=1, window=CLEAR_WINDOW)[0])
