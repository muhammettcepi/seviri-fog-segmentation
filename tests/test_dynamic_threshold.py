import dataclasses
from pathlib import Path

import numpy as np
import pytest
from helpers import PROVISIONAL_DYNAMIC

from fogseg.config import load_config
from fogseg.dynamic_threshold import DynamicParams, ThresholdChoice, clear_ground_peak, dynamic_threshold

PARAMS = PROVISIONAL_DYNAMIC
FIXED = 5.0
ONE_BIN = PARAMS.bin_width

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def clear_hump_with_cloud_shoulder(seed: int, ground_btd: float = 3.0) -> np.ndarray:
    """A clear-ground hump at `ground_btd` and a smaller, wider low-cloud shoulder 2.5 K above it."""
    rng = np.random.default_rng(seed)
    clear_ground = rng.normal(ground_btd, 0.4, 8000)
    low_cloud = rng.normal(ground_btd + 2.5, 0.8, 3000)
    return np.concatenate([clear_ground, low_cloud])


def only_cloud_hump(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(6.0, 0.6, 10000)


def test_threshold_is_clear_ground_peak_plus_offset():
    choice = dynamic_threshold(clear_hump_with_cloud_shoulder(0), FIXED, PARAMS)
    assert choice.source == "dynamic"
    assert choice.value == pytest.approx(3.0 + PARAMS.offset, abs=ONE_BIN)
    assert type(choice.value) is float


@pytest.mark.parametrize("shift", [-1.0, 1.0])
def test_threshold_follows_the_ground_level(shift):
    unclamped = dataclasses.replace(PARAMS, clamp=(0.0, 10.0))
    base = dynamic_threshold(clear_hump_with_cloud_shoulder(1), FIXED, unclamped)
    shifted = dynamic_threshold(clear_hump_with_cloud_shoulder(1, ground_btd=3.0 + shift), FIXED, unclamped)
    assert shifted.source == "dynamic"
    assert shifted.value - base.value == pytest.approx(shift, abs=ONE_BIN)


def test_too_few_pixels_falls_back():
    btd = clear_hump_with_cloud_shoulder(2)[: PARAMS.min_pixels - 1]
    assert dynamic_threshold(btd, FIXED, PARAMS) == ThresholdChoice(FIXED, "fallback")


def test_peak_on_the_window_edge_falls_back():
    assert dynamic_threshold(only_cloud_hump(3), FIXED, PARAMS) == ThresholdChoice(FIXED, "fallback")


def test_threshold_is_clamped_low():
    raised_floor = dataclasses.replace(PARAMS, clamp=(5.0, 6.0))
    choice = dynamic_threshold(clear_hump_with_cloud_shoulder(4), FIXED, raised_floor)
    assert choice == ThresholdChoice(5.0, "dynamic")


def test_threshold_is_clamped_high():
    lowered_ceiling = dataclasses.replace(PARAMS, clamp=(3.5, 4.0))
    choice = dynamic_threshold(clear_hump_with_cloud_shoulder(5), FIXED, lowered_ceiling)
    assert choice == ThresholdChoice(4.0, "dynamic")


def test_non_finite_samples_are_ignored():
    finite = clear_hump_with_cloud_shoulder(6)[: PARAMS.min_pixels - 1]
    with_nans = np.concatenate([finite, np.full(PARAMS.min_pixels, np.nan)])
    assert with_nans.size >= PARAMS.min_pixels
    assert dynamic_threshold(with_nans, FIXED, PARAMS) == ThresholdChoice(FIXED, "fallback")


def test_non_finite_samples_do_not_change_the_threshold():
    btd = clear_hump_with_cloud_shoulder(7)
    with_nans = np.concatenate([btd, np.full(500, np.nan), np.full(500, np.inf)])
    assert dynamic_threshold(with_nans, FIXED, PARAMS) == dynamic_threshold(btd, FIXED, PARAMS)


def test_clear_ground_peak_is_the_bin_centre_of_the_hump_top():
    peak = clear_ground_peak(clear_hump_with_cloud_shoulder(8), PARAMS)
    assert peak == pytest.approx(3.0, abs=ONE_BIN)
    bins_from_range_start = (peak - PARAMS.bin_range[0]) / PARAMS.bin_width
    assert bins_from_range_start % 1 == pytest.approx(0.5)


def test_clear_ground_peak_ignores_a_bigger_hump_above_the_window():
    rng = np.random.default_rng(9)
    small_clear_hump = rng.normal(2.6, 0.4, 2000)
    big_cloud_hump = rng.normal(7.0, 0.4, 8000)
    peak = clear_ground_peak(np.concatenate([small_clear_hump, big_cloud_hump]), PARAMS)
    assert peak == pytest.approx(2.6, abs=ONE_BIN)


def test_clear_ground_peak_is_nan_when_the_top_is_the_last_bin_of_the_window():
    assert np.isnan(clear_ground_peak(only_cloud_hump(10), PARAMS))


def test_clear_ground_peak_is_nan_when_the_hump_continues_below_the_window():
    below = np.random.default_rng(11).normal(0.5, 0.5, 10000)
    assert np.isnan(clear_ground_peak(below, PARAMS))


def test_from_config_maps_every_yaml_field():
    dynamic_cfg = load_config(DEFAULT_CONFIG)["labels"]["dynamic"]
    params = DynamicParams.from_config(dynamic_cfg)
    for field in dataclasses.fields(DynamicParams):
        expected = dynamic_cfg[field.name]
        if isinstance(expected, list):
            expected = tuple(expected)
        assert getattr(params, field.name) == expected
    assert isinstance(params.bin_range, tuple)
    assert isinstance(params.clear_window, tuple)
    assert isinstance(params.clamp, tuple)


def test_params_have_exactly_the_clear_ground_peak_fields():
    names = [field.name for field in dataclasses.fields(DynamicParams)]
    assert names == [
        "bin_width",
        "bin_range",
        "smoothing_bins",
        "clear_window",
        "offset",
        "min_pixels",
        "clamp",
    ]
