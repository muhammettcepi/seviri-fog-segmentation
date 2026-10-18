import dataclasses
from pathlib import Path

import numpy as np
import pytest
from helpers import PROVISIONAL_DYNAMIC

from fogseg.config import load_config
from fogseg.dynamic_threshold import DynamicParams, ThresholdChoice, dynamic_threshold

PARAMS = PROVISIONAL_DYNAMIC

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def two_hump_btd(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(1.8, 0.4, 6000), rng.normal(5.0, 0.6, 6000)])


def test_two_humps_give_the_valley():
    choice = dynamic_threshold(two_hump_btd(0), fixed=2.5, params=PARAMS)
    assert choice.source == "dynamic" and 2.8 < choice.value < 4.0
    assert type(choice.value) is float


def test_one_hump_falls_back_to_fixed():
    btd = np.random.default_rng(1).normal(2.0, 0.5, 10000)
    assert dynamic_threshold(btd, 2.5, PARAMS) == ThresholdChoice(2.5, "fallback")


def test_too_few_pixels_fall_back():
    assert dynamic_threshold(np.array([1.0, 5.0]), 2.5, PARAMS).source == "fallback"


def test_tiny_second_hump_falls_back():
    rng = np.random.default_rng(2)
    btd = np.concatenate([rng.normal(1.8, 0.4, 9900), rng.normal(5.0, 0.3, 100)])
    assert dynamic_threshold(btd, 2.5, PARAMS).source == "fallback"


def test_valley_outside_the_plausible_range_is_clamped():
    narrow = dataclasses.replace(PARAMS, clamp=(1.5, 3.0))
    assert dynamic_threshold(two_hump_btd(3), 2.5, narrow) == ThresholdChoice(3.0, "dynamic")


def test_flat_valley_bottom_picks_its_middle():
    rng = np.random.default_rng(4)
    btd = np.concatenate([rng.uniform(1.0, 2.0, 6000), rng.uniform(5.0, 6.0, 6000)])
    assert dynamic_threshold(btd, 2.5, PARAMS).value == pytest.approx(3.5, abs=0.25)


def test_shallow_valley_falls_back():
    strict = dataclasses.replace(PARAMS, max_valley_ratio=0.0001)
    assert dynamic_threshold(two_hump_btd(5), 2.5, strict) == ThresholdChoice(2.5, "fallback")


def test_adjacent_peaks_fall_back():
    # Both windows meet at 3.0, so a single narrow hump there makes the two peaks neighbours.
    rng = np.random.default_rng(6)
    btd = rng.normal(3.0, 0.05, 10000)
    assert dynamic_threshold(btd, 2.5, PARAMS) == ThresholdChoice(2.5, "fallback")


def test_non_finite_samples_are_ignored():
    btd = np.concatenate([two_hump_btd(0), np.full(500, np.nan)])
    assert dynamic_threshold(btd, 2.5, PARAMS) == dynamic_threshold(two_hump_btd(0), 2.5, PARAMS)


def test_from_config_maps_every_yaml_field():
    dynamic_cfg = load_config(DEFAULT_CONFIG)["labels"]["dynamic"]
    params = DynamicParams.from_config(dynamic_cfg)
    for field in dataclasses.fields(DynamicParams):
        expected = dynamic_cfg[field.name]
        if isinstance(expected, list):
            expected = tuple(expected)
        assert getattr(params, field.name) == expected
    assert isinstance(params.bin_range, tuple)
    assert isinstance(params.clamp, tuple)
