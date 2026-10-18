import dataclasses
import warnings

import numpy as np
import pytest

from fogseg.config import load_config
from fogseg.labels import (
    FOG,
    IGNORE,
    NO_FOG,
    LabelParams,
    histogram_pixels,
    label_scene,
    local_std,
    small_blobs,
    trusted_mask,
)

PROVISIONAL = LabelParams(
    night_min_sza=95.0,
    btd_fixed=2.5,
    warm_min_bt108=260.0,
    cirrus_max_split=2.0,
    smooth_max_std=1.5,
    uncertainty_band=0.5,
    min_blob_pixels=5,
)


def test_params_from_config_reads_every_threshold():
    labels_cfg = load_config("configs/default.yaml")["labels"]
    params = LabelParams.from_config(labels_cfg)
    for field in dataclasses.fields(LabelParams):
        assert getattr(params, field.name) == labels_cfg[field.name]


def test_params_from_config_ignores_non_threshold_keys():
    labels_cfg = load_config("configs/default.yaml")["labels"]
    non_threshold_keys = {"mode", "output_dir", "figures_dir", "dynamic"}
    assert non_threshold_keys <= set(labels_cfg)
    params = LabelParams.from_config(labels_cfg)
    field_names = {field.name for field in dataclasses.fields(params)}
    assert field_names == set(labels_cfg) - non_threshold_keys


def test_daytime_and_twilight_pixels_are_not_trusted():
    sza = np.array([[80.0, 95.0, 95.1]])
    bt = np.full((1, 3), 270.0)
    assert trusted_mask(bt, bt, bt, sza, 95.0).tolist() == [[False, False, True]]


def test_missing_channel_value_is_not_trusted():
    ir039 = np.array([[np.nan, 270.0]])
    bt = np.full((1, 2), 270.0)
    sza = np.full((1, 2), 120.0)
    assert trusted_mask(ir039, bt, bt, sza, 95.0).tolist() == [[False, True]]


def test_histogram_pixels_keep_only_warm_non_cirrus_night_pixels():
    # columns: valid fog-like, cold top, thin cirrus, daytime
    ir108 = np.array([[270.0, 230.0, 270.0, 270.0]])
    ir039 = ir108 - 3.0
    ir120 = np.array([[269.5, 229.5, 266.0, 269.5]])
    sza = np.array([[120.0, 120.0, 120.0, 80.0]])
    btd = histogram_pixels(ir039, ir108, ir120, sza, PROVISIONAL)
    assert btd.tolist() == [3.0]


def scene(
    btd: float,
    ir108: float = 270.0,
    split: float = 0.5,
    sza: float = 120.0,
    btd_threshold: float = 2.5,
    checkerboard_amplitude: float = 0.0,
) -> np.ndarray:
    """Label a uniform 5x5 scene described by its BTD, IR_108, split-window difference and sun angle."""
    rows, cols = np.indices((5, 5))
    checkerboard_sign = np.where((rows + cols) % 2 == 0, 1.0, -1.0)
    ir108_field = np.full((5, 5), ir108) + checkerboard_amplitude * checkerboard_sign
    ir039_field = ir108_field - btd
    ir120_field = ir108_field - split
    sza_field = np.full((5, 5), sza)
    return label_scene(ir039_field, ir108_field, ir120_field, sza_field, btd_threshold, PROVISIONAL)


def test_warm_high_btd_pixel_is_fog():
    assert scene(btd=4.0)[2, 2] == FOG


def test_cold_top_is_not_fog():
    assert scene(btd=4.0, ir108=230.0)[2, 2] == NO_FOG


def test_thin_cirrus_is_not_fog():
    assert scene(btd=4.0, split=3.0)[2, 2] == NO_FOG


def test_low_btd_pixel_is_not_fog():
    assert scene(btd=1.0)[2, 2] == NO_FOG


def test_borderline_btd_is_ignored():
    assert scene(btd=2.5 + 0.3, btd_threshold=2.5)[2, 2] == IGNORE


def test_rough_top_is_not_fog():
    labels = scene(btd=4.0, checkerboard_amplitude=3.0)
    assert labels[2, 2] == NO_FOG


def test_labels_are_uint8():
    assert scene(btd=4.0).dtype == np.uint8


def test_local_std_ignores_missing_neighbours():
    field = np.full((3, 3), 270.0)
    field[0, 0] = np.nan
    assert local_std(field)[1, 1] == pytest.approx(0.0)


def test_local_std_matches_numpy_in_a_full_window():
    rng = np.random.default_rng(0)
    field = rng.normal(270.0, 2.0, size=(5, 5))
    assert local_std(field)[2, 2] == pytest.approx(np.std(field[1:4, 1:4]))


def test_local_std_of_all_missing_field_is_nan_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_std(np.full((4, 4), np.nan))
    assert np.isnan(result).all()


def test_three_pixel_island_is_ignored_but_ten_pixel_deck_stays_fog():
    mask = np.zeros((8, 12), bool)
    mask[1, 1:4] = True  # 3 pixels
    mask[5:7, 2:7] = True  # 10 pixels
    assert small_blobs(mask, 5)[1, 2] and not small_blobs(mask, 5)[5, 3]


def test_diagonal_pixels_form_one_blob():
    mask = np.eye(6, dtype=bool)  # 6 pixels touching only at corners
    assert not small_blobs(mask, 5).any()


def test_deck_touching_the_image_edge_is_kept():
    mask = np.zeros((8, 12), bool)
    mask[0:2, 7:12] = True  # 10 pixels along the top edge
    assert not small_blobs(mask, 5).any()


def test_scene_without_night_pixels_is_all_ignored():
    assert (scene(btd=4.0, sza=80.0) == IGNORE).all()


def test_small_fog_island_in_a_scene_is_ignored():
    ir108 = np.full((8, 8), 270.0)
    ir039 = ir108 - 0.5
    ir039[1, 1:4] = ir108[1, 1:4] - 4.0  # 3 fog pixels
    ir039[5:7, 2:7] = ir108[5:7, 2:7] - 4.0  # 10 fog pixels
    ir120 = ir108 - 0.5
    sza = np.full((8, 8), 120.0)
    labels = label_scene(ir039, ir108, ir120, sza, 2.5, PROVISIONAL)
    assert labels[1, 2] == IGNORE and labels[5, 3] == FOG and labels[3, 3] == NO_FOG
