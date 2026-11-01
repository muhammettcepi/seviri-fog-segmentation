import dataclasses
import warnings

import numpy as np
import pytest
from helpers import TEST_PARAMS

from fogseg.config import load_config
from fogseg.labels import (
    FOG,
    IGNORE,
    NO_FOG,
    OBSCURED,
    LabelParams,
    emissivity_39,
    emissivity_btd_threshold,
    histogram_pixels,
    label_scene,
    local_std,
    small_blobs,
    trusted_mask,
    warm_and_not_cirrus,
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
    btd = histogram_pixels(ir039, ir108, ir120, sza, TEST_PARAMS)
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
    return label_scene(ir039_field, ir108_field, ir120_field, sza_field, btd_threshold, TEST_PARAMS)


def test_warm_high_btd_pixel_is_fog():
    assert scene(btd=4.0)[2, 2] == FOG


def test_cold_top_is_obscured():
    assert scene(btd=4.0, ir108=230.0)[2, 2] == OBSCURED


def test_thin_cirrus_is_obscured():
    assert scene(btd=4.0, split=3.0)[2, 2] == OBSCURED


def test_low_btd_pixel_is_not_fog():
    assert scene(btd=1.0)[2, 2] == NO_FOG


def test_borderline_btd_above_the_threshold_is_ignored():
    assert scene(btd=2.5 + 0.3, btd_threshold=2.5)[2, 2] == IGNORE


def test_borderline_btd_below_the_threshold_is_ignored():
    assert scene(btd=2.5 - 0.3, btd_threshold=2.5)[2, 2] == IGNORE


def test_btd_clearly_below_the_threshold_is_not_fog():
    assert scene(btd=2.5 - 0.7, btd_threshold=2.5)[2, 2] == NO_FOG


def test_borderline_btd_below_the_threshold_of_a_cold_top_is_obscured():
    assert scene(btd=2.5 - 0.3, ir108=230.0, btd_threshold=2.5)[2, 2] == OBSCURED


def test_borderline_btd_below_the_threshold_of_thin_cirrus_is_obscured():
    assert scene(btd=2.5 - 0.3, split=3.0, btd_threshold=2.5)[2, 2] == OBSCURED


def test_borderline_btd_below_the_threshold_of_a_rough_top_is_not_fog():
    assert scene(btd=2.5 - 0.3, checkerboard_amplitude=3.0, btd_threshold=2.5)[2, 2] == NO_FOG


def test_rough_top_is_not_fog():
    labels = scene(btd=4.0, checkerboard_amplitude=3.0)
    assert labels[2, 2] == NO_FOG


def test_cold_or_cirrus_trusted_pixel_is_obscured_and_warm_ground_is_not():
    # pixel 0: cold top (IR_108 250 K); pixel 1: cirrus split 3 K; pixel 2: clear warm ground
    ir108 = np.array([[250.0, 270.0, 270.0]])
    ir039 = ir108 - 1.0
    ir120 = ir108 - np.array([[0.5, 3.0, 0.5]])
    sza = np.full_like(ir108, 120.0)
    labels = label_scene(ir039, ir108, ir120, sza, 4.0, TEST_PARAMS)
    assert labels.tolist() == [[OBSCURED, OBSCURED, NO_FOG]]


def test_untrusted_cold_pixel_stays_ignored():
    ir108 = np.full((1, 1), 250.0)
    labels = label_scene(ir108 - 1.0, ir108, ir108 - 0.5, np.full_like(ir108, 80.0), 4.0, TEST_PARAMS)
    assert labels[0, 0] == IGNORE


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
    labels = label_scene(ir039, ir108, ir120, sza, 2.5, TEST_PARAMS)
    assert labels[1, 2] == IGNORE and labels[5, 3] == FOG and labels[3, 3] == NO_FOG


def scene_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An 8 x 12 night scene with every label value: a fog deck, a borderline strip, an island and more."""
    ir108 = np.full((8, 12), 270.0)
    ir039 = ir108 - 0.5
    ir120 = ir108 - 0.5
    sza = np.full((8, 12), 120.0)

    ir039[2:5, 0:5] = ir108[2:5, 0:5] - 5.0  # fog deck of 15 pixels
    ir039[7, 0:3] = ir108[7, 0:3] - 4.2  # BTD within the uncertainty band of 4.0 K
    ir039[0, 8:10] = ir108[0, 8:10] - 5.0  # fog island of 2 pixels
    ir108[0:3, 11] = 230.0  # cold top
    ir039[0:3, 11] = ir108[0:3, 11] - 0.5
    ir120[5, 8:10] = ir108[5, 8:10] - 3.0  # thin cirrus
    ir108[6:8, 4:7] += np.array([[3.0, -3.0, 3.0], [-3.0, 3.0, -3.0]])  # rough top
    ir039[6:8, 4:7] = ir108[6:8, 4:7] - 5.0
    ir120[6:8, 4:7] = ir108[6:8, 4:7] - 0.5
    ir039[6, 6] = np.nan  # missing channel
    sza[7, 10:12] = 80.0  # daytime
    return ir039, ir108, ir120, sza


EXPECTED_LABELS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0],
        [255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255],
    ],
    dtype=np.uint8,
)

BASE_FIELDS = (
    "night_min_sza",
    "btd_fixed",
    "warm_min_bt108",
    "cirrus_max_split",
    "smooth_max_std",
    "uncertainty_band",
    "min_blob_pixels",
)


def uniform_top_scene(ir108: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A smooth 6 x 6 night scene with one IR_108 value and BTDs of 6.0, 2.0 and 4.3 K in column pairs."""
    btd = np.tile(np.repeat([6.0, 2.0, 4.3], 2), (6, 1))
    ir108_field = np.full((6, 6), ir108)
    return ir108_field - btd, ir108_field, ir108_field - 0.5, np.full((6, 6), 120.0)


def test_defaults_reproduce_the_labels():
    ir039, ir108, ir120, sza = scene_arrays()
    old_style = LabelParams(**{name: getattr(TEST_PARAMS, name) for name in BASE_FIELDS})
    assert np.array_equal(label_scene(ir039, ir108, ir120, sza, 4.0, old_style), EXPECTED_LABELS)


def test_from_config_without_the_new_keys_takes_the_defaults():
    params = LabelParams.from_config({name: getattr(TEST_PARAMS, name) for name in BASE_FIELDS})
    assert (params.texture, params.fog_index, params.emissivity_max) == ("ir108_std3", "btd", None)
    assert (params.split_window, params.temporal_relax, params.temporal_max_dt108) == ("fixed", 0.0, 2.0)


def test_unknown_option_names_the_allowed_values():
    with pytest.raises(ValueError, match="ir108_mad5"):
        dataclasses.replace(TEST_PARAMS, texture="sobel")


def test_kma_split_window_follows_the_temperature():
    ir108 = np.array([280.0, 280.0])
    ir120 = ir108 - np.array([0.9, -0.2])  # expected split at 280 K is -0.268 K
    fixed = warm_and_not_cirrus(ir108, ir120, TEST_PARAMS)
    kma = warm_and_not_cirrus(ir108, ir120, dataclasses.replace(TEST_PARAMS, split_window="kma"))
    assert fixed.tolist() == [True, True] and kma.tolist() == [False, True]


def test_emissivity_threshold_is_the_btd_with_that_emissivity():
    ir108 = np.array([270.0, 285.0])
    emissivity = emissivity_39(ir108 - 4.0, ir108)
    assert emissivity_btd_threshold(ir108, float(emissivity[0]))[0] == pytest.approx(4.0, abs=1e-6)
    assert emissivity[1] > emissivity[0]  # the same BTD is a weaker signal on a warmer top


def test_emissivity_index_labels_like_the_btd_on_a_uniform_top():
    ir039, ir108, ir120, sza = uniform_top_scene(ir108=280.0)
    threshold = float(emissivity_39(np.array([276.0]), np.array([280.0]))[0])
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    emissivity_labels = label_scene(ir039, ir108, ir120, sza, threshold, params)
    assert np.array_equal(emissivity_labels, label_scene(ir039, ir108, ir120, sza, 4.0, TEST_PARAMS))
    assert set(emissivity_labels.ravel()) == {FOG, NO_FOG, IGNORE}
