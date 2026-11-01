import dataclasses
import warnings

import numpy as np
import pytest
from helpers import BASE_FIELDS, TEST_PARAMS

from fogseg.config import load_config
from fogseg.labels import (
    FOG,
    IGNORE,
    NO_FOG,
    OBSCURED,
    TEXTURES,
    LabelParams,
    emissivity_39,
    emissivity_btd_threshold,
    histogram_pixels,
    label_night,
    label_scene,
    local_mad,
    local_std,
    small_blobs,
    texture_ok,
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


def rough_field() -> np.ndarray:
    """A 6 x 6 IR_108 checkerboard whose 3 x 3 standard deviation is about 3 K everywhere."""
    rows, cols = np.indices((6, 6))
    return 270.0 + 3.0 * np.where((rows + cols) % 2 == 0, 1.0, -1.0)


LEFT_BLOB = (slice(2, 5), slice(1, 5))
RIGHT_BLOB = (slice(2, 5), slice(8, 12))


def two_blobs() -> tuple[np.ndarray, np.ndarray]:
    """An IR_108 field and a region of two groups: a smooth one with a hot edge pixel and a rough one."""
    rows, cols = np.indices((8, 14))
    ir108 = np.full((8, 14), 270.0)
    ir108[RIGHT_BLOB] = 270.0 + 3.0 * np.where((rows + cols) % 2 == 0, 1.0, -1.0)[RIGHT_BLOB]
    ir108[2, 1] = 280.0
    region = np.zeros((8, 14), dtype=bool)
    region[LEFT_BLOB] = True
    region[RIGHT_BLOB] = True
    return ir108, region


def test_mad5_ignores_one_hot_pixel_that_fails_std3():
    ir108 = np.full((7, 7), 270.0)
    ir108[3, 3] = 276.0
    btd = np.full((7, 7), 5.0)
    region = np.ones((7, 7), dtype=bool)
    std3 = texture_ok(ir108, btd, region, TEST_PARAMS)
    mad5 = texture_ok(ir108, btd, region, dataclasses.replace(TEST_PARAMS, texture="ir108_mad5"))
    assert not std3[3, 3] and mad5.all()


def test_btd_std3_looks_at_the_btd_not_ir108():
    ir108 = rough_field()
    btd = np.full(ir108.shape, 5.0)
    params = dataclasses.replace(TEST_PARAMS, texture="btd_std3")
    assert texture_ok(ir108, btd, np.ones(ir108.shape, dtype=bool), params).all()


def test_blob_judges_each_group_by_its_median_spread():
    ir108, region = two_blobs()
    params = dataclasses.replace(TEST_PARAMS, texture="blob")
    passed = texture_ok(ir108, np.zeros_like(ir108), region, params)
    assert passed[LEFT_BLOB].all() and not passed[RIGHT_BLOB].any() and not passed[~region].any()


def test_blob_texture_of_an_empty_region_fails_everywhere():
    ir108 = rough_field()
    params = dataclasses.replace(TEST_PARAMS, texture="blob")
    assert not texture_ok(ir108, ir108, np.zeros(ir108.shape, dtype=bool), params).any()


def test_no_texture_test_passes_everything():
    params = dataclasses.replace(TEST_PARAMS, texture="none")
    assert texture_ok(rough_field(), np.zeros((6, 6)), np.zeros((6, 6), dtype=bool), params).all()


def test_local_mad_is_nan_wherever_a_missing_value_reaches_it(recwarn):
    field = np.full((13, 13), 270.0)
    field[6, 6] = np.nan
    mad = local_mad(field)
    assert np.isnan(mad[2:11, 2:11]).all() and np.isfinite(mad[0, 0]) and np.isfinite(mad[12, 12])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_nan_channel_stays_ignored_under_every_texture(recwarn):
    ir039, ir108, ir120, sza = scene_arrays()
    ir108[2, 2] = np.nan
    for texture in TEXTURES:
        labels = label_scene(ir039, ir108, ir120, sza, 4.0, dataclasses.replace(TEST_PARAMS, texture=texture))
        assert labels[2, 2] == IGNORE
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_emissivity_index_labels_like_the_btd_on_a_uniform_top():
    ir039, ir108, ir120, sza = uniform_top_scene(ir108=280.0)
    threshold = float(emissivity_39(np.array([276.0]), np.array([280.0]))[0])
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    emissivity_labels = label_scene(ir039, ir108, ir120, sza, threshold, params)
    assert np.array_equal(emissivity_labels, label_scene(ir039, ir108, ir120, sza, 4.0, TEST_PARAMS))
    assert set(emissivity_labels.ravel()) == {FOG, NO_FOG, IGNORE}


def test_nan_channel_stays_ignored_under_the_emissivity_index(recwarn):
    ir039, ir108, ir120, sza = scene_arrays()
    ir108[2, 2] = np.nan
    ir039[3, 3] = np.nan
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    labels = label_scene(ir039, ir108, ir120, sza, 0.85, params)
    assert labels[2, 2] == IGNORE and labels[3, 3] == IGNORE
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def night_arrays(
    btd: list[float], ir108: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A night of 9 x 9 scenes of warm ground, each with a central 3 x 3 block of the given BTD and IR_108."""
    n_scenes = len(btd)
    ir108_night = np.full((n_scenes, 9, 9), 270.0)
    ir039_night = ir108_night - 0.5
    for s in range(n_scenes):
        ir108_night[s, 3:6, 3:6] = ir108[s]
        ir039_night[s, 3:6, 3:6] = ir108[s] - btd[s]
    return ir039_night, ir108_night, ir108_night - 0.5, np.full((n_scenes, 9, 9), 120.0)


def relaxed(temporal_relax: float) -> LabelParams:
    return dataclasses.replace(TEST_PARAMS, temporal_relax=temporal_relax)


def test_hysteresis_turns_a_near_miss_next_to_fog_into_fog():
    night = night_arrays(btd=[5.0, 3.3, 3.3], ir108=[270.0, 270.5, 270.5])
    labels = label_night(*night, np.full(3, 4.0), relaxed(1.0))
    assert labels[:, 4, 4].tolist() == [FOG, FOG, NO_FOG]  # scene 2 only sees the strict label of scene 1


def test_hysteresis_needs_a_steady_top():
    night = night_arrays(btd=[5.0, 3.3], ir108=[270.0, 273.0])
    labels = label_night(*night, np.full(2, 4.0), relaxed(1.0))
    assert labels[1, 4, 4] == NO_FOG


def test_no_relax_gives_the_scene_labels():
    night = night_arrays(btd=[5.0, 3.7, 4.2], ir108=[270.0, 270.0, 270.0])
    labels = label_night(*night, np.full(3, 4.0), TEST_PARAMS)
    for s in range(3):
        assert np.array_equal(labels[s], label_scene(*(a[s] for a in night), 4.0, TEST_PARAMS))


def test_obscured_pixel_next_to_fog_stays_obscured():
    night = night_arrays(btd=[5.0, 3.9], ir108=[270.0, 250.0])
    labels = label_night(*night, np.full(2, 4.0), relaxed(0.5))
    assert labels[1, 4, 4] == OBSCURED


def test_neighbours_are_the_stored_scenes():
    night = night_arrays(btd=[5.0, 3.7], ir108=[270.0, 270.0])  # e.g. 21:00 and 03:00, 00:00 missing
    labels = label_night(*night, np.full(2, 4.0), relaxed(0.5))
    assert labels[1, 4, 4] == FOG


def test_hysteresis_also_looks_at_the_later_scene():
    night = night_arrays(btd=[3.3, 5.0], ir108=[270.0, 270.0])
    labels = label_night(*night, np.full(2, 4.0), relaxed(1.0))
    assert labels[0, 4, 4] == FOG


def test_rough_near_miss_next_to_fog_stays_not_fog():
    ir039, ir108, ir120, sza = night_arrays(btd=[5.0, 3.3], ir108=[270.0, 270.0])
    rows, cols = np.indices((3, 3))
    bumps = 1.8 * np.where((rows + cols) % 2 == 0, 1.0, -1.0)  # rough, but still a steady top
    for field in (ir039, ir108, ir120):
        field[1, 3:6, 3:6] += bumps
    labels = label_night(ir039, ir108, ir120, sza, np.full(2, 4.0), relaxed(1.0))
    assert labels[1, 4, 4] == NO_FOG


def test_btd_farther_below_the_threshold_than_the_relax_stays_not_fog():
    night = night_arrays(btd=[5.0, 2.5], ir108=[270.0, 270.0])
    labels = label_night(*night, np.full(2, 4.0), relaxed(1.0))
    assert labels[1, 4, 4] == NO_FOG


def test_untrusted_pixel_next_to_fog_stays_ignored():
    ir039, ir108, ir120, sza = night_arrays(btd=[5.0, 3.3], ir108=[270.0, 270.0])
    sza[1] = 80.0
    labels = label_night(ir039, ir108, ir120, sza, np.full(2, 4.0), relaxed(1.0))
    assert labels[1, 4, 4] == IGNORE
