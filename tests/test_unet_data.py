import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import TEST_PARAMS
from test_experiment import experiment_config, tiny_season
from test_features import teacher_dataset

from fogseg.features import CHANNELS
from fogseg.labels import OBSCURED, label_scene
from fogseg.unet_data import (
    INPUT_CHANNELS,
    Normalization,
    fit_normalization,
    model_inputs,
    obscured_mask,
    point_targets,
    raw_inputs,
    read_inputs,
    read_targets,
    scene_night_folds,
    targets,
    tune_season_scenes,
)
from fogseg.validation import TRUTH_NO, TRUTH_YES

NIGHT_MIN_SZA = TEST_PARAMS.night_min_sza


def one_pixel(bt_values: dict[str, float], sza: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    bt = np.full((2, len(CHANNELS), 1, 1), 270.0, dtype="float32")
    for channel, value in bt_values.items():
        bt[:, CHANNELS.index(channel)] = value
    return bt, np.full((2, 1, 1), sza, dtype="float32")


def test_input_channels_by_hand():
    bt, sza = one_pixel({"IR_039": 265.0, "IR_108": 270.0, "IR_120": 268.5, "WV_062": 240.0})
    raw, trusted = raw_inputs(bt, sza, np.array([[432.0]]), NIGHT_MIN_SZA)
    named = dict(zip(INPUT_CHANNELS, raw.transpose(1, 0, 2, 3), strict=False))
    assert len(INPUT_CHANNELS) == 14
    assert raw.shape == (2, 13, 1, 1)
    assert raw.dtype == np.float32
    assert trusted.all()
    assert named["IR_108-IR_039"][:, 0, 0].tolist() == [5.0, 5.0]
    assert named["IR_108-IR_120"][:, 0, 0].tolist() == [1.5, 1.5]
    assert named["IR_108-WV_062"][:, 0, 0].tolist() == [30.0, 30.0]
    assert named["elevation"][:, 0, 0].tolist() == [432.0, 432.0]
    assert named["sza"][:, 0, 0].tolist() == [120.0, 120.0]


def test_untrusted_and_nan_pixels_are_zero_and_flagged():
    bt, sza = one_pixel({})
    bt = np.repeat(bt, 3, axis=3)
    bt[0, CHANNELS.index("IR_108"), 0, 1] = np.nan
    sza = np.repeat(sza, 3, axis=2)
    sza[0, 0, 2] = 90.0
    raw, trusted = raw_inputs(bt, sza, np.zeros((1, 3)), NIGHT_MIN_SZA)
    inputs = model_inputs(raw, trusted, fit_normalization(raw, trusted))
    assert inputs.shape == (2, 14, 1, 3)
    assert np.isfinite(inputs).all()
    assert (inputs[0, :, 0, 1:] == 0).all()
    assert inputs[0, -1, 0, :].tolist() == [1.0, 0.0, 0.0]
    assert inputs[1, -1, 0, :].tolist() == [1.0, 1.0, 1.0]


def test_normalization_ignores_untrusted_pixels():
    raw = np.random.default_rng(0).normal(size=(2, 13, 3, 4)).astype("float32")
    raw[:, 0] = 7.0
    trusted = np.ones((2, 3, 4), dtype=bool)
    trusted[:, 0] = False
    raw[:, 1][~trusted] = 1e6
    normalization = fit_normalization(raw, trusted)
    kept = raw[:, 1][trusted]
    assert normalization.mean[1] == pytest.approx(kept.mean(), rel=1e-5)
    assert normalization.std[1] == pytest.approx(kept.std(), rel=1e-5)
    assert normalization.std[0] == 1.0
    again = Normalization.from_dict(normalization.to_dict())
    np.testing.assert_array_equal(again.mean, normalization.mean)
    np.testing.assert_array_equal(again.std, normalization.std)


def test_soft_and_hard_targets():
    label = np.array([[0, 1, 2, 255]], dtype="uint8")
    probability = np.array([[25, 200, 255, 125]], dtype="uint8")
    soft, soft_weight = targets(label, probability, "soft")
    hard, hard_weight = targets(label, probability, "hard")
    assert soft.dtype == hard.dtype == np.float32
    assert soft_weight.tolist() == [[1, 1, 0, 1]]
    assert soft[soft_weight == 1].tolist() == pytest.approx([0.1, 0.8, 0.5])
    assert hard_weight.tolist() == [[1, 1, 0, 0]]
    assert hard[hard_weight == 1].tolist() == [0.0, 1.0]


def test_soft_target_needs_fog_probability():
    with pytest.raises(ValueError, match="fog_probability"):
        targets(np.zeros((1, 2), dtype="uint8"), None, "soft")


def test_obscured_is_the_rules_obscured_class(tmp_path):
    data = teacher_dataset(tiny_season(tmp_path, ir120=267.0)[0])
    scenes = np.arange(data.sizes["time"])
    raw, trusted, obscured = read_inputs(data, scenes, np.zeros(data["sza"].shape[1:]), TEST_PARAMS)
    bt = data["bt"].sel(channel=["IR_039", "IR_108", "IR_120"]).values
    sza = data["sza"].values
    expected = np.stack([label_scene(*bt[i], sza[i], 4.0, TEST_PARAMS) == OBSCURED for i in scenes])
    assert raw.shape == (4, 13, 8, 10)
    assert expected.any()
    np.testing.assert_array_equal(obscured, expected)
    np.testing.assert_array_equal(
        obscured, obscured_mask(data["bt"].sel(channel=CHANNELS).values, trusted, TEST_PARAMS)
    )


def test_scene_folds_follow_the_split_file(tmp_path):
    experiment_config(tmp_path)
    splits = pd.read_csv(tmp_path / "stations" / "splits_2023-24.csv")
    times = tiny_season(tmp_path, n_nights=3)[0]["time"].values
    start = np.datetime64("2023-11-20")
    assert scene_night_folds(times, splits, start, 1).tolist() == [1, 1, 2, 2, 5, 5]
    with pytest.raises(ValueError, match="2023-11-22"):
        scene_night_folds(times, splits[splits["key"] != "2023-11-22"], start, 1)


def test_tune_season_scenes_skip_the_test_season(tmp_path):
    cfg = experiment_config(tmp_path)
    times = np.array(["2023-11-20T21:00", "2023-11-21T00:00", "2024-11-20T21:00"], dtype="datetime64[ns]")
    assert tune_season_scenes(times, cfg).tolist() == [0, 1]


def test_point_targets():
    times = np.array(["2023-11-20T21:00", "2023-11-21T00:00"], dtype="datetime64[ns]")
    other_scene = np.datetime64("2023-12-01T00:00", "ns")
    cases = pd.DataFrame(
        {
            "scene_time": [times[1], times[0], times[0], times[1], other_scene],
            "iy": [1, 2, 3, 4, 5],
            "ix": [6, 7, 8, 9, 10],
            "truth_fls": [TRUTH_YES, TRUTH_NO, -1, TRUTH_YES, TRUTH_YES],
        }
    )
    points = point_targets(cases, times)
    assert points.columns.tolist() == ["scene", "iy", "ix", "truth"]
    assert points["scene"].tolist() == [1, 0, 1]
    assert points["iy"].tolist() == [1, 2, 4]
    assert points["truth"].tolist() == [1.0, 0.0, 1.0]
    assert points["truth"].dtype == np.float32


def test_read_targets_from_a_label_store():
    label = np.array([[[0, 1, 255]], [[1, 2, 0]]], dtype="uint8")
    probability = np.array([[[0, 250, 255]], [[250, 255, 0]]], dtype="uint8")
    dims = ("time", "y", "x")
    labels = xr.Dataset({"label": (dims, label), "fog_probability": (dims, probability)})
    target, weight = read_targets(labels, np.array([1]), "soft")
    assert target.shape == weight.shape == (1, 1, 3)
    assert weight.tolist() == [[[1, 0, 1]]]
    hard, hard_weight = read_targets(labels[["label"]], np.array([0]), "hard")
    assert hard_weight.tolist() == [[[1, 1, 0]]]
    assert hard[0, 0, :2].tolist() == [0.0, 1.0]
