import dataclasses

import numpy as np

from fogseg.config import load_config
from fogseg.labels import LabelParams, histogram_pixels, trusted_mask

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
