"""Tests for writing the label stores."""

import argparse
import dataclasses
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from helpers import (
    LABEL_DATASET_CREATED,
    LABEL_NX,
    LABEL_NY,
    TEST_DYNAMIC,
    TEST_PARAMS,
    make_label_dataset,
)

import fogseg
from fogseg import label_store
from fogseg.cli import cmd_label
from fogseg.config import load_config
from fogseg.label_store import label_dataset, label_store_path, summarize_store
from fogseg.labels import FOG, IGNORE

# Small enough for the 80-pixel test scenes, so the dynamic threshold really finds the clear-ground peak
SMALL_SCENE_DYNAMIC = dataclasses.replace(TEST_DYNAMIC, min_pixels=10)
PATCH_PIXELS = 5 * 5
SCENE_PIXELS = LABEL_NY * LABEL_NX


def open_store(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def test_label_store_path_is_named_after_the_mode(tmp_path):
    assert label_store_path(tmp_path, "fixed") == tmp_path / "labels_fixed.zarr"
    assert label_store_path(tmp_path, "dynamic") == tmp_path / "labels_dynamic.zarr"


def test_store_has_spec_variables_and_aligns_with_the_dataset(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    assert label_dataset(data, out, "fixed", TEST_PARAMS) == 3
    labels = open_store(out)
    assert labels["label"].dtype == np.uint8
    assert set(np.unique(labels["label"])) <= {0, 1, 255}
    assert list(labels["threshold_source"].values) == ["fixed"] * 3
    merged = xr.merge([open_store(data), labels])
    assert merged.sizes["time"] == 3
    assert labels.attrs["btd_fixed"] == TEST_PARAMS.btd_fixed


def test_threshold_source_is_stored_as_a_variable_length_string(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)

    metadata = json.loads((out / "threshold_source" / "zarr.json").read_text())
    assert metadata["data_type"] == "string"


def test_store_variables_have_the_documented_layout(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    labels = open_store(out)

    assert labels["label"].dims == ("time", "y", "x")
    assert labels["label"].encoding["chunks"] == (8, LABEL_NY, LABEL_NX)
    for name in ["btd_threshold", "fog_fraction", "ignore_fraction"]:
        assert labels[name].dims == ("time",)
        assert labels[name].dtype == np.float32
    assert labels["threshold_source"].dims == ("time",)
    np.testing.assert_array_equal(labels["time"].values, open_store(data)["time"].values)
    np.testing.assert_array_equal(labels["y"].values, open_store(data)["y"].values)
    np.testing.assert_array_equal(labels["x"].values, open_store(data)["x"].values)


def test_fractions_count_fog_among_decided_pixels_and_ignore_among_all(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    labels = open_store(out)

    assert (labels["label"] == FOG).sum(dim=["y", "x"]).values.tolist() == [PATCH_PIXELS] * 3
    np.testing.assert_allclose(labels["fog_fraction"].values, PATCH_PIXELS / SCENE_PIXELS)
    np.testing.assert_allclose(labels["ignore_fraction"].values, 0.0)
    np.testing.assert_allclose(labels["btd_threshold"].values, TEST_PARAMS.btd_fixed)


def test_ignore_fraction_is_taken_over_all_pixels(tmp_path):
    # the fog patch is 25 of 80 pixels; a 2.0 K band around BTD 5.0 K makes all of it uncertain
    wide_band = dataclasses.replace(TEST_PARAMS, btd_fixed=4.0, uncertainty_band=2.0)
    data = make_label_dataset(tmp_path / "data.zarr", n_scenes=1)
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", wide_band)
    labels = open_store(out)

    np.testing.assert_allclose(labels["ignore_fraction"].values, PATCH_PIXELS / SCENE_PIXELS)
    np.testing.assert_allclose(labels["fog_fraction"].values, 0.0)


def test_dynamic_mode_records_threshold_source(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_dynamic.zarr"
    label_dataset(data, out, "dynamic", TEST_PARAMS, dynamic=TEST_DYNAMIC)
    sources = set(open_store(out)["threshold_source"].values)
    assert sources <= {"dynamic", "fallback"}


def test_dynamic_mode_puts_the_threshold_an_offset_above_the_clear_ground_peak(tmp_path):
    clear_ground_btd = 3.0
    data = make_label_dataset(tmp_path / "data.zarr", background_btd=clear_ground_btd)
    out = tmp_path / "labels_dynamic.zarr"
    label_dataset(data, out, "dynamic", TEST_PARAMS, dynamic=SMALL_SCENE_DYNAMIC)
    labels = open_store(out)

    assert list(labels["threshold_source"].values) == ["dynamic"] * 3
    thresholds = labels["btd_threshold"].values
    one_bin = SMALL_SCENE_DYNAMIC.bin_width
    np.testing.assert_allclose(thresholds, clear_ground_btd + SMALL_SCENE_DYNAMIC.offset, atol=one_bin)
    assert not np.allclose(thresholds, TEST_PARAMS.btd_fixed)
    np.testing.assert_allclose(labels["fog_fraction"].values, PATCH_PIXELS / SCENE_PIXELS)


def test_fallback_scene_uses_the_fixed_threshold(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", n_scenes=1)
    out = tmp_path / "labels_dynamic.zarr"
    label_dataset(data, out, "dynamic", TEST_PARAMS, dynamic=TEST_DYNAMIC)
    labels = open_store(out)

    assert list(labels["threshold_source"].values) == ["fallback"]
    np.testing.assert_allclose(labels["btd_threshold"].values, TEST_PARAMS.btd_fixed)


def test_scene_without_night_gets_nan_fog_fraction(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", scene_sza={1: 80.0})
    out = tmp_path / "labels_fixed.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        label_dataset(data, out, "fixed", TEST_PARAMS)
    labels = open_store(out)

    assert np.all(labels["label"].isel(time=1).values == IGNORE)
    assert np.isnan(labels["fog_fraction"].values[1])
    assert labels["ignore_fraction"].values[1] == 1.0
    assert not np.isnan(labels["fog_fraction"].values[[0, 2]]).any()


def test_batches_give_the_same_store_as_one_pass(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", n_scenes=5, scene_sza={3: 80.0})
    one_pass = tmp_path / "one_pass.zarr"
    batched = tmp_path / "batched.zarr"
    label_dataset(data, one_pass, "fixed", TEST_PARAMS, batch=32)
    label_dataset(data, batched, "fixed", TEST_PARAMS, batch=2)

    xr.testing.assert_identical(open_store(one_pass).drop_attrs(), open_store(batched).drop_attrs())


def test_dynamic_batches_keep_every_source_label(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", n_scenes=3)
    out = tmp_path / "labels_dynamic.zarr"
    label_dataset(data, out, "dynamic", TEST_PARAMS, dynamic=TEST_DYNAMIC, batch=1)
    assert list(open_store(out)["threshold_source"].values) == ["fallback"] * 3


def test_fixed_store_attrs_describe_the_run(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    attrs = open_store(out).attrs

    assert attrs["mode"] == "fixed"
    for field in dataclasses.fields(TEST_PARAMS):
        assert attrs[field.name] == getattr(TEST_PARAMS, field.name)
    assert not [key for key in attrs if key.startswith("dynamic_")]
    assert attrs["fogseg_version"] == fogseg.__version__
    assert attrs["source_dataset"] == str(data)
    assert attrs["source_created"] == LABEL_DATASET_CREATED
    assert attrs["created"].endswith("+00:00")


def test_dynamic_store_attrs_include_the_dynamic_params(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_dynamic.zarr"
    label_dataset(data, out, "dynamic", TEST_PARAMS, dynamic=TEST_DYNAMIC)
    attrs = open_store(out).attrs

    assert attrs["mode"] == "dynamic"
    assert attrs["dynamic_bin_width"] == TEST_DYNAMIC.bin_width
    assert attrs["dynamic_min_pixels"] == TEST_DYNAMIC.min_pixels
    assert attrs["dynamic_bin_range"] == list(TEST_DYNAMIC.bin_range)
    assert attrs["dynamic_clamp"] == list(TEST_DYNAMIC.clamp)
    assert attrs["dynamic_clear_window"] == list(TEST_DYNAMIC.clear_window)
    assert attrs["dynamic_offset"] == TEST_DYNAMIC.offset
    for field in dataclasses.fields(TEST_DYNAMIC):
        assert f"dynamic_{field.name}" in attrs
    for removed in ["cloud_window", "min_hump_share", "max_valley_ratio"]:
        assert f"dynamic_{removed}" not in attrs


def test_missing_dataset_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.zarr"):
        label_dataset(tmp_path / "nowhere.zarr", tmp_path / "labels.zarr", "fixed", TEST_PARAMS)


def test_missing_sza_is_reported(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    xr.open_zarr(data, consolidated=False).drop_vars("sza").to_zarr(
        tmp_path / "no_sza.zarr", zarr_format=3, consolidated=False
    )
    with pytest.raises(ValueError, match="sza"):
        label_dataset(tmp_path / "no_sza.zarr", tmp_path / "labels.zarr", "fixed", TEST_PARAMS)
    assert not (tmp_path / "labels.zarr").exists()


def test_missing_channel_is_reported(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    without_ir120 = xr.open_zarr(data, consolidated=False).sel(channel=["IR_039", "IR_108"])
    without_ir120.to_zarr(tmp_path / "two_channels.zarr", zarr_format=3, consolidated=False)
    with pytest.raises(ValueError, match="IR_120"):
        label_dataset(tmp_path / "two_channels.zarr", tmp_path / "labels.zarr", "fixed", TEST_PARAMS)


def test_missing_bt_is_reported(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    xr.open_zarr(data, consolidated=False).drop_vars("bt").to_zarr(
        tmp_path / "no_bt.zarr", zarr_format=3, consolidated=False
    )
    with pytest.raises(ValueError, match="bt"):
        label_dataset(tmp_path / "no_bt.zarr", tmp_path / "labels.zarr", "fixed", TEST_PARAMS)


def test_unknown_mode_is_rejected(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    with pytest.raises(ValueError, match="mode.*sometimes"):
        label_dataset(data, tmp_path / "labels.zarr", "sometimes", TEST_PARAMS)


def test_dynamic_mode_requires_dynamic_params(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    with pytest.raises(ValueError, match="dynamic"):
        label_dataset(data, tmp_path / "labels_dynamic.zarr", "dynamic", TEST_PARAMS)


def test_refuses_to_overwrite_without_the_flag(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    with pytest.raises(FileExistsError, match="labels_fixed.zarr"):
        label_dataset(data, out, "fixed", TEST_PARAMS)


def test_overwrite_replaces_the_store(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    stricter = dataclasses.replace(TEST_PARAMS, btd_fixed=4.0, uncertainty_band=0.1)
    label_dataset(data, out, "fixed", stricter, overwrite=True)

    assert open_store(out).attrs["btd_fixed"] == 4.0
    assert not out.with_name("labels_fixed.zarr.part").exists()


def test_failure_while_overwriting_keeps_the_old_store(tmp_path, monkeypatch):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    before = open_store(out).load()

    real_label_scene = label_store.label_scene
    calls = []

    def fail_on_second_scene(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated failure")
        return real_label_scene(*args, **kwargs)

    monkeypatch.setattr(label_store, "label_scene", fail_on_second_scene)
    with pytest.raises(RuntimeError, match="simulated failure"):
        label_dataset(data, out, "fixed", TEST_PARAMS, overwrite=True, batch=1)

    xr.testing.assert_identical(open_store(out).load(), before)
    assert not out.with_name("labels_fixed.zarr.part").exists()


def test_failure_on_a_first_write_leaves_nothing_behind(tmp_path, monkeypatch):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"

    def always_fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(label_store, "label_scene", always_fail)
    with pytest.raises(RuntimeError):
        label_dataset(data, out, "fixed", TEST_PARAMS)
    assert not out.exists()
    assert not out.with_name("labels_fixed.zarr.part").exists()


def test_summary_counts_sources_and_averages_fog_fraction_ignoring_nan(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", scene_sza={1: 80.0})
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        source_counts, mean_fog_fraction = summarize_store(out)
    assert source_counts == {"fixed": 3}
    assert mean_fog_fraction == pytest.approx(PATCH_PIXELS / SCENE_PIXELS)


def test_summary_of_a_store_without_any_night_is_nan(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr", n_scenes=1, scene_sza={0: 80.0})
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, mean_fog_fraction = summarize_store(out)
    assert np.isnan(mean_fog_fraction)


def write_config(tmp_path: Path, dataset: Path, mode: str) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset)
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["mode"] = mode
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_cli_uses_the_configured_mode_by_default(tmp_path, caplog):
    data = make_label_dataset(tmp_path / "data.zarr")
    config = write_config(tmp_path, data, mode="dynamic")

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_label(argparse.Namespace(config=config, mode=None, overwrite=False))

    store = tmp_path / "labels" / "labels_dynamic.zarr"
    assert open_store(store).attrs["mode"] == "dynamic"
    assert "3 scenes" in caplog.text
    assert "fallback" in caplog.text


def test_cli_mode_option_overrides_the_config(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    config = write_config(tmp_path, data, mode="dynamic")

    cmd_label(argparse.Namespace(config=config, mode="fixed", overwrite=False))

    assert open_store(tmp_path / "labels" / "labels_fixed.zarr").attrs["mode"] == "fixed"
    assert not (tmp_path / "labels" / "labels_dynamic.zarr").exists()
