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
    BASE_FIELDS,
    LABEL_DATASET_CREATED,
    LABEL_NX,
    LABEL_NY,
    TEST_DYNAMIC,
    TEST_PARAMS,
    label_scenes,
    make_label_dataset,
)

import fogseg
from fogseg import label_store
from fogseg.build import variable_length_strings
from fogseg.cli import cmd_label
from fogseg.config import load_config
from fogseg.label_store import (
    check_mode,
    label_dataset,
    label_fractions,
    label_store_path,
    night_batches,
    stored_params,
    summarize_store,
)
from fogseg.labels import FOG, IGNORE, OBSCURED, LabelParams, label_night

# Small enough for the 80-pixel test scenes, so the dynamic threshold really finds the clear-ground peak
SMALL_SCENE_DYNAMIC = dataclasses.replace(TEST_DYNAMIC, min_pixels=10)
PATCH_PIXELS = 5 * 5
SCENE_PIXELS = LABEL_NY * LABEL_NX
FIRST_SLOT = np.datetime64("2023-11-20T21:00", "ns")
SLOT = np.timedelta64(3, "h")


def open_store(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def scene_times(stamps: list[str]) -> list[np.datetime64]:
    return [np.datetime64(stamp, "ns") for stamp in stamps]


def write_dataset(
    tmp_path: Path,
    n_nights: int = 1,
    slots: int = 3,
    scene_sza: dict[int, float] | None = None,
    patch_btd: dict[int, float] | None = None,
) -> Path:
    """Like `make_label_dataset`, with `slots` scenes at 21:00, 00:00, ... on each of `n_nights` days.

    `patch_btd` maps a scene index to the BTD of its fog patch in K instead of 5.0.
    """
    dataset = label_scenes(n_nights * slots, scene_sza, background_btd=1.8)
    night_starts = FIRST_SLOT + np.arange(n_nights) * np.timedelta64(1, "D")
    dataset["time"] = (night_starts[:, None] + np.arange(slots) * SLOT).ravel()
    for scene, btd in (patch_btd or {}).items():
        dataset["bt"][scene, 0, :5, :5] = 270.0 - btd

    path = tmp_path / "data.zarr"
    variable_length_strings(dataset).to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return path


def night_inputs(dataset: Path, night: int, slots: int = 3) -> tuple[np.ndarray, ...]:
    """IR_039, IR_108, IR_120 and sza of the scenes of one night of a `write_dataset` dataset."""
    with xr.open_zarr(dataset, consolidated=False) as data:
        scenes = data.isel(time=slice(night * slots, (night + 1) * slots))
        bt = scenes["bt"].sel(channel=["IR_039", "IR_108", "IR_120"]).values
        return bt[:, 0], bt[:, 1], bt[:, 2], scenes["sza"].values


def test_label_store_path_is_named_after_the_mode(tmp_path):
    assert label_store_path(tmp_path, "fixed") == tmp_path / "labels_fixed.zarr"
    assert label_store_path(tmp_path, "dynamic") == tmp_path / "labels_dynamic.zarr"


def test_store_has_spec_variables_and_aligns_with_the_dataset(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    assert label_dataset(data, out, "fixed", TEST_PARAMS) == 3
    labels = open_store(out)
    assert labels["label"].dtype == np.uint8
    assert set(np.unique(labels["label"])) <= {0, 1, 2, 255}
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


def test_label_fractions_report_obscured_separately():
    fog, ignored, obscured = label_fractions(np.array([1, 0, 2, 2, 255], dtype=np.uint8))
    assert (fog, ignored, obscured) == (0.5, 0.2, 0.4)


def test_label_fractions_exclude_obscured_pixels_from_the_decided_ones():
    fog, _, _ = label_fractions(np.array([1, 2, 2, 2], dtype=np.uint8))
    assert fog == 1.0


def test_store_has_obscured_fraction_and_lists_value_2_in_the_label_name(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    labels = open_store(out)

    assert labels["obscured_fraction"].dims == ("time",)
    assert labels["obscured_fraction"].dtype == np.float32
    assert "2 obscured" in labels["label"].attrs["long_name"]
    expected = (labels["label"] == OBSCURED).mean(dim=["y", "x"]).values
    np.testing.assert_allclose(labels["obscured_fraction"].values, expected)


def test_store_variables_have_the_documented_layout(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    labels = open_store(out)

    assert labels["label"].dims == ("time", "y", "x")
    assert labels["label"].encoding["chunks"] == (8, LABEL_NY, LABEL_NX)
    for name in ["btd_threshold", "fog_fraction", "ignore_fraction", "obscured_fraction"]:
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
    data = write_dataset(tmp_path, n_nights=3, slots=2, scene_sza={3: 80.0})
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


def test_night_batches_never_split_a_night():
    times = np.array(scene_times(["2023-10-01T21", "2023-10-02T00", "2023-10-02T03", "2023-10-03T21"]))
    assert night_batches(times, 2) == [(0, 3), (3, 4)]


def test_night_batches_fill_a_batch_with_whole_nights():
    times = np.array(scene_times(["2023-10-01T21", "2023-10-02T03", "2023-10-02T21", "2023-10-03T21"]))
    assert night_batches(times, 3) == [(0, 3), (3, 4)]
    assert night_batches(times, 4) == [(0, 4)]


def test_night_runs_from_noon_to_noon():
    times = np.array(scene_times(["2023-10-01T11", "2023-10-01T12"]))
    assert night_batches(times, 1) == [(0, 1), (1, 2)]


def test_store_with_temporal_relax_matches_label_night(tmp_path):
    dataset = write_dataset(tmp_path, n_nights=2, slots=3, patch_btd={1: 2.3})
    params = dataclasses.replace(TEST_PARAMS, temporal_relax=0.5)
    label_dataset(dataset, tmp_path / "labels.zarr", "fixed", params, batch=2)
    stored = open_store(tmp_path / "labels.zarr")["label"].values

    thresholds = np.full(3, params.btd_fixed)
    assert np.array_equal(stored[:3], label_night(*night_inputs(dataset, 0), thresholds, params))
    assert np.array_equal(stored[3:], label_night(*night_inputs(dataset, 1), thresholds, params))
    assert stored[1, 2, 2] == FOG  # the strict label of this pixel is IGNORE


def test_dynamic_store_matches_label_night(tmp_path):
    dataset = write_dataset(tmp_path, n_nights=1, slots=3)
    params = dataclasses.replace(TEST_PARAMS, temporal_relax=0.5)
    label_dataset(dataset, tmp_path / "labels.zarr", "dynamic", params, SMALL_SCENE_DYNAMIC)
    labels = open_store(tmp_path / "labels.zarr")

    expected = label_night(*night_inputs(dataset, 0), labels["btd_threshold"].values, params)
    assert np.array_equal(labels["label"].values, expected)


def test_emissivity_store_records_its_threshold(tmp_path):
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity", emissivity_max=0.9)
    label_dataset(write_dataset(tmp_path), tmp_path / "labels.zarr", "fixed", params)
    store = open_store(tmp_path / "labels.zarr")
    assert np.allclose(store["btd_threshold"].values, 0.9) and store.attrs["fog_index"] == "emissivity"


def test_dynamic_mode_refuses_the_emissivity_index():
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity", emissivity_max=0.9)
    with pytest.raises(ValueError, match="dynamic"):
        check_mode("dynamic", TEST_DYNAMIC, params)


def test_emissivity_index_needs_its_threshold():
    params = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    with pytest.raises(ValueError, match=r"labels\.emissivity_max"):
        check_mode("fixed", None, params)


def test_store_records_the_new_options_and_reads_them_back(tmp_path):
    params = dataclasses.replace(TEST_PARAMS, texture="blob", split_window="kma", temporal_relax=0.5)
    label_dataset(write_dataset(tmp_path), tmp_path / "labels.zarr", "fixed", params)
    attrs = open_store(tmp_path / "labels.zarr").attrs

    assert attrs["emissivity_max"] is None
    assert stored_params(attrs, TEST_DYNAMIC)[0] == params


def test_old_store_attributes_give_the_default_options():
    attrs = {"mode": "fixed", **{name: getattr(TEST_PARAMS, name) for name in BASE_FIELDS}}
    params, _ = stored_params(attrs, TEST_DYNAMIC)
    assert params == LabelParams(**{name: getattr(TEST_PARAMS, name) for name in BASE_FIELDS})


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
    data = write_dataset(tmp_path, n_nights=2, slots=2)
    out = tmp_path / "labels_fixed.zarr"
    label_dataset(data, out, "fixed", TEST_PARAMS)
    before = open_store(out).load()

    real_label_night = label_store.label_night
    calls = []

    def fail_on_second_night(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated failure")
        return real_label_night(*args, **kwargs)

    monkeypatch.setattr(label_store, "label_night", fail_on_second_night)
    with pytest.raises(RuntimeError, match="simulated failure"):
        label_dataset(data, out, "fixed", TEST_PARAMS, overwrite=True, batch=1)

    xr.testing.assert_identical(open_store(out).load(), before)
    assert not out.with_name("labels_fixed.zarr.part").exists()


def test_failure_on_a_first_write_leaves_nothing_behind(tmp_path, monkeypatch):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"

    def always_fail(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(label_store, "label_night", always_fail)
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
