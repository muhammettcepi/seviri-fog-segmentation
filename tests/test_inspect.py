"""Tests for the check figures and `fogseg inspect`."""

import argparse
import dataclasses
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from helpers import TEST_DYNAMIC, TEST_PARAMS, make_label_dataset

from fogseg.cli import cmd_inspect
from fogseg.config import load_config
from fogseg.inspect import find_scene, histogram_title, inspect_figure, night_microphysics_rgb
from fogseg.label_store import label_dataset, label_store_path, stored_params
from fogseg.labels import LabelParams

WANTED = pd.Timestamp("2023-11-21 00:00")  # the second scene of the synthetic dataset


def make_stores(tmp_path: Path, mode: str = "fixed") -> tuple[Path, Path]:
    dataset_path = make_label_dataset(tmp_path / "data.zarr")
    store_path = label_store_path(tmp_path / "labels", mode)
    dynamic = TEST_DYNAMIC if mode == "dynamic" else None
    label_dataset(dataset_path, store_path, mode, TEST_PARAMS, dynamic)
    return dataset_path, store_path


def open_zarr(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def test_rgb_maps_recipe_ranges_to_zero_and_one():
    rgb = night_microphysics_rgb(np.array([[293.0]]) - 10, np.array([[293.0]]), np.array([[295.0]]))
    assert rgb[0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_rgb_maps_the_lower_ends_to_zero():
    rgb = night_microphysics_rgb(np.array([[243.0]]), np.array([[243.0]]), np.array([[239.0]]))
    assert rgb[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_rgb_scales_linearly_between_the_ends():
    rgb = night_microphysics_rgb(np.array([[263.0]]), np.array([[268.0]]), np.array([[267.0]]))
    assert rgb[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_rgb_clips_to_zero_and_one():
    rgb = night_microphysics_rgb(np.array([[300.0]]), np.array([[200.0]]), np.array([[250.0]]))
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_rgb_makes_missing_pixels_black_without_warning():
    nan = np.full((2, 2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rgb = night_microphysics_rgb(nan, nan, nan)
    assert rgb.shape == (2, 2, 3)
    assert not rgb.any()


def test_find_scene_returns_the_index_of_an_exact_time():
    times = pd.DatetimeIndex(["2023-11-20 21:00", "2023-11-21 00:00", "2023-11-21 03:00"])
    assert find_scene(times, pd.Timestamp("2023-11-21 00:00")) == 1


def test_missing_time_lists_nearest_available_times():
    times = pd.DatetimeIndex(["2023-11-20 21:00", "2023-11-21 00:00", "2023-11-21 03:00"])
    with pytest.raises(ValueError, match="2023-11-21 00:00"):
        find_scene(times, pd.Timestamp("2023-11-22 00:00"))


def test_missing_time_lists_exactly_three_nearest_times():
    times = pd.DatetimeIndex(pd.date_range("2023-11-20 00:00", periods=6, freq="3h"))
    with pytest.raises(ValueError) as error:
        find_scene(times, pd.Timestamp("2023-11-20 07:00"))
    message = str(error.value)
    for listed in ["2023-11-20 06:00", "2023-11-20 09:00", "2023-11-20 03:00"]:
        assert listed in message
    assert "2023-11-20 12:00" not in message


def test_histogram_title_names_threshold_and_source():
    title = histogram_title(2.5, "fallback")
    assert "2.50" in title
    assert "fallback" in title


def test_inspect_writes_a_png(tmp_path):
    dataset_path, store_path = make_stores(tmp_path)
    out_dir = tmp_path / "figures"
    with open_zarr(dataset_path) as data, open_zarr(store_path) as labels:
        png = inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, out_dir)
    assert png == out_dir / "inspect_fixed_20231121T0000.png"
    assert png.stat().st_size > 0
    assert png.read_bytes().startswith(b"\x89PNG")


def test_inspect_names_the_file_after_the_label_mode(tmp_path):
    dataset_path, store_path = make_stores(tmp_path, mode="dynamic")
    with open_zarr(dataset_path) as data, open_zarr(store_path) as labels:
        png = inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, tmp_path / "figures")
    assert png.name == "inspect_dynamic_20231121T0000.png"


def test_inspect_handles_a_scene_with_day_and_night_pixels_without_warnings(tmp_path):
    dataset_path, store_path = make_stores(tmp_path)
    with open_zarr(dataset_path) as data, open_zarr(store_path) as labels:
        sza = data["sza"].load()
        sza[1, :, :5] = 90.0
        data["sza"] = sza
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            png = inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, tmp_path / "figures")
    assert png.exists()


def test_inspect_handles_a_scene_without_any_night_pixel(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", scene_sza={1: 40.0})
    store_path = label_store_path(tmp_path / "labels", "fixed")
    label_dataset(dataset_path, store_path, "fixed", TEST_PARAMS)
    with open_zarr(dataset_path) as data, open_zarr(store_path) as labels, warnings.catch_warnings():
        warnings.simplefilter("error")
        png = inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, tmp_path / "figures")
    assert png.exists()


def test_inspect_rejects_a_time_that_is_not_in_the_dataset(tmp_path):
    dataset_path, store_path = make_stores(tmp_path)
    missing = pd.Timestamp("2023-11-22 00:00")
    with (
        open_zarr(dataset_path) as data,
        open_zarr(store_path) as labels,
        pytest.raises(ValueError, match="2023-11-21 00:00"),
    ):
        inspect_figure(data, labels, missing, TEST_PARAMS, TEST_DYNAMIC, tmp_path)


def test_inspect_rejects_labels_of_another_dataset_build(tmp_path):
    dataset_path, store_path = make_stores(tmp_path)
    with open_zarr(dataset_path) as data, open_zarr(store_path) as labels:
        data.attrs["created"] = "2026-02-01T00:00:00+00:00"  # rebuilt: same times, other pixels
        with pytest.raises(ValueError, match="fogseg label"):
            inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_inspect_rejects_labels_of_other_scenes(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=4)
    store_path = label_store_path(tmp_path / "labels", "fixed")
    label_dataset(make_label_dataset(tmp_path / "other.zarr", n_scenes=3), store_path, "fixed", TEST_PARAMS)
    with (
        open_zarr(dataset_path) as data,
        open_zarr(store_path) as labels,
        pytest.raises(ValueError, match="other scenes"),
    ):
        inspect_figure(data, labels, WANTED, TEST_PARAMS, TEST_DYNAMIC, tmp_path / "figures")


def write_config(tmp_path: Path, dataset: Path, mode: str) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset)
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["figures_dir"] = str(tmp_path / "figures")
    cfg["labels"]["mode"] = mode
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_cli_uses_the_configured_mode_and_logs_the_written_path(tmp_path, caplog):
    dataset_path, _ = make_stores(tmp_path, mode="dynamic")
    config = write_config(tmp_path, dataset_path, mode="dynamic")

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_inspect(argparse.Namespace(config=config, time="2023-11-21T00:00", labels=None))

    png = tmp_path / "figures" / "inspect_dynamic_20231121T0000.png"
    assert png.exists()
    assert str(png) in caplog.text


def test_cli_labels_option_overrides_the_config(tmp_path):
    dataset_path, _ = make_stores(tmp_path, mode="fixed")
    config = write_config(tmp_path, dataset_path, mode="dynamic")

    cmd_inspect(argparse.Namespace(config=config, time="2023-11-21T00:00", labels="fixed"))

    assert (tmp_path / "figures" / "inspect_fixed_20231121T0000.png").exists()


def test_cli_time_with_a_utc_zone_finds_the_scene(tmp_path):
    dataset_path, _ = make_stores(tmp_path, mode="fixed")
    config = write_config(tmp_path, dataset_path, mode="fixed")

    cmd_inspect(argparse.Namespace(config=config, time="2023-11-21T00:00Z", labels=None))

    assert (tmp_path / "figures" / "inspect_fixed_20231121T0000.png").exists()


def test_cli_time_with_another_zone_is_converted_to_utc(tmp_path):
    dataset_path, _ = make_stores(tmp_path, mode="fixed")
    config = write_config(tmp_path, dataset_path, mode="fixed")

    cmd_inspect(argparse.Namespace(config=config, time="2023-11-21T01:00+01:00", labels=None))

    assert (tmp_path / "figures" / "inspect_fixed_20231121T0000.png").exists()


def test_stored_params_of_a_dynamic_store_come_from_the_store_not_the_config(tmp_path):
    stricter = dataclasses.replace(TEST_PARAMS, warm_min_bt108=271.0)
    store_path = label_store_path(tmp_path / "labels", "dynamic")
    label_dataset(make_label_dataset(tmp_path / "data.zarr"), store_path, "dynamic", stricter, TEST_DYNAMIC)
    config_dynamic = dataclasses.replace(TEST_DYNAMIC, offset=0.5)

    with open_zarr(store_path) as labels:
        params, dynamic = stored_params(labels.attrs, config_dynamic)

    assert params == stricter
    assert dynamic == TEST_DYNAMIC


def test_stored_params_of_a_fixed_store_use_the_config_dynamic_params(tmp_path):
    stricter = dataclasses.replace(TEST_PARAMS, warm_min_bt108=271.0)
    store_path = label_store_path(tmp_path / "labels", "fixed")
    label_dataset(make_label_dataset(tmp_path / "data.zarr"), store_path, "fixed", stricter)

    with open_zarr(store_path) as labels:
        params, dynamic = stored_params(labels.attrs, TEST_DYNAMIC)

    assert params == stricter
    assert dynamic == TEST_DYNAMIC


def test_cli_figure_is_built_from_the_store_params_after_a_config_edit(tmp_path, monkeypatch):
    stricter = dataclasses.replace(TEST_PARAMS, warm_min_bt108=271.0)
    dataset_path = make_label_dataset(tmp_path / "data.zarr")
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", stricter)
    config = write_config(tmp_path, dataset_path, mode="fixed")
    used = []

    def record_params(data, labels, wanted, params: LabelParams, dynamic, out_dir):
        used.append(params)
        return out_dir / "recorded.png"

    monkeypatch.setattr("fogseg.inspect.inspect_figure", record_params)
    cmd_inspect(argparse.Namespace(config=config, time="2023-11-21T00:00", labels=None))

    assert used[0].warm_min_bt108 == 271.0
    assert load_config(config)["labels"]["warm_min_bt108"] != 271.0
