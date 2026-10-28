"""Tests for choosing the check scenes and `fogseg select-scenes`."""

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
from helpers import CHANNELS, TEST_PARAMS, make_label_dataset

from fogseg.check_scenes import (
    MIN_DECIDED_FRACTION,
    night_of,
    scene_statistics,
    season_of,
    select_check_scenes,
)
from fogseg.cli import cmd_select_scenes
from fogseg.config import load_config
from fogseg.label_store import label_dataset, label_store_path

SCENES_PER_SEASON = 20
SLOT_OFFSETS_HOURS = [0, 3, 6, 8]  # 21:00, 00:00, 03:00 and 05:00 UTC of one night
CLI_SCENES = 120  # 15 nights of 8 scenes: enough nights for 15 check scenes
TRAINING_FALLBACK_ROWS = [4, 9, 15]
GROUP_SIZES = {"foggy": 3, "clear": 3, "high_cloud": 3, "twilight": 2, "fallback": 2, "spread": 2}


def make_stats(scenes_per_season: int = SCENES_PER_SEASON) -> pd.DataFrame:
    """Scene statistics of one scene a week from early October in both seasons; all fractions distinct."""
    training = pd.date_range("2023-10-02 21:00", periods=scenes_per_season, freq="7D")
    test = pd.date_range("2024-10-02 21:00", periods=scenes_per_season, freq="7D")
    times = training.append(test)
    n_scenes = len(times)

    rng = np.random.default_rng(0)
    sources = np.full(n_scenes, "dynamic", dtype=object)
    fallback_rows = [row for row in TRAINING_FALLBACK_ROWS if row < scenes_per_season]
    sources[fallback_rows] = "fallback"
    return pd.DataFrame(
        {
            "time": times,
            "season": [season_of(time) for time in times],
            "fog_fraction": rng.permutation(n_scenes) / n_scenes,
            "decided_fraction": 1.0,
            "high_cloud_fraction": rng.permutation(n_scenes) / n_scenes,
            "twilight_fraction": rng.permutation(n_scenes) / n_scenes,
            "threshold_source": sources,
        }
    )


def make_night_stats(n_nights: int = 20) -> pd.DataFrame:
    """Training-season statistics with the four slots of every night; some nights hold all the extremes."""
    evenings = pd.date_range("2023-11-01 21:00", periods=n_nights, freq="D")
    slots = [evening + pd.Timedelta(hours=hours) for evening in evenings for hours in SLOT_OFFSETS_HOURS]
    n_scenes = len(slots)

    rng = np.random.default_rng(1)
    fog = rng.permutation(n_scenes) / n_scenes
    high_cloud = rng.permutation(n_scenes) / n_scenes
    twilight = rng.permutation(n_scenes) / n_scenes
    night_number = np.repeat(np.arange(n_nights), len(SLOT_OFFSETS_HOURS))
    fog[night_number < 3] += 10
    high_cloud[(night_number >= 3) & (night_number < 6)] += 10
    twilight[(night_number >= 6) & (night_number < 8)] += 10
    return pd.DataFrame(
        {
            "time": slots,
            "season": [season_of(time) for time in slots],
            "fog_fraction": fog,
            "decided_fraction": 1.0,
            "high_cloud_fraction": high_cloud,
            "twilight_fraction": twilight,
            "threshold_source": "dynamic",
        }
    )


def test_night_of_groups_the_slots_of_one_night():
    slots = pd.DatetimeIndex(
        ["2024-02-08 21:00", "2024-02-09 00:00", "2024-02-09 03:00", "2024-02-09 05:00", "2024-02-09 21:00"]
    )

    nights = night_of(slots)

    expected = [pd.Timestamp("2024-02-08")] * 4 + [pd.Timestamp("2024-02-09")]
    assert list(nights) == expected


def test_season_runs_from_october_to_february():
    assert season_of(pd.Timestamp("2023-10-01 21:00")) == "2023-24"
    assert season_of(pd.Timestamp("2023-12-31 21:00")) == "2023-24"
    assert season_of(pd.Timestamp("2024-02-29 05:00")) == "2023-24"
    assert season_of(pd.Timestamp("2024-10-02 00:00")) == "2024-25"
    assert season_of(pd.Timestamp("2025-01-15 03:00")) == "2024-25"


def test_selection_uses_only_the_training_season():
    stats = make_stats()
    is_test_season = stats["season"] == "2024-25"
    stats.loc[is_test_season, "fog_fraction"] = np.linspace(0.99, 1.0, int(is_test_season.sum()))

    selected = select_check_scenes(stats)

    assert len(selected) == 15
    assert set(selected["season"]) == {"2023-24"}
    assert all(time < pd.Timestamp("2024-03-01") for time in selected["time"])


def test_selection_is_deterministic_and_has_no_duplicates():
    stats = make_stats()

    first = select_check_scenes(stats)
    second = select_check_scenes(stats)

    pd.testing.assert_frame_equal(first, second)
    assert first["time"].is_unique
    assert first["time"].is_monotonic_increasing
    assert len(first) == 15


def test_selection_contains_each_group():
    stats = make_stats()
    # Middling fractions, so no earlier group takes the fallback scenes
    fractions = ["fog_fraction", "high_cloud_fraction", "twilight_fraction"]
    stats.loc[TRAINING_FALLBACK_ROWS, fractions] = 0.5

    selected = select_check_scenes(stats)

    sizes = selected["group"].value_counts().to_dict()
    assert sizes == GROUP_SIZES
    fallback_times = list(selected.loc[selected["group"] == "fallback", "time"])
    assert fallback_times == list(stats.loc[TRAINING_FALLBACK_ROWS[:2], "time"])


def test_selection_groups_pick_the_extremes():
    stats = make_stats()
    training = stats[stats["season"] == "2023-24"]

    selected = select_check_scenes(stats)

    foggy = selected.loc[selected["group"] == "foggy", "time"]
    most_fog = training.nlargest(3, "fog_fraction")["time"]
    assert set(foggy) == set(most_fog)
    clear = selected.loc[selected["group"] == "clear", "time"]
    least_fog = training.nsmallest(3, "fog_fraction")["time"]
    assert set(clear) == set(least_fog)


def test_ties_go_to_the_earlier_scene():
    stats = make_stats()
    stats["fog_fraction"] = 0.5

    selected = select_check_scenes(stats)

    foggy = list(selected.loc[selected["group"] == "foggy", "time"])
    clear = list(selected.loc[selected["group"] == "clear", "time"])
    assert foggy == list(stats["time"][:3])
    assert clear == list(stats["time"][3:6])


def test_foggy_and_clear_skip_scenes_with_few_decided_pixels():
    stats = make_stats()
    training = stats[stats["season"] == "2023-24"]
    most_fog = training["fog_fraction"].idxmax()
    least_fog = training["fog_fraction"].idxmin()
    # Mostly twilight: the extreme fog fractions rest on a few pixels
    stats.loc[[most_fog, least_fog], "decided_fraction"] = MIN_DECIDED_FRACTION - 0.01

    selected = select_check_scenes(stats)

    in_fog_groups = selected.loc[selected["group"].isin(["foggy", "clear"]), "time"]
    assert stats.loc[most_fog, "time"] not in set(in_fog_groups)
    assert stats.loc[least_fog, "time"] not in set(in_fog_groups)
    assert (selected["group"] == "foggy").sum() == 3
    assert (selected["group"] == "clear").sum() == 3


def test_scene_with_exactly_the_minimum_decided_share_can_be_foggy():
    stats = make_stats()
    training = stats[stats["season"] == "2023-24"]
    most_fog = training["fog_fraction"].idxmax()
    stats.loc[most_fog, "decided_fraction"] = MIN_DECIDED_FRACTION

    selected = select_check_scenes(stats)

    assert stats.loc[most_fog, "time"] in set(selected.loc[selected["group"] == "foggy", "time"])


def test_selection_takes_at_most_one_scene_per_night():
    stats = make_night_stats()

    selected = select_check_scenes(stats)

    nights = night_of(selected["time"])
    assert len(selected) == 15
    assert nights.is_unique
    foggy_nights = night_of(selected.loc[selected["group"] == "foggy", "time"])
    assert set(foggy_nights) == set(night_of(stats["time"][:12]))
    spread_hours = selected.loc[selected["group"] == "spread", "time"].dt.hour
    assert list(spread_hours) == [21] * len(spread_hours)


def test_selection_with_few_training_scenes_returns_them_all():
    stats = make_stats(scenes_per_season=5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected = select_check_scenes(stats)

    training_times = stats.loc[stats["season"] == "2023-24", "time"]
    assert list(selected["time"]) == list(training_times)
    assert selected["group"].notna().all()


def with_known_fractions(data: xr.Dataset) -> xr.Dataset:
    """The 4 synthetic scenes changed so their high-cloud and twilight fractions are known exactly."""
    data = data.load()
    bt = data["bt"].values.copy()
    sza = data["sza"].values.copy()
    ir108 = bt[:, CHANNELS.index("IR_108")]

    ir108[0, :2, :] = 230.0  # 20 of 80 pixels high cloud
    sza[1, :, :5] = 98.0  # half the scene twilight
    ir108[1, :, 5:7] = 230.0  # 16 of the 40 night pixels high cloud ...
    ir108[1, 0, 9] = np.nan  # ... and one night pixel without IR_108
    sza[2] = 40.0  # day: no night pixel
    sza[3] = np.nan
    ir108[3] = np.nan

    data["bt"] = (data["bt"].dims, bt)
    data["sza"] = (data["sza"].dims, sza)
    return data


def test_scene_statistics_reads_store_and_computes_fractions(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=4)
    store_path = label_store_path(tmp_path / "labels", "fixed")
    # A night limit other than 95 shows that the limit the labels were made with is used
    params = dataclasses.replace(TEST_PARAMS, night_min_sza=100.0)
    label_dataset(dataset_path, store_path, "fixed", params)

    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats = scene_statistics(with_known_fractions(data), labels)
        expected_fog = labels["fog_fraction"].values
        expected_ignore = labels["ignore_fraction"].values.astype(float)

    assert list(stats.columns) == [
        "time",
        "season",
        "fog_fraction",
        "decided_fraction",
        "high_cloud_fraction",
        "twilight_fraction",
        "threshold_source",
    ]
    assert list(stats.index) == [0, 1, 2, 3]
    np.testing.assert_allclose(stats["decided_fraction"], 1.0 - expected_ignore)
    assert stats["time"].is_monotonic_increasing
    assert list(stats["season"]) == ["2023-24"] * 4
    assert list(stats["threshold_source"]) == ["fixed"] * 4
    np.testing.assert_array_equal(stats["fog_fraction"], expected_fog)
    assert list(stats["high_cloud_fraction"]) == pytest.approx([0.25, 16 / 39, 0.0, 0.0])
    assert list(stats["twilight_fraction"]) == pytest.approx([0.0, 0.5, 1.0, 0.0])


def test_scene_statistics_rejects_a_label_store_of_other_scenes(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=4)
    store_path = label_store_path(tmp_path / "labels", "fixed")
    label_dataset(make_label_dataset(tmp_path / "other.zarr", n_scenes=3), store_path, "fixed", TEST_PARAMS)

    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
        pytest.raises(ValueError, match="fogseg label"),
    ):
        scene_statistics(data, labels)


def write_config(tmp_path: Path, dataset: Path, mode: str) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset)
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["mode"] = mode
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def test_cli_writes_the_check_scenes_next_to_the_config(tmp_path, caplog):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=CLI_SCENES)
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", TEST_PARAMS)
    config = write_config(tmp_path, dataset_path, mode="fixed")

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_select_scenes(argparse.Namespace(config=config, labels=None, overwrite=False))

    out_path = tmp_path / "check_scenes.yaml"
    written = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert list(written) == ["labels", "scenes"]
    assert written["labels"] == "fixed"
    assert len(written["scenes"]) == 15
    dataset_times = pd.date_range("2023-11-20 21:00", periods=CLI_SCENES, freq="3h")
    for scene in written["scenes"]:
        assert list(scene) == ["time", "group"]
        assert pd.Timestamp(scene["time"]) in dataset_times
    assert written["scenes"][0]["time"] == "2023-11-20T21:00"
    assert str(out_path) in caplog.text


def test_cli_keeps_existing_check_scenes_without_overwrite(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=CLI_SCENES)
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", TEST_PARAMS)
    config = write_config(tmp_path, dataset_path, mode="fixed")
    out_path = tmp_path / "check_scenes.yaml"
    out_path.write_text("reviewed: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        cmd_select_scenes(argparse.Namespace(config=config, labels=None, overwrite=False))
    assert out_path.read_text(encoding="utf-8") == "reviewed: true\n"

    cmd_select_scenes(argparse.Namespace(config=config, labels=None, overwrite=True))
    assert len(yaml.safe_load(out_path.read_text(encoding="utf-8"))["scenes"]) == 15


def test_scene_statistics_rejects_a_label_store_of_another_dataset_build(tmp_path):
    dataset_path = make_label_dataset(tmp_path / "data.zarr", n_scenes=4)
    store_path = label_store_path(tmp_path / "labels", "fixed")
    label_dataset(dataset_path, store_path, "fixed", TEST_PARAMS)

    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
        pytest.raises(ValueError, match="fogseg label"),
    ):
        data.attrs["created"] = "2026-02-01T00:00:00+00:00"
        scene_statistics(data, labels)
