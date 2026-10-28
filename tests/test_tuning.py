"""Tests for the threshold grid, its scores and the decision to change a value."""

import argparse
import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from helpers import (
    GRID_LAT0,
    GRID_LON0,
    PLANTED_CLEAR_PIXEL,
    PLANTED_FOG_PIXEL,
    PLANTED_PATCH_BTD,
    PLANTED_PEAK,
    PLANTED_STRIP_BTD,
    PLANTED_STRIP_PIXEL,
    TEST_DYNAMIC,
    TEST_PARAMS,
    make_planted_dataset,
)

from fogseg.cli import cmd_tune
from fogseg.config import load_config
from fogseg.label_store import label_dataset, label_store_path
from fogseg.tuning import Decision, decide, grid_labels, grid_values, scene_thresholds

# 80 pixels only: the clear background alone is 47. The clamp is wide so no offset of the grid is cut.
PLANTED_DYNAMIC = dataclasses.replace(TEST_DYNAMIC, min_pixels=20, clamp=(2.0, 6.0))
PLANTED_PARAMS = dataclasses.replace(TEST_PARAMS, btd_fixed=5.0)
OFFSETS = grid_values([0.5, 4.0, 0.25])
SCENE_TIMES = pd.to_datetime(["2023-11-20T21:00", "2023-11-21T00:00", "2023-11-21T03:00"])
STATION_PIXELS = {
    "FOG": (PLANTED_FOG_PIXEL, 1),
    "STRIP": (PLANTED_STRIP_PIXEL, 0),
    "CLEAR": (PLANTED_CLEAR_PIXEL, 0),
}


def test_grid_values_includes_stop():
    assert grid_values([0.5, 1.0, 0.25]).tolist() == [0.5, 0.75, 1.0]


def test_grid_values_are_rounded_so_they_compare_equal_to_decimals():
    assert grid_values([0.1, 0.4, 0.1]).tolist() == [0.1, 0.2, 0.3, 0.4]


def test_scene_threshold_clamps_and_falls_back():
    clamp = (3.5, 6.0)

    assert scene_thresholds(2.125, 1.5, 5.0, clamp) == pytest.approx(3.625)
    assert scene_thresholds(1.0, 1.5, 5.0, clamp) == 3.5
    assert scene_thresholds(5.0, 3.0, 5.0, clamp) == 6.0
    assert scene_thresholds(float("nan"), 1.5, 7.0, clamp) == 7.0


def planted_cases(stations=tuple(STATION_PIXELS)) -> pd.DataFrame:
    """Each of `stations` at every scene, all on one night."""
    rows = [
        {
            "scene_time": time,
            "night": np.datetime64("2023-11-20"),
            "station": station,
            "iy": pixel[0],
            "ix": pixel[1],
            "truth_fls": truth,
        }
        for time in SCENE_TIMES
        for station, (pixel, truth) in STATION_PIXELS.items()
        if station in stations
    ]
    return pd.DataFrame(rows)


def planted_grid(
    tmp_path, parameter, values, mode, dynamic=PLANTED_DYNAMIC, stations=tuple(STATION_PIXELS)
) -> pd.DataFrame:
    dataset_path = make_planted_dataset(tmp_path / "planted.zarr")
    with xr.open_zarr(dataset_path, consolidated=False) as data:
        return grid_labels(data, planted_cases(stations), PLANTED_PARAMS, dynamic, parameter, values, mode)


def labels_at(grid: pd.DataFrame, value: float) -> list[int]:
    return grid[np.isclose(grid["value"], value)]["label"].unique().tolist()


def test_grid_labels_repeat_the_cases_per_value(tmp_path):
    grid = planted_grid(tmp_path, "offset", OFFSETS, "dynamic")

    assert list(grid.columns) == ["value", "night", "label", "truth_fls"]
    assert len(grid) == len(OFFSETS) * len(planted_cases())
    assert (grid.groupby("value").size() == len(planted_cases())).all()


def test_grid_labels_move_with_the_offset(tmp_path):
    grid = planted_grid(tmp_path, "offset", OFFSETS, "dynamic")
    patch = grid[grid["truth_fls"] == 1]

    assert patch.groupby("value")["label"].first().loc[[0.5, 2.0, 3.0, 4.0]].tolist() == [1, 1, 255, 0]


def test_grid_finds_planted_threshold(tmp_path):
    grid = planted_grid(tmp_path, "offset", OFFSETS, "dynamic")

    table, decision = decide(grid, "offset", current=4.0, n_samples=50, seed=0)

    # The patch is fog (label 1) for a threshold up to patch BTD - band, the strip is no fog (label 0)
    # from strip BTD + band on. Offsets between the two bounds get both right.
    band = PLANTED_PARAMS.uncertainty_band
    lowest = PLANTED_STRIP_BTD + band - PLANTED_PEAK
    highest = PLANTED_PATCH_BTD - band - PLANTED_PEAK
    assert lowest <= decision.best <= highest
    assert decision.csi_best == 1.0
    assert decision.csi_current == 0.0
    assert decision.changed
    assert table.set_index("value").loc[decision.best, "csi"] == 1.0


def test_grid_in_fixed_mode_uses_the_value_as_threshold(tmp_path):
    values = grid_values([4.0, 5.0, 0.5])
    strip = planted_grid(tmp_path, "btd_fixed", values, "fixed", stations=["STRIP"])
    patch = planted_grid(tmp_path, "btd_fixed", values, "fixed", stations=["FOG"])

    # 3.6 K is within the band of 4.0 K, so the strip is ignored there; 4.5 K labels it 0 and the patch 1
    assert labels_at(strip, 4.0) == [255]
    assert labels_at(strip, 4.5) == [0]
    assert labels_at(patch, 4.5) == [1]
    assert labels_at(patch, 5.0) == [255]


def test_fallback_grid_in_dynamic_mode_equals_fixed_mode_where_there_is_no_peak(tmp_path):
    values = grid_values([4.0, 6.0, 0.5])
    no_peak = dataclasses.replace(PLANTED_DYNAMIC, min_pixels=1000)

    dynamic_grid = planted_grid(tmp_path, "btd_fixed", values, "dynamic", no_peak)
    fixed_grid = planted_grid(tmp_path, "btd_fixed", values, "fixed")

    pd.testing.assert_frame_equal(dynamic_grid, fixed_grid)


@pytest.mark.parametrize(
    ("parameter", "mode"), [("offset", "fixed"), ("clamp", "dynamic"), ("offset", "other")]
)
def test_grid_labels_reject_a_parameter_and_mode_that_do_not_go_together(tmp_path, parameter, mode):
    with pytest.raises(ValueError, match=parameter):
        planted_grid(tmp_path, parameter, OFFSETS, mode)


def test_grid_labels_at_the_current_offset_equal_the_label_store(tmp_path):
    dataset_path = make_planted_dataset(tmp_path / "planted.zarr")
    store = tmp_path / "labels_dynamic.zarr"
    label_dataset(dataset_path, store, "dynamic", PLANTED_PARAMS, PLANTED_DYNAMIC)
    cases = planted_cases()

    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store, consolidated=False) as labels,
    ):
        grid = grid_labels(
            data,
            cases,
            PLANTED_PARAMS,
            PLANTED_DYNAMIC,
            "offset",
            np.array([PLANTED_DYNAMIC.offset]),
            "dynamic",
        )
        scene = pd.Index(labels["time"].values).get_indexer(cases["scene_time"])
        stored = labels["label"].values[scene, cases["iy"], cases["ix"]]

    assert grid["label"].tolist() == stored.tolist()


def grid_of_nights(n_nights: int, labels: dict[float, int]) -> pd.DataFrame:
    """Per value, one FLS case per night with the given label."""
    nights = pd.date_range("2023-11-01", periods=n_nights).to_numpy()
    return pd.concat(
        [
            pd.DataFrame({"value": value, "night": nights, "label": label, "truth_fls": 1})
            for value, label in labels.items()
        ],
        ignore_index=True,
    )


def test_decide_keeps_current_without_evidence():
    grid = grid_of_nights(20, {1.0: 1, 1.5: 1})

    table, decision = decide(grid, "offset", current=1.5, n_samples=100, seed=0)

    assert table["csi"].tolist() == [1.0, 1.0]
    assert decision.best == 1.5
    assert not decision.changed


def test_decide_changes_with_clear_evidence():
    grid = grid_of_nights(20, {1.0: 1, 1.5: 0})

    table, decision = decide(grid, "offset", current=1.5, n_samples=100, seed=0)

    assert decision.best == 1.0
    assert (decision.csi_best, decision.csi_current) == (1.0, 0.0)
    assert decision.difference_low > 0
    assert decision.changed
    assert table.set_index("value").loc[1.5, ["hits", "misses"]].tolist() == [0, 20]


def test_decide_does_not_change_when_the_difference_interval_reaches_zero():
    grid = grid_of_nights(20, {1.0: 1, 1.5: 1})
    two_first_nights = grid["night"] < grid["night"].min() + pd.Timedelta(days=2)
    grid.loc[(grid["value"] == 1.5) & two_first_nights, "label"] = 0

    _, decision = decide(grid, "offset", current=1.5, n_samples=200, seed=0)

    assert decision.best == 1.0
    assert decision.csi_best > decision.csi_current
    assert decision.difference_low <= 0
    assert not decision.changed


def test_decide_table_columns_and_ignored_cases():
    grid = grid_of_nights(4, {1.0: 255, 1.5: 1})

    table, _ = decide(grid, "offset", current=1.5, n_samples=20, seed=0)

    assert list(table.columns) == [
        "value",
        "n_cases",
        "hits",
        "misses",
        "false_alarms",
        "correct_negatives",
        "pod",
        "far",
        "csi",
        "csi_low",
        "csi_high",
        "bias",
        "n_ignored",
    ]
    assert table["n_cases"].tolist() == [0, 4]
    assert table["n_ignored"].tolist() == [4, 0]
    assert np.isnan(table.loc[0, "csi"])


def test_decide_breaks_ties_towards_the_current_value():
    grid = grid_of_nights(5, {1.0: 1, 1.5: 1, 2.0: 1, 2.5: 1})

    _, decision = decide(grid, "offset", current=2.0, n_samples=20, seed=0)

    assert decision.best == 2.0


def test_decide_rejects_a_current_value_outside_the_grid():
    grid = grid_of_nights(5, {1.0: 1, 1.5: 1})

    with pytest.raises(ValueError, match="1.25"):
        decide(grid, "offset", current=1.25, n_samples=20, seed=0)


def test_decision_changed_needs_a_different_value_and_a_positive_lower_bound():
    decision = Decision("offset", 1.5, 2.0, 0.5, 0.6, 0.01, 0.2)

    assert decision.changed
    assert not dataclasses.replace(decision, difference_low=0.0).changed
    assert not dataclasses.replace(decision, best=1.5).changed
    assert not dataclasses.replace(decision, difference_low=float("nan")).changed


def tune_config(tmp_path, dataset_path, offset: float) -> argparse.Namespace:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset_path)
    cfg["data"]["log_dir"] = str(tmp_path / "logs")
    cfg["labels"] |= dataclasses.asdict(PLANTED_PARAMS)
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["dynamic"] = {
        **{
            name: list(value) if isinstance(value, tuple) else value
            for name, value in dataclasses.asdict(PLANTED_DYNAMIC).items()
        },
        "offset": offset,
    }
    cfg["stations"] |= {
        "data_dir": str(tmp_path / "stations"),
        "offset_grid": [0.5, 4.0, 0.25],
        "btd_fixed_grid": [4.0, 6.0, 0.5],
        "bootstrap_samples": 20,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path)


def write_observations(tmp_path, visibilities: dict[str, float]) -> None:
    rows = [
        {
            "source": "metar",
            "station": station,
            "time": scene_time + pd.Timedelta(minutes=12),
            "lat": GRID_LAT0 - 0.1 * STATION_PIXELS[station][0][0],
            "lon": GRID_LON0 + 0.1 * STATION_PIXELS[station][0][1],
            "elevation_m": 100.0,
            "visibility_m": visibility,
            "ceiling_m": np.inf,
        }
        for scene_time in SCENE_TIMES
        for station, visibility in visibilities.items()
    ]
    data_dir = tmp_path / "stations"
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "observations_metar.csv", index=False)


def tune_setup(tmp_path, offset: float = 4.0) -> argparse.Namespace:
    dataset_path = make_planted_dataset(tmp_path / "data.zarr")
    params = PLANTED_PARAMS
    dynamic = dataclasses.replace(PLANTED_DYNAMIC, offset=offset)
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "dynamic"), "dynamic", params, dynamic)
    write_observations(tmp_path, {"FOG": 500.0, "STRIP": 9999.0, "CLEAR": 9999.0})
    return tune_config(tmp_path, dataset_path, offset)


def test_tune_writes_three_tables_and_logs_each_decision(tmp_path, caplog):
    args = tune_setup(tmp_path)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_tune(args)

    data_dir = tmp_path / "stations"
    offsets = pd.read_csv(data_dir / "tuning_offset.csv")
    fallbacks = pd.read_csv(data_dir / "tuning_btd_fixed.csv")
    fixed_mode = pd.read_csv(data_dir / "tuning_btd_fixed_fixed_mode.csv")
    assert offsets["value"].tolist() == OFFSETS.tolist()
    assert fallbacks["value"].tolist() == [4.0, 4.5, 5.0, 5.5, 6.0]
    assert fixed_mode["value"].tolist() == fallbacks["value"].tolist()
    assert offsets.loc[offsets["value"] == 1.5, "n_ignored"].tolist() == [3]
    decisions = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Tune ")]
    assert [message.split(":")[0] for message in decisions] == [
        "Tune offset",
        "Tune btd_fixed",
        "Tune btd_fixed (fixed mode)",
    ]


def test_tune_changes_the_offset_where_the_current_one_misses_the_fog(tmp_path, caplog):
    args = tune_setup(tmp_path, offset=4.0)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_tune(args)

    offset_line = next(
        record.getMessage() for record in caplog.records if "Tune offset" in record.getMessage()
    )
    assert "current 4, best 2.5" in offset_line
    assert offset_line.endswith("changed")


def test_tune_warns_and_stops_when_no_case_matches(tmp_path, caplog):
    args = tune_setup(tmp_path)
    data_dir = tmp_path / "stations"
    pd.read_csv(data_dir / "observations_metar.csv").assign(lat=60.0).to_csv(
        data_dir / "observations_metar.csv", index=False
    )

    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cmd_tune(args)

    assert "2023-24" in caplog.text
    assert not list(data_dir.glob("tuning_*.csv"))
