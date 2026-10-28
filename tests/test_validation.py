"""Tests for the station truth rule, the matching and the case table."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from helpers import COLD_PIXEL, GRID_LAT0, GRID_LON0, TEST_PARAMS, make_station_dataset

from fogseg.cli import cmd_validate
from fogseg.config import load_config
from fogseg.label_store import label_dataset, label_store_path
from fogseg.validation import (
    CASE_COLUMNS,
    check_season_allowed,
    deduplicate,
    fls_truth,
    fog_truth,
    match_observations,
    nearest_pixels,
    night_of,
    read_observations,
    score_table,
    season_bounds,
    select_stations,
    station_cases,
)

STATIONS_CFG = {
    "max_distance_km": 5,
    "max_elevation_m": 1000,
    "truth_limit_m": 1000,
    "scan_delay_min": 12,
    "time_tolerance_min": 30,
}


def small_grid() -> tuple[np.ndarray, np.ndarray]:
    """A 3 x 3 grid with 0.1 degree spacing from 10 degE, 50 degN, north at the top."""
    lon, lat = np.meshgrid(10.0 + 0.1 * np.arange(3), 50.0 - 0.1 * np.arange(3))
    return lon, lat


def observation_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Observations of (station, source, time) with a fixed position and no fog."""
    return pd.DataFrame(
        {
            "source": [row[1] for row in rows],
            "station": [row[0] for row in rows],
            "time": pd.to_datetime([row[2] for row in rows]),
            "lat": 50.0,
            "lon": 10.0,
            "elevation_m": 100.0,
            "visibility_m": 9999.0,
            "ceiling_m": np.inf,
        }
    )


def test_fls_truth_edges():
    visibility = np.array([999, 1000, 5000, np.nan, 5000, np.nan])
    ceiling = np.array([np.inf, np.inf, 999, 200, np.nan, np.inf])
    truth = fls_truth(visibility, ceiling, limit_m=1000)
    assert truth.dtype == np.int8
    assert truth.tolist() == [1, 0, 1, 1, -1, -1]


def test_fog_truth():
    truth = fog_truth(np.array([999, 1000, np.nan]), limit_m=1000)
    assert truth.dtype == np.int8
    assert truth.tolist() == [1, 0, -1]


def test_night_of_morning_slot_belongs_to_evening():
    times = pd.Series(pd.to_datetime(["2023-11-21T03:00", "2023-11-20T21:00"]))
    nights = night_of(times)
    assert nights.dtype == np.dtype("datetime64[D]")
    assert nights.tolist() == [np.datetime64("2023-11-20"), np.datetime64("2023-11-20")]


def test_nearest_pixel_and_distance():
    lon, lat = small_grid()
    on_centre = nearest_pixels(lon, lat, np.array([10.1]), np.array([49.9]))
    assert on_centre[["iy", "ix"]].iloc[0].tolist() == [1, 1]
    assert on_centre["distance_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    east = nearest_pixels(lon, lat, np.array([10.14]), np.array([49.9]))
    assert east[["iy", "ix"]].iloc[0].tolist() == [1, 1]
    expected_km = 0.04 * 111.32 * np.cos(np.radians(49.9))
    assert east["distance_km"].iloc[0] == pytest.approx(expected_km, rel=1e-2)


def test_select_stations_drops_far_and_high():
    lon, lat = small_grid()
    stations = pd.DataFrame(
        {
            "station": ["NEAR", "FAR", "HIGH", "NOELEV"],
            "source": "metar",
            "lat": [49.9, 50.0, 49.9, 49.8],
            "lon": [10.1, 9.7, 10.1, 10.2],
            "elevation_m": [100.0, 100.0, 1500.0, np.nan],
        }
    )
    selected = select_stations(stations, lon, lat, max_distance_km=5, max_elevation_m=1000)
    assert selected["station"].tolist() == ["NEAR", "NOELEV"]
    assert selected[["iy", "ix"]].values.tolist() == [[1, 1], [2, 2]]


def test_match_takes_closest_observation_after_scan_delay():
    observations = observation_frame([("A", "metar", "2023-11-21T04:50"), ("A", "metar", "2023-11-21T05:20")])
    scene_times = np.array(["2023-11-21T05:00"], dtype="datetime64[ns]")
    matched = match_observations(observations, scene_times, scan_delay_min=12, tolerance_min=30)
    assert len(matched) == 1
    assert matched["time"].iloc[0] == pd.Timestamp("2023-11-21T05:20")
    assert matched["scene_time"].iloc[0] == pd.Timestamp("2023-11-21T05:00")


def test_match_outside_tolerance_gives_no_case():
    observations = observation_frame([("A", "metar", "2023-11-21T04:20")])
    scene_times = np.array(["2023-11-21T05:00"], dtype="datetime64[ns]")
    matched = match_observations(observations, scene_times, scan_delay_min=12, tolerance_min=30)
    assert matched.empty
    assert "scene_time" in matched.columns


def test_deduplicate_prefers_metar():
    cases = pd.DataFrame(
        {
            "scene_time": pd.Timestamp("2023-11-21T05:00"),
            "station": ["Z-SYNOP", "A-METAR", "OTHER"],
            "source": ["synop", "metar", "synop"],
            "iy": [1, 1, 2],
            "ix": [1, 1, 2],
        }
    )
    kept = deduplicate(cases)
    assert kept["station"].tolist() == ["A-METAR", "OTHER"]


def test_station_cases_reads_labels_and_flags_hidden(tmp_path):
    dataset_path = make_station_dataset(tmp_path / "data.zarr")
    labels_path = tmp_path / "labels_fixed.zarr"
    label_dataset(dataset_path, labels_path, "fixed", TEST_PARAMS)

    def pixel_station(name: str, iy: int, ix: int) -> tuple[str, float, float]:
        return name, GRID_LAT0 - 0.1 * iy, GRID_LON0 + 0.1 * ix

    stations = [pixel_station("FOG", 2, 2), pixel_station("CLEAR", 5, 1), pixel_station("COLD", *COLD_PIXEL)]
    observations = pd.DataFrame(
        {
            "source": "metar",
            "station": [name for name, _, _ in stations],
            "time": pd.Timestamp("2023-11-21T00:12"),
            "lat": [lat for _, lat, _ in stations],
            "lon": [lon for _, _, lon in stations],
            "elevation_m": 100.0,
            "visibility_m": [500.0, 9999.0, 9999.0],
            "ceiling_m": [np.inf, np.inf, 300.0],
        }
    )

    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(labels_path, consolidated=False) as labels,
    ):
        cases = station_cases(data, labels, observations, STATIONS_CFG, TEST_PARAMS).set_index("station")

    assert list(cases.columns) == [
        "scene_time",
        "night",
        "slot",
        "source",
        "lat",
        "lon",
        "iy",
        "ix",
        "visibility_m",
        "ceiling_m",
        "truth_fls",
        "truth_fog",
        "label",
        "hidden",
        "threshold_source",
    ]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "label"].tolist() == [1, 0, 0]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "hidden"].tolist() == [False, False, True]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "truth_fls"].tolist() == [1, 0, 1]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "truth_fog"].tolist() == [1, 0, 0]
    assert set(cases["slot"]) == {"00:00"}
    assert (cases["night"] == np.datetime64("2023-11-20")).all()
    assert set(cases["threshold_source"]) == {"fixed"}


def season_cfg() -> dict:
    return {
        "time": {
            "seasons": [
                {"name": "2023-24", "start": "2023-10-01", "end": "2024-02-29"},
                {"name": "2024-25", "start": "2024-10-01", "end": "2025-02-28"},
            ]
        },
        "stations": {"tune_season": "2023-24"},
    }


def test_check_season_refuses_test_season():
    with pytest.raises(PermissionError, match="2026-10-28-station-validation-design"):
        check_season_allowed(season_cfg(), "2024-25", final=False)
    check_season_allowed(season_cfg(), "2024-25", final=True)
    check_season_allowed(season_cfg(), "2023-24", final=False)


def test_season_bounds_run_from_noon_of_the_first_day_to_noon_after_the_last():
    start, end = season_bounds(season_cfg(), "2023-24")
    assert start == np.datetime64("2023-10-01T12:00")
    assert end == np.datetime64("2024-03-01T12:00")


def test_season_bounds_unknown_name():
    with pytest.raises(ValueError, match="2023-24"):
        season_bounds(season_cfg(), "1999-00")


def case(night: str, station: str, source: str, iy: int, truth: int, label: int, **overrides) -> dict:
    """One case of the score tests; `truth` is the FLS truth, the fog truth is 0 unless given."""
    return {
        "night": night,
        "station": station,
        "source": source,
        "iy": iy,
        "truth_fls": truth,
        "label": label,
        "slot": "00:00",
        "lat": 49.0,
        "truth_fog": 0,
        "hidden": False,
        "threshold_source": "fallback",
    } | overrides


def case_frame(rows: list[dict]) -> pd.DataFrame:
    """Cases in the layout of `station_cases`, with the columns the scores do not use filled in."""
    cases = pd.DataFrame(rows).assign(lon=10.0, ix=0, visibility_m=9999.0, ceiling_m=np.inf)
    cases["night"] = pd.to_datetime(cases["night"])
    cases["scene_time"] = cases["night"] + pd.Timedelta(hours=24)
    return cases[CASE_COLUMNS]


def score_cases() -> pd.DataFrame:
    first, second = "2023-11-20", "2023-11-21"
    evening = {"slot": "21:00", "threshold_source": "dynamic"}
    return case_frame(
        [
            case(first, "A", "metar", 1, 1, 1, truth_fog=1, lat=45.0, **evening),
            case(first, "B", "metar", 2, 0, 1, lat=45.0, **evening),
            case(first, "C", "synop", 3, 0, 0, **evening),
            case(first, "A-SYNOP", "synop", 1, 1, 1, truth_fog=1, lat=45.0, **evening),
            case(second, "D", "metar", 4, 1, 0),
            case(second, "E", "synop", 5, 1, 1, truth_fog=1),
            case(second, "F", "synop", 6, 1, 255),
            case(second, "G", "metar", 7, 1, 0, hidden=True),
            case(second, "H", "synop", 8, 0, 255),
        ]
    )


def table_of(cases: pd.DataFrame) -> pd.DataFrame:
    return score_table(cases, [43, 47, 51, 56], n_samples=50, seed=0).set_index(["view", "group"])


def test_score_table_views():
    table = table_of(score_cases())
    count_names = ["hits", "misses", "false_alarms", "correct_negatives"]

    main = table.loc[("main", "all")]
    assert main[count_names].tolist() == [2, 1, 1, 1]
    assert main["n_cases"] == 5
    assert main["pod"] == pytest.approx(2 / 3)
    assert main["far"] == pytest.approx(1 / 3)
    assert main["csi"] == pytest.approx(2 / 4)
    assert main["bias"] == pytest.approx(1.0)
    assert main["csi_low"] <= main["csi_high"]
    assert (main["n_fls_ignored"], main["n_no_fls_ignored"], main["n_fls_hidden"]) == (1, 1, 1)

    assert table.loc[("with_hidden", "all"), count_names].tolist() == [2, 2, 1, 1]
    assert table.loc[("source", "metar"), count_names].tolist() == [1, 1, 1, 0]
    assert table.loc[("source", "synop"), count_names].tolist() == [2, 0, 0, 1]
    assert table.loc[("threshold_source", "dynamic"), count_names].tolist() == [1, 0, 1, 1]
    assert table.loc[("threshold_source", "fallback"), count_names].tolist() == [1, 1, 0, 0]
    assert table.loc[("lat_band", "43-47"), count_names].tolist() == [1, 0, 1, 0]
    assert table.loc[("lat_band", "47-51"), count_names].tolist() == [1, 1, 0, 1]
    assert table.loc[("slot", "21:00"), count_names].tolist() == [1, 0, 1, 1]
    assert table.loc[("slot", "00:00"), count_names].tolist() == [1, 1, 0, 0]
    assert table.loc[("fog_only", "all"), count_names].tolist() == [2, 0, 1, 2]


def test_score_table_empty_group_is_nan():
    empty_band = table_of(score_cases()).loc[("lat_band", "51-56")]
    assert empty_band["n_cases"] == 0
    assert np.isnan(empty_band[["pod", "far", "csi", "csi_low", "csi_high", "bias"]].astype(float)).all()


def station_files(tmp_path: Path, sources: tuple[str, ...]) -> Path:
    data_dir = tmp_path / "stations"
    data_dir.mkdir()
    for source in sources:
        observation_frame([("06240", source, "2023-11-21T00:10")]).to_csv(
            data_dir / f"observations_{source}.csv", index=False
        )
    return data_dir


def test_read_observations_keeps_station_ids_as_text(tmp_path, caplog):
    data_dir = station_files(tmp_path, ("metar",))

    with caplog.at_level(logging.WARNING, logger="fogseg"):
        observations = read_observations(data_dir)

    assert observations["station"].tolist() == ["06240"]
    assert observations["time"].iloc[0] == pd.Timestamp("2023-11-21T00:10")
    assert "observations_synop.csv" in caplog.text


def test_read_observations_without_any_table_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg stations"):
        read_observations(station_files(tmp_path, ()))


def validate_config(tmp_path: Path, dataset_path: Path) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset_path)
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["mode"] = "fixed"
    cfg["stations"]["data_dir"] = str(tmp_path / "stations")
    cfg["stations"]["bootstrap_samples"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def validate_setup(tmp_path: Path) -> argparse.Namespace:
    """Dataset, fixed label store and a METAR table with a fog and a clear station; no SYNOP table."""
    dataset_path = make_station_dataset(tmp_path / "data.zarr")
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", TEST_PARAMS)
    stations = [("FOG", 2, 2, 500.0), ("CLEAR", 5, 1, 9999.0)]
    data_dir = station_files(tmp_path, ())
    pd.DataFrame(
        {
            "source": "metar",
            "station": [name for name, *_ in stations],
            "time": pd.Timestamp("2023-11-21T00:12"),
            "lat": [GRID_LAT0 - 0.1 * iy for _, iy, _, _ in stations],
            "lon": [GRID_LON0 + 0.1 * ix for _, _, ix, _ in stations],
            "elevation_m": 100.0,
            "visibility_m": [visibility for *_, visibility in stations],
            "ceiling_m": np.inf,
        }
    ).to_csv(data_dir / "observations_metar.csv", index=False)
    return argparse.Namespace(
        config=validate_config(tmp_path, dataset_path), labels=None, season="2023-24", final=False
    )


def test_validate_writes_cases_and_scores(tmp_path, caplog):
    args = validate_setup(tmp_path)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_validate(args)

    data_dir = tmp_path / "stations"
    cases = pd.read_csv(data_dir / "cases_fixed_2023-24.csv", dtype={"station": str})
    scores = pd.read_csv(data_dir / "scores_fixed_2023-24.csv").set_index(["view", "group"])
    assert sorted(cases["station"]) == ["CLEAR", "FOG"]
    main = scores.loc[("main", "all")]
    assert main[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [1, 0, 0, 1]
    assert "observations_synop.csv" in caplog.text
    assert "scores_fixed_2023-24.csv" in caplog.text


def test_validate_refuses_the_test_season_without_final(tmp_path):
    args = validate_setup(tmp_path)
    args.season = "2024-25"

    with pytest.raises(PermissionError):
        cmd_validate(args)
    assert not list((tmp_path / "stations").glob("cases_*"))


def test_validate_rejects_labels_of_other_scenes(tmp_path):
    args = validate_setup(tmp_path)
    shorter_path = tmp_path / "shorter.zarr"
    with xr.open_zarr(tmp_path / "data.zarr", consolidated=False) as data:
        data.isel(time=[0, 1]).load().to_zarr(shorter_path, mode="w", zarr_format=3, consolidated=False)
    args.config = validate_config(tmp_path, shorter_path)

    with pytest.raises(ValueError, match="fogseg label"):
        cmd_validate(args)
