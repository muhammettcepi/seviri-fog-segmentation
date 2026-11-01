"""Tests for the station truth rule, the matching and the case table."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from helpers import (
    COLD_PIXEL,
    GRID_LAT0,
    GRID_LON0,
    LABEL_NX,
    LABEL_NY,
    TEST_PARAMS,
    make_station_dataset,
)

from fogseg.cli import cmd_validate
from fogseg.config import load_config
from fogseg.label_store import label_dataset, label_store_path
from fogseg.labels import OBSCURED
from fogseg.parallax import apparent_position
from fogseg.splits import make_splits, splits_path
from fogseg.validation import (
    CASE_COLUMNS,
    View,
    check_season_allowed,
    deduplicate,
    fls_truth,
    fog_truth,
    match_observations,
    nearest_pixels,
    night_of,
    on_grid_stations,
    read_observations,
    score_row,
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
    "parallax_cloud_top_m": 500.0,
    "lat_bands": [43, 47, 51, 56],
    "elevation_diff_bins": [-100.0, 100.0],
    "n_folds": 5,
}
COUNT_NAMES = ["hits", "misses", "false_alarms", "correct_negatives"]


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


def test_station_without_elevation_is_shifted_for_the_cloud_top_alone():
    km_in_degrees = 180 / np.pi / 6371
    row_lat = 50.0 + 0.5 * km_in_degrees * np.arange(-5, 6)
    lon, lat = np.meshgrid([10.0, 10.1], row_lat)
    data = xr.Dataset(coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)})
    observations = observation_frame([("NOELEV", "metar", "2023-11-21T00:00")]).assign(elevation_m=np.nan)

    _, apparent_lat = apparent_position(np.array([10.0]), np.array([50.0]), np.array([500.0]))
    expected_row = int(np.argmin(np.abs(row_lat - apparent_lat[0])))
    true_row = int(np.argmin(np.abs(row_lat - 50.0)))

    selected = on_grid_stations(data, observations, STATIONS_CFG)

    assert expected_row > true_row
    assert selected[["iy", "ix"]].values.tolist() == [[expected_row, 0]]
    assert selected["lat"].tolist() == [50.0]


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


def pixel_station(name: str, iy: int, ix: int) -> tuple[str, float, float]:
    return name, GRID_LAT0 - 0.1 * iy, GRID_LON0 + 0.1 * ix


STATION_ELEVATION_M = 100.0


def station_observations() -> pd.DataFrame:
    """METARs written before the AUTO, ceiling amount and shallow fog columns existed, at three pixels."""
    stations = [pixel_station("FOG", 2, 2), pixel_station("CLEAR", 5, 1), pixel_station("COLD", *COLD_PIXEL)]
    return pd.DataFrame(
        {
            "source": "metar",
            "station": [name for name, _, _ in stations],
            "time": pd.Timestamp("2023-11-21T00:12"),
            "lat": [lat for _, lat, _ in stations],
            "lon": [lon for _, _, lon in stations],
            "elevation_m": STATION_ELEVATION_M,
            "visibility_m": [500.0, 9999.0, 9999.0],
            "ceiling_m": [np.inf, np.inf, 300.0],
        }
    )


def cases_of(tmp_path: Path, observations: pd.DataFrame, dem: xr.Dataset | None = None) -> pd.DataFrame:
    """`station_cases` of the station dataset and its fixed labels, indexed by station."""
    dataset_path = make_station_dataset(tmp_path / "data.zarr")
    labels_path = tmp_path / "labels_fixed.zarr"
    label_dataset(dataset_path, labels_path, "fixed", TEST_PARAMS)
    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(labels_path, consolidated=False) as labels,
    ):
        cases = station_cases(data, labels, observations, STATIONS_CFG, TEST_PARAMS, dem=dem)
    return cases.set_index("station")


def test_station_cases_reads_labels_and_flags_hidden(tmp_path):
    cases = cases_of(tmp_path, station_observations())

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
        "auto",
        "ceiling_amount",
        "shallow_fog",
        "elevation_diff_m",
    ]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "label"].tolist() == [1, 0, OBSCURED]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "hidden"].tolist() == [False, False, True]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "truth_fls"].tolist() == [1, 0, 1]
    assert cases.loc[["FOG", "CLEAR", "COLD"], "truth_fog"].tolist() == [1, 0, 0]
    assert set(cases["slot"]) == {"00:00"}
    assert (cases["night"] == np.datetime64("2023-11-20")).all()
    assert set(cases["threshold_source"]) == {"fixed"}


def test_old_observation_table_gives_empty_metar_columns(tmp_path):
    cases = cases_of(tmp_path, station_observations())

    assert cases["ceiling_amount"].eq("").all()
    assert cases["auto"].isna().all()
    assert cases["shallow_fog"].isna().all()
    assert cases["elevation_diff_m"].isna().all()


def test_station_cases_take_the_metar_columns_from_the_observations(tmp_path):
    observations = station_observations().assign(
        auto=pd.array([True, False, pd.NA], dtype="boolean"),
        ceiling_amount=["", "", "BKN"],
        shallow_fog=pd.array([False, True, pd.NA], dtype="boolean"),
    )

    cases = cases_of(tmp_path, observations).loc[["FOG", "CLEAR", "COLD"]]

    assert cases["auto"].tolist() == [True, False, pd.NA]
    assert cases["ceiling_amount"].tolist() == ["", "", "BKN"]
    assert cases["shallow_fog"].tolist() == [False, True, pd.NA]


def test_elevation_diff_uses_the_dem_at_the_case_pixel(tmp_path):
    observations = station_observations().iloc[[0]]
    elevation = np.full((LABEL_NY, LABEL_NX), 1000.0, dtype="float32")
    elevation[2, 2] = 120.0
    dem = xr.Dataset({"elevation_mean": (("y", "x"), elevation)})

    cases = cases_of(tmp_path, observations, dem)

    assert cases[["iy", "ix"]].values.tolist() == [[2, 2]]
    assert cases["elevation_diff_m"].tolist() == [STATION_ELEVATION_M - 120.0]


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
        "ceiling_m": np.inf,
        "auto": pd.NA,
        "ceiling_amount": "",
        "shallow_fog": pd.NA,
        "elevation_diff_m": np.nan,
    } | overrides


def case_frame(rows: list[dict]) -> pd.DataFrame:
    """Cases in the layout of `station_cases`, with the columns the scores do not use filled in."""
    cases = pd.DataFrame(rows).assign(lon=10.0, ix=0, visibility_m=9999.0)
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
    return score_table(cases, STATIONS_CFG, n_samples=50, seed=0).set_index(["view", "group"])


def test_full_count_columns_count_ignored_fls_as_misses():
    cases = case_frame(
        [
            case("2023-11-20", "A", "metar", 1, 1, 1),
            case("2023-11-20", "B", "metar", 2, 1, 255),
            case("2023-11-20", "C", "metar", 3, 1, 0),
            case("2023-11-20", "D", "metar", 4, 0, 1),
        ]
    )

    row = score_row(View("main", "all", cases), 50, 0)

    assert (row["pod"], row["pod_full"]) == (pytest.approx(1 / 2), pytest.approx(1 / 3))
    assert row["far_full"] == pytest.approx(1 / 2)
    assert row["csi_full"] == pytest.approx(1 / 4)
    assert row["bias_full"] == pytest.approx(2 / 3)
    assert row["pod_full_low"] <= row["pod_full"] <= row["pod_full_high"]
    assert row["far_full_low"] <= row["far_full"] <= row["far_full_high"]
    assert np.isfinite(row["hss_full"])


def test_full_count_skips_undefined_truth_and_counts_obscured_as_no():
    cases = case_frame(
        [
            case("2023-11-20", "A", "metar", 1, 1, OBSCURED),
            case("2023-11-20", "B", "metar", 2, -1, 1),
            case("2023-11-20", "C", "metar", 3, 1, 1),
        ]
    )

    row = score_row(View("main", "all", cases), 50, 0)

    assert (row["pod_full"], row["far_full"]) == (pytest.approx(1 / 2), pytest.approx(0.0))


def strata_cases() -> pd.DataFrame:
    night = "2023-11-20"
    return case_frame(
        [
            case(night, "AUTO", "metar", 1, 1, 1, auto=True),
            case(night, "MANNED", "metar", 2, 1, 0, auto=False),
            case(night, "SYNOP", "synop", 3, 1, 1),
            case(night, "LOW-BKN", "metar", 4, 1, 1, ceiling_m=200.0, ceiling_amount="BKN"),
            case(night, "LOW-OVC", "metar", 5, 1, 0, ceiling_m=300.0, ceiling_amount="OVC"),
            case(night, "HIGH-OVC", "metar", 6, 0, 0, ceiling_m=3000.0, ceiling_amount="OVC"),
            case(night, "VV", "metar", 7, 1, 1, ceiling_m=30.0, ceiling_amount="VV"),
            case(night, "SHALLOW", "metar", 8, 1, 0, shallow_fog=True),
            case(night, "BELOW", "metar", 9, 1, 1, elevation_diff_m=-150.0),
            case(night, "EDGE", "metar", 10, 0, 0, elevation_diff_m=-100.0),
            case(night, "ABOVE", "metar", 11, 1, 0, elevation_diff_m=100.0),
        ]
    )


def test_score_table_station_strata():
    table = table_of(strata_cases())

    assert table.loc[("auto", "true"), COUNT_NAMES].tolist() == [1, 0, 0, 0]
    assert table.loc[("auto", "false"), COUNT_NAMES].tolist() == [0, 1, 0, 0]
    assert table.loc[("ceiling_amount", "BKN"), COUNT_NAMES].tolist() == [1, 0, 0, 0]
    assert table.loc[("ceiling_amount", "OVC"), COUNT_NAMES].tolist() == [0, 1, 0, 0]
    assert table.loc[("ceiling_amount", "VV"), COUNT_NAMES].tolist() == [1, 0, 0, 0]
    assert table.loc[("shallow_fog", "true"), COUNT_NAMES].tolist() == [0, 1, 0, 0]
    assert table.loc[("elevation_diff", "<-100"), COUNT_NAMES].tolist() == [1, 0, 0, 0]
    assert table.loc[("elevation_diff", "-100-100"), COUNT_NAMES].tolist() == [0, 0, 0, 1]
    assert table.loc[("elevation_diff", ">=100"), COUNT_NAMES].tolist() == [0, 1, 0, 0]
    assert "split" not in table.index.get_level_values("view")


def test_split_view_groups():
    night = "2023-11-20"
    cases_with_folds = case_frame(
        [
            case(night, "SEL", "metar", 1, 1, 1),
            case(night, "NIGHT", "metar", 2, 1, 0),
            case(night, "STATION", "metar", 3, 0, 1),
            case(night, "BOTH", "metar", 4, 0, 0),
        ]
    ).assign(night_fold=[1, 5, 2, 5], station_fold=[3, 4, 5, 5])

    table = score_table(cases_with_folds, STATIONS_CFG, 50, 0)

    split = table[table["view"] == "split"].set_index("group")
    assert split.index.tolist() == [
        "selection",
        "held_out",
        "held_out_nights",
        "held_out_stations",
        "held_out_both",
    ]
    assert split["n_cases"].tolist() == [1, 3, 1, 1, 1]
    assert split.loc["held_out", COUNT_NAMES].tolist() == [0, 1, 1, 1]


def test_score_table_views():
    table = table_of(score_cases())

    main = table.loc[("main", "all")]
    assert main[COUNT_NAMES].tolist() == [2, 1, 1, 1]
    assert main["n_cases"] == 5
    assert main["pod"] == pytest.approx(2 / 3)
    assert main["far"] == pytest.approx(1 / 3)
    assert main["csi"] == pytest.approx(2 / 4)
    assert main["bias"] == pytest.approx(1.0)
    assert main["csi_low"] <= main["csi_high"]
    assert (main["n_fls_ignored"], main["n_no_fls_ignored"], main["n_fls_hidden"]) == (1, 1, 1)

    assert table.loc[("with_hidden", "all"), COUNT_NAMES].tolist() == [2, 2, 1, 1]
    assert table.loc[("source", "metar"), COUNT_NAMES].tolist() == [1, 1, 1, 0]
    assert table.loc[("source", "synop"), COUNT_NAMES].tolist() == [2, 0, 0, 1]
    assert table.loc[("threshold_source", "dynamic"), COUNT_NAMES].tolist() == [1, 0, 1, 1]
    assert table.loc[("threshold_source", "fallback"), COUNT_NAMES].tolist() == [1, 1, 0, 0]
    assert table.loc[("lat_band", "43-47"), COUNT_NAMES].tolist() == [1, 0, 1, 0]
    assert table.loc[("lat_band", "47-51"), COUNT_NAMES].tolist() == [1, 1, 0, 1]
    assert table.loc[("slot", "21:00"), COUNT_NAMES].tolist() == [1, 0, 1, 1]
    assert table.loc[("slot", "00:00"), COUNT_NAMES].tolist() == [1, 1, 0, 0]
    assert table.loc[("fog_only", "all"), COUNT_NAMES].tolist() == [2, 0, 1, 2]


def test_score_table_counts_a_hidden_fls_case_with_ignored_label_in_both_diagnostics():
    cases = case_frame(
        [
            case("2023-11-20", "A", "metar", 1, 1, 255, hidden=True),
            case("2023-11-20", "B", "metar", 2, 0, 255),
        ]
    )

    main = table_of(cases).loc[("main", "all")]

    assert main["n_cases"] == 0
    assert (main["n_fls_ignored"], main["n_no_fls_ignored"], main["n_fls_hidden"]) == (1, 1, 1)


def test_score_table_puts_a_station_on_a_band_edge_into_the_upper_band():
    cases = case_frame([case("2023-11-20", "A", "metar", 1, 1, 1, lat=47.0)])

    table = table_of(cases)

    assert table.loc[("lat_band", "43-47"), "n_cases"] == 0
    assert table.loc[("lat_band", "47-51"), "n_cases"] == 1


def test_score_table_empty_group_is_nan():
    empty_band = table_of(score_cases()).loc[("lat_band", "51-56")]
    assert empty_band["n_cases"] == 0
    assert np.isnan(
        empty_band[["pod", "far", "csi", "csi_low", "csi_high", "bias", "hss"]].astype(float)
    ).all()


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


def test_read_observations_without_any_table_names_the_files_and_the_command(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="observations_metar.csv.*observations_synop.csv.*fogseg stations"
    ):
        read_observations(station_files(tmp_path, ()))


def test_read_observations_keeps_the_metar_columns_of_both_sources(tmp_path):
    data_dir = station_files(tmp_path, ())
    metar = observation_frame([("EDDF", "metar", "2023-11-21T00:20"), ("EDDM", "metar", "2023-11-21T00:20")])
    metar = metar.assign(
        auto=pd.array([True, False], dtype="boolean"),
        ceiling_amount=["BKN", ""],
        shallow_fog=pd.array([False, True], dtype="boolean"),
    )
    synop = observation_frame([("06240", "synop", "2023-11-21T00:00")]).assign(
        auto=pd.array([pd.NA], dtype="boolean"),
        ceiling_amount="",
        shallow_fog=pd.array([pd.NA], dtype="boolean"),
    )
    metar.to_csv(data_dir / "observations_metar.csv", index=False)
    synop.to_csv(data_dir / "observations_synop.csv", index=False)

    observations = read_observations(data_dir)

    assert observations["auto"].dtype == "boolean"
    assert observations["shallow_fog"].dtype == "boolean"
    assert observations["auto"].tolist() == [True, False, pd.NA]
    assert observations["ceiling_amount"].tolist() == ["BKN", "", ""]
    assert observations["shallow_fog"].tolist() == [False, True, pd.NA]


def test_read_observations_of_an_old_table_gives_empty_metar_columns(tmp_path):
    observations = read_observations(station_files(tmp_path, ("metar",)))

    assert observations["ceiling_amount"].tolist() == [""]
    assert observations["auto"].isna().all()
    assert observations["shallow_fog"].isna().all()


def validate_config(tmp_path: Path, dataset_path: Path) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset_path)
    cfg["dem"]["out_path"] = str(tmp_path / "dem.zarr")
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["mode"] = "fixed"
    cfg["stations"]["data_dir"] = str(tmp_path / "stations")
    cfg["stations"]["bootstrap_samples"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


FOG_AND_CLEAR = [
    ("FOG", GRID_LAT0 - 0.2, GRID_LON0 + 0.2, 500.0, "2023-11-21T00:12"),
    ("CLEAR", GRID_LAT0 - 0.5, GRID_LON0 + 0.1, 9999.0, "2023-11-21T00:12"),
]


def validate_setup(
    tmp_path: Path,
    observations: list[tuple[str, float, float, float, str]] = FOG_AND_CLEAR,
    n_scenes: int = 3,
) -> argparse.Namespace:
    """Dataset, fixed label store and a METAR table of (station, lat, lon, visibility, time), no SYNOP."""
    dataset_path = make_station_dataset(tmp_path / "data.zarr", n_scenes)
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", TEST_PARAMS)
    data_dir = station_files(tmp_path, ())
    pd.DataFrame(
        {
            "source": "metar",
            "station": [row[0] for row in observations],
            "time": pd.to_datetime([row[4] for row in observations]),
            "lat": [row[1] for row in observations],
            "lon": [row[2] for row in observations],
            "elevation_m": 100.0,
            "visibility_m": [row[3] for row in observations],
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


def test_validate_keeps_only_the_scenes_of_the_season(tmp_path):
    observations = [
        ("FOG", GRID_LAT0 - 0.2, GRID_LON0 + 0.2, 500.0, "2023-11-21T00:12"),
        ("FOG", GRID_LAT0 - 0.2, GRID_LON0 + 0.2, 500.0, "2023-11-21T12:12"),
    ]
    args = validate_setup(tmp_path, observations, n_scenes=6)
    cfg = load_config(args.config)
    cfg["time"]["seasons"].append({"name": "one-night", "start": "2023-11-20", "end": "2023-11-20"})
    args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args.season, args.final = "one-night", True

    cmd_validate(args)

    cases = pd.read_csv(tmp_path / "stations" / "cases_fixed_one-night.csv", parse_dates=["scene_time"])
    assert cases["scene_time"].tolist() == [pd.Timestamp("2023-11-21T00:00")]


def test_validate_lists_the_on_grid_stations_without_a_case_in_the_season(tmp_path, caplog):
    observations = [
        *FOG_AND_CLEAR,
        ("LATE", GRID_LAT0 - 0.3, GRID_LON0 + 0.3, 500.0, "2024-03-01T00:12"),
        ("ABSENT", GRID_LAT0 - 0.4, GRID_LON0 + 0.4, 500.0, "2024-03-02T00:12"),
        ("OFFGRID", 60.0, 0.0, 500.0, "2023-11-21T00:12"),
    ]
    args = validate_setup(tmp_path, observations)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_validate(args)

    assert "2 of 4 on-grid stations have no case in season 2023-24" in caplog.text
    assert "Stations without a case: ABSENT, LATE" in caplog.text
    assert "OFFGRID" not in caplog.text


def test_validate_does_not_list_stations_when_every_one_has_a_case(tmp_path, caplog):
    args = validate_setup(tmp_path)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_validate(args)

    assert "without a case" not in caplog.text


def test_validate_warns_and_still_writes_when_no_case_matches(tmp_path, caplog):
    args = validate_setup(tmp_path, [("FAR", 60.0, 0.0, 500.0, "2023-11-21T00:12")])

    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cmd_validate(args)

    data_dir = tmp_path / "stations"
    assert pd.read_csv(data_dir / "cases_fixed_2023-24.csv").empty
    assert (data_dir / "scores_fixed_2023-24.csv").exists()
    assert "2023-24" in caplog.text
    assert "observations_metar.csv" in caplog.text


def write_splits(args: argparse.Namespace) -> None:
    """The split file of season 2023-24 in the stations folder of the validate config."""
    cfg = load_config(args.config)
    splits = make_splits(
        np.datetime64("2023-10-01"), np.datetime64("2024-02-29"), cfg["region"]["bbox"], cfg["stations"]
    )
    splits.to_csv(splits_path(Path(cfg["stations"]["data_dir"]), "2023-24"), index=False)


def test_validate_adds_the_folds_and_scores_the_split_in_the_tuning_season(tmp_path, caplog):
    args = validate_setup(tmp_path)
    write_splits(args)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_validate(args)

    data_dir = tmp_path / "stations"
    cases = pd.read_csv(data_dir / "cases_fixed_2023-24.csv")
    scores = pd.read_csv(data_dir / "scores_fixed_2023-24.csv")
    assert cases["night_fold"].between(1, 5).all()
    assert cases["station_fold"].between(1, 5).all()
    assert "held_out" in scores.loc[scores["view"] == "split", "group"].tolist()
    assert "Full count" in caplog.text
    assert "held-out" in caplog.text


def test_validate_adds_no_folds_outside_the_tuning_season(tmp_path, caplog):
    args = validate_setup(tmp_path)
    cfg = load_config(args.config)
    cfg["stations"]["tune_season"] = "2024-25"
    args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_splits(args)
    args.final = True

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_validate(args)

    data_dir = tmp_path / "stations"
    assert "night_fold" not in pd.read_csv(data_dir / "cases_fixed_2023-24.csv").columns
    assert "split" not in pd.read_csv(data_dir / "scores_fixed_2023-24.csv")["view"].tolist()
    assert "held-out" not in caplog.text


def test_validate_fills_the_elevation_difference_from_the_dem_store(tmp_path):
    args = validate_setup(tmp_path)
    dem = xr.Dataset({"elevation_mean": (("y", "x"), np.full((LABEL_NY, LABEL_NX), 40.0, dtype="float32"))})
    dem.to_zarr(tmp_path / "dem.zarr", zarr_format=3, consolidated=False)

    cmd_validate(args)

    cases = pd.read_csv(tmp_path / "stations" / "cases_fixed_2023-24.csv")
    assert cases["elevation_diff_m"].tolist() == [60.0, 60.0]
