"""Tests for the station truth rule, the matching and the case table."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import COLD_PIXEL, GRID_LAT0, GRID_LON0, TEST_PARAMS, make_station_dataset

from fogseg.label_store import label_dataset
from fogseg.validation import (
    deduplicate,
    fls_truth,
    fog_truth,
    match_observations,
    nearest_pixels,
    night_of,
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
