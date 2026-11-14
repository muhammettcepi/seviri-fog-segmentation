"""Tests for ERA5 surface fields on the SEVIRI grid."""

import argparse
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr
import yaml

from fogseg.cli import cmd_era5
from fogseg.era5 import (
    ERA5_VARIABLES,
    build_era5_store,
    download_months,
    era5_area,
    has_cds_credentials,
    month_requests,
    regrid,
)

AREA = [55.5, 1.5, 42.5, 20.5]
SCENES = np.array(["2023-11-20T21:00", "2023-11-21T03:00", "2023-12-01T00:00"], dtype="datetime64[ns]")


def write_era5_file(path: Path, hours: list[str], time_name: str = "valid_time") -> None:
    """A tiny ERA5 file: 2 x 2 points over lon 5-6, lat 45-46, latitude descending, like the CDS delivers.

    t2m rises with latitude by 1 K per degree and with the hour index by 10 K; the other fields are offsets of it.
    """
    lat, lon = np.array([46.0, 45.0]), np.array([5.0, 6.0])
    base = 270.0 + 10.0 * np.arange(len(hours))[:, None, None] + (lat - 45.0)[None, :, None] + 0 * lon
    dims = (time_name, "latitude", "longitude")
    xr.Dataset(
        {
            "t2m": (dims, base),
            "skt": (dims, base - 1.0),
            "d2m": (dims, base - 2.0),
            "sd": (dims, base * 0.0 + 0.01),
        },
        coords={
            time_name: np.array(hours, dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
            "number": 0,
            "expver": "0001",
        },
    ).to_netcdf(path)


def make_grid(lons: list[float], lats: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(lons, lats)
    return lon, lat, np.arange(len(lats), 0.0, -1.0), np.arange(len(lons), dtype="float64") + 10.0


def test_month_requests_group_scene_days_and_hours():
    times = np.array(["2023-11-20T21:00", "2023-11-21T03:00", "2023-12-01T00:00"], dtype="datetime64[ns]")
    requests = month_requests(times, [55.5, 1.5, 42.5, 20.5], ["2m_temperature"])
    assert [month for month, _ in requests] == ["2023-11", "2023-12"]
    november = requests[0][1]
    assert november["day"] == ["20", "21"]
    assert november["time"] == ["03:00", "21:00"]


def test_month_request_asks_for_netcdf_on_the_area():
    _, request = month_requests(SCENES[:1], AREA, ["2m_temperature", "snow_depth"])[0]
    assert request == {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "snow_depth"],
        "year": ["2023"],
        "month": ["11"],
        "day": ["20"],
        "time": ["21:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def test_era5_area_snaps_the_grid_plus_margin_outward_to_the_era5_grid():
    lon, lat = np.meshgrid([1.57, 14.0, 27.44], [42.59, 49.0, 56.12])
    assert era5_area(lon, lat) == [56.75, 1.0, 42.0, 28.0]


def test_era5_area_ignores_nan_pixels():
    lon, lat = np.array([[5.0, np.nan]]), np.array([[45.0, np.nan]])
    assert era5_area(lon, lat, margin=0.0) == [45.0, 5.0, 45.0, 5.0]


def test_era5_area_covers_a_grid_wider_than_the_bbox():
    lon, lat = np.meshgrid(np.linspace(1.57, 27.44, 50), np.linspace(42.59, 56.12, 40))
    north, west, south, east = era5_area(lon, lat)
    assert west <= lon.min()
    assert lon.max() <= east
    assert south <= lat.min()
    assert lat.max() <= north


def test_era5_variables_map_to_the_short_names_in_the_files():
    assert ERA5_VARIABLES == {
        "2m_temperature": "t2m",
        "skin_temperature": "skt",
        "2m_dewpoint_temperature": "d2m",
        "snow_depth": "sd",
    }


def test_credentials_need_the_key_file_or_both_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    assert not has_cds_credentials()

    monkeypatch.setenv("CDSAPI_KEY", "set")
    assert not has_cds_credentials()

    monkeypatch.setenv("CDSAPI_URL", "set")
    assert has_cds_credentials()

    monkeypatch.delenv("CDSAPI_URL")
    monkeypatch.delenv("CDSAPI_KEY")
    (tmp_path / ".cdsapirc").touch()
    assert has_cds_credentials()


def test_regrid_is_bilinear_for_descending_latitudes():
    field = xr.DataArray([[2.0, 2.0], [0.0, 0.0]], coords={"latitude": [46.0, 45.0], "longitude": [5.0, 6.0]})
    assert regrid(field, np.array([[5.5]]), np.array([[45.5]]))[0, 0] == pytest.approx(1.0)


def test_regrid_gives_the_same_result_for_ascending_latitudes():
    field = xr.DataArray([[0.0, 0.0], [2.0, 2.0]], coords={"latitude": [45.0, 46.0], "longitude": [5.0, 6.0]})
    lon, lat = np.meshgrid([5.25, 5.75], [45.25, 45.5, 45.75])
    np.testing.assert_allclose(regrid(field, lon, lat), [[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])


def test_regrid_is_nan_outside_the_era5_grid():
    field = xr.DataArray([[2.0, 2.0], [0.0, 0.0]], coords={"latitude": [46.0, 45.0], "longitude": [5.0, 6.0]})
    assert np.isnan(regrid(field, np.array([[7.0]]), np.array([[45.5]]))).all()


def test_download_months_writes_the_files_through_part_files(tmp_path):
    seen = []

    def retrieve(request, path):
        seen.append(path.name)
        path.write_text(request["month"])

    assert download_months([("2023-11", {"month": "11"})], tmp_path / "raw", retrieve) == []
    assert seen == ["era5_2023-11.nc.part"]
    assert [file.name for file in (tmp_path / "raw").iterdir()] == ["era5_2023-11.nc"]


def test_download_months_skips_existing_files(tmp_path):
    (tmp_path / "era5_2023-11.nc").write_text("done")
    calls = []
    failed = download_months([("2023-11", {})], tmp_path, lambda request, path: calls.append(path))
    assert failed == []
    assert calls == []


def test_failed_month_is_reported_and_not_written(tmp_path, caplog):
    def retrieve(request, path):
        path.write_text("half")
        raise RuntimeError("queue full")

    with caplog.at_level(logging.WARNING, logger="fogseg"):
        assert download_months([("2023-11", {})], tmp_path, retrieve) == ["2023-11"]
    assert not list(tmp_path.glob("*.nc*"))
    assert "2023-11" in caplog.text
    assert "queue full" in caplog.text


def test_failed_month_does_not_stop_the_other_months(tmp_path):
    def retrieve(request, path):
        if request["month"] == "11":
            raise RuntimeError("queue full")
        path.write_text("ok")

    requests = [("2023-11", {"month": "11"}), ("2023-12", {"month": "12"})]
    assert download_months(requests, tmp_path, retrieve) == ["2023-11"]
    assert (tmp_path / "era5_2023-12.nc").exists()


def test_build_era5_store_puts_the_scene_hour_on_the_grid(tmp_path):
    write_era5_file(
        tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00", "2023-11-21T00:00", "2023-11-21T03:00"]
    )
    lon, lat, y, x = make_grid([5.25, 5.75], [45.75, 45.25])
    out_path = tmp_path / "era5.zarr"
    build_era5_store(SCENES[:2], lon, lat, y, x, tmp_path, out_path)

    with xr.open_zarr(out_path, consolidated=False) as store:
        assert set(store.data_vars) == {"t2m", "skt", "d2m", "sd"}
        assert store["t2m"].dims == ("time", "y", "x")
        np.testing.assert_array_equal(store["time"], SCENES[:2])
        np.testing.assert_array_equal(store["y"], y)
        np.testing.assert_array_equal(store["x"], x)
        np.testing.assert_allclose(store["lon"], lon)
        np.testing.assert_allclose(store["lat"], lat)
        np.testing.assert_allclose(store["t2m"].isel(time=0), [[270.75, 270.75], [270.25, 270.25]])
        np.testing.assert_allclose(store["t2m"].isel(time=1), [[290.75, 290.75], [290.25, 290.25]])
        np.testing.assert_allclose(store["skt"].isel(time=0), store["t2m"].isel(time=0) - 1.0)
        np.testing.assert_allclose(store["d2m"].isel(time=0), store["t2m"].isel(time=0) - 2.0)
        np.testing.assert_allclose(store["sd"].isel(time=0), 0.01)


def test_build_era5_store_uses_the_nominal_hour_of_a_scene(tmp_path):
    write_era5_file(tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00", "2023-11-21T03:00"])
    scenes = np.array(["2023-11-21T03:12"], dtype="datetime64[ns]")
    out_path = tmp_path / "era5.zarr"
    build_era5_store(scenes, *make_grid([5.5], [45.5]), tmp_path, out_path)
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert store["t2m"].values.ravel() == pytest.approx([280.5])


def test_build_era5_store_reads_files_with_a_time_coordinate(tmp_path):
    write_era5_file(tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00"], time_name="time")
    out_path = tmp_path / "era5.zarr"
    build_era5_store(SCENES[:1], *make_grid([5.5], [45.5]), tmp_path, out_path)
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert store["t2m"].values.ravel() == pytest.approx([270.5])


def test_scene_without_era5_file_is_nan(tmp_path):
    write_era5_file(tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00", "2023-11-21T03:00"])
    out_path = tmp_path / "era5.zarr"
    build_era5_store(SCENES, *make_grid([5.25, 5.75], [45.75, 45.25]), tmp_path, out_path)
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert np.isfinite(store["t2m"].sel(time="2023-11-20T21:00")).all()
        assert np.isnan(store["t2m"].sel(time="2023-12-01T00:00")).all()
        assert np.isnan(store["sd"].sel(time="2023-12-01T00:00")).all()


def test_scene_hour_missing_from_its_file_is_nan(tmp_path):
    write_era5_file(tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00"])
    out_path = tmp_path / "era5.zarr"
    build_era5_store(SCENES[:2], *make_grid([5.5], [45.5]), tmp_path, out_path)
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert np.isfinite(store["t2m"].isel(time=0)).all()
        assert np.isnan(store["t2m"].isel(time=1)).all()


def test_build_era5_store_replaces_an_existing_store(tmp_path):
    write_era5_file(tmp_path / "era5_2023-11.nc", ["2023-11-20T21:00"])
    out_path = tmp_path / "era5.zarr"
    out_path.mkdir()
    (out_path / "old").touch()
    build_era5_store(SCENES[:1], *make_grid([5.5], [45.5]), tmp_path, out_path)
    assert not (out_path / "old").exists()
    assert not list(tmp_path.glob("*.part"))


def write_config(tmp_path: Path, dataset_path: Path) -> Path:
    config = {
        "data": {"processed_path": str(dataset_path)},
        "region": {"bbox": [2, 43, 20, 55]},
        "era5": {
            "raw_dir": str(tmp_path / "raw"),
            "out_path": str(tmp_path / "era5.zarr"),
            "variables": list(ERA5_VARIABLES),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def write_dataset(path: Path, n_scenes: int = 2) -> None:
    lon, lat, y, x = make_grid([5.25, 5.75], [45.75, 45.25])
    xr.Dataset(
        coords={
            "time": SCENES[:n_scenes],
            "y": y,
            "x": x,
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        }
    ).to_zarr(path, zarr_format=3, consolidated=False)


class FakeClient:
    requests: ClassVar[list[dict]] = []
    failing_months: ClassVar[set[str]] = set()

    def retrieve(self, name: str, request: dict, target: str) -> None:
        assert name == "reanalysis-era5-single-levels"
        FakeClient.requests.append(request)
        if request["month"][0] in FakeClient.failing_months:
            raise RuntimeError("queue full")
        write_era5_file(Path(target), ["2023-11-20T21:00", "2023-11-21T03:00"])


@pytest.fixture
def cds_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    FakeClient.requests = []
    FakeClient.failing_months = set()
    monkeypatch.setattr("cdsapi.Client", FakeClient)
    return tmp_path / "home"


def test_cmd_era5_refuses_an_existing_store_without_overwrite(tmp_path, cds_environment):
    (tmp_path / "era5.zarr").mkdir()
    args = argparse.Namespace(config=str(write_config(tmp_path, tmp_path / "seviri.zarr")), overwrite=False)
    with pytest.raises(FileExistsError, match="--overwrite"):
        cmd_era5(args)


def test_cmd_era5_without_credentials_points_to_the_cds_instructions(tmp_path, cds_environment):
    write_dataset(tmp_path / "seviri.zarr")
    args = argparse.Namespace(config=str(write_config(tmp_path, tmp_path / "seviri.zarr")), overwrite=False)
    with pytest.raises(SystemExit, match=r"https://cds\.climate\.copernicus\.eu/how-to-api"):
        cmd_era5(args)
    assert FakeClient.requests == []
    assert not (tmp_path / "era5.zarr").exists()


def test_cmd_era5_downloads_the_months_and_writes_the_store(tmp_path, cds_environment):
    cds_environment.mkdir()
    (cds_environment / ".cdsapirc").touch()
    write_dataset(tmp_path / "seviri.zarr")
    args = argparse.Namespace(config=str(write_config(tmp_path, tmp_path / "seviri.zarr")), overwrite=False)
    cmd_era5(args)

    assert [request["month"] for request in FakeClient.requests] == [["11"]]
    assert FakeClient.requests[0]["area"] == [46.25, 4.75, 44.75, 6.25]
    with xr.open_zarr(tmp_path / "era5.zarr", consolidated=False) as store:
        np.testing.assert_array_equal(store["time"], SCENES[:2])
        assert np.isfinite(store["t2m"]).all()

    cmd_era5(argparse.Namespace(config=args.config, overwrite=True))
    assert len(FakeClient.requests) == 1


def test_cmd_era5_builds_no_store_while_a_month_is_missing(tmp_path, cds_environment, caplog):
    cds_environment.mkdir()
    (cds_environment / ".cdsapirc").touch()
    write_dataset(tmp_path / "seviri.zarr", n_scenes=3)
    args = argparse.Namespace(config=str(write_config(tmp_path, tmp_path / "seviri.zarr")), overwrite=False)

    FakeClient.failing_months = {"12"}
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cmd_era5(args)
    assert not (tmp_path / "era5.zarr").exists()
    assert "2023-12" in caplog.text
    assert "fogseg era5" in caplog.text

    FakeClient.failing_months = set()
    FakeClient.requests = []
    cmd_era5(args)
    assert [request["month"] for request in FakeClient.requests] == [["12"]]
    with xr.open_zarr(tmp_path / "era5.zarr", consolidated=False) as store:
        np.testing.assert_array_equal(store["time"], SCENES)
