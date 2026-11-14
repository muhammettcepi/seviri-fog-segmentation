"""Tests for the Copernicus DEM on the SEVIRI grid."""

import argparse
import io
import logging
import shutil
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import yaml
from rasterio.transform import from_origin

from fogseg import dem
from fogseg.cli import cmd_dem
from fogseg.dem import (
    build_dem_store,
    fetch_tile,
    grid_statistics,
    http_fetch,
    read_dem,
    tile_name,
    tile_samples,
    tile_url,
    tiles_for_grid,
)


def write_geotiff(path: Path, values: np.ndarray, west: float, north: float, pixel: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, pixel, pixel),
    ) as file:
        file.write(values.astype("float32"), 1)


@pytest.fixture
def tile_file(tmp_path: Path) -> Path:
    """A fake tile of 20 x 20 pixels covering lon 5-6, lat 45-46; the elevation is the column index."""
    path = tmp_path / "source" / "tile.tif"
    path.parent.mkdir()
    write_geotiff(path, np.tile(np.arange(20.0), (20, 1)), west=5.0, north=46.0, pixel=0.05)
    return path


def copy_tile(source: Path, calls: list[str] | None = None):
    def fetch(url: str, path: Path) -> bool:
        if calls is not None:
            calls.append(url)
        shutil.copy(source, path)
        return True

    return fetch


def make_grid(lons: list[float], lats: list[float]) -> xr.Dataset:
    lon, lat = np.meshgrid(lons, lats)
    return xr.Dataset(
        coords={
            "y": ("y", np.arange(len(lats), 0.0, -1.0)),
            "x": ("x", np.arange(len(lons), dtype="float64") + 10.0),
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        }
    )


def test_tile_name_and_url():
    assert tile_name(47, 8) == "Copernicus_DSM_COG_30_N47_00_E008_00_DEM"
    assert tile_url("a") == "https://copernicus-dem-90m.s3.amazonaws.com/a/a.tif"


def test_tile_name_for_negative_longitude():
    assert tile_name(43, -1) == "Copernicus_DSM_COG_30_N43_00_W001_00_DEM"


def test_tiles_for_grid_cover_the_range():
    lon, lat = np.meshgrid([2.4, 3.6], [43.1, 44.2])
    assert sorted(tiles_for_grid(lon, lat)) == [(43, 2), (43, 3), (44, 2), (44, 3)]


def test_fetch_tile_caches_the_file(tmp_path, tile_file):
    calls = []
    first = fetch_tile(45, 5, tmp_path / "raw", copy_tile(tile_file, calls))
    second = fetch_tile(45, 5, tmp_path / "raw", copy_tile(tile_file, calls))
    assert first == second == tmp_path / "raw" / f"{tile_name(45, 5)}.tif"
    assert first.exists()
    assert len(calls) == 1


def test_missing_tile_is_marked_sea_and_not_fetched_again(tmp_path):
    calls = []

    def fetch(url, path):
        calls.append(url)
        return False

    assert fetch_tile(54, 2, tmp_path, fetch) is None
    assert fetch_tile(54, 2, tmp_path, fetch) is None
    assert len(calls) == 1
    assert (tmp_path / f"{tile_name(54, 2)}.sea").exists()


def test_tile_samples_block_means_with_transform_coordinates(tmp_path):
    path = tmp_path / "tile.tif"
    write_geotiff(path, np.arange(16.0).reshape(4, 4), west=5.0, north=46.0, pixel=0.25)
    lon, lat, elevation = tile_samples(path, factor=2)
    assert lon.tolist() == [5.25, 5.75, 5.25, 5.75]
    assert lat.tolist() == [45.75, 45.75, 45.25, 45.25]
    assert elevation.tolist() == [2.5, 4.5, 10.5, 12.5]


def test_tile_samples_use_the_pixel_size_of_each_file(tmp_path):
    path = tmp_path / "wide.tif"
    write_geotiff(path, np.ones((2, 2)), west=10.0, north=51.0, pixel=0.5)
    lon, _, _ = tile_samples(path, factor=2)
    assert lon.tolist() == [10.5]


def test_grid_statistics_mean_std_and_sea():
    lon, lat = np.meshgrid([5.0, 6.0], [45.0])
    mean, std = grid_statistics(
        np.array([5.0, 5.001, 5.002]), np.array([45.0, 45.0, 45.0]), np.array([100.0, 200.0, 300.0]), lon, lat
    )
    assert mean.tolist() == [[200.0, 0.0]]
    assert std[0, 0] == pytest.approx(np.std([100, 200, 300]))
    assert std[0, 1] == 0.0


def test_grid_statistics_ignore_samples_outside_the_grid():
    lon, lat = np.meshgrid([5.0, 5.1], [45.0, 45.1])
    mean, _ = grid_statistics(
        np.array([5.0, 7.0]), np.array([45.0, 47.0]), np.array([100.0, 900.0]), lon, lat
    )
    assert mean.max() == 100.0


def test_build_dem_store_aligns_with_the_grid(tmp_path, tile_file):
    grid = make_grid([5.25, 5.75], [45.75, 45.25])
    out_path = tmp_path / "dem.zarr"
    build_dem_store(grid, tmp_path / "raw", out_path, copy_tile(tile_file))

    with xr.open_zarr(out_path, consolidated=False) as store:
        assert store["elevation_mean"].dims == ("y", "x")
        assert store["elevation_std"].dims == ("y", "x")
        np.testing.assert_array_equal(store["y"], grid["y"])
        np.testing.assert_array_equal(store["x"], grid["x"])
        np.testing.assert_allclose(store["lon"], grid["lon"])
        np.testing.assert_allclose(store["lat"], grid["lat"])
        assert store["elevation_mean"].values.tolist() == [[4.5, 14.5], [4.5, 14.5]]
        assert (store["elevation_std"] == 0).all()


def test_build_dem_store_gives_the_relief_inside_a_pixel(tmp_path, tile_file):
    out_path = tmp_path / "dem.zarr"
    build_dem_store(make_grid([5.5], [45.5]), tmp_path / "raw", out_path, copy_tile(tile_file))
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert store["elevation_mean"].values.tolist() == [[9.5]]
        assert store["elevation_std"].values.tolist() == [[5.0]]


def test_build_dem_store_over_sea_is_zero(tmp_path):
    out_path = tmp_path / "dem.zarr"
    build_dem_store(make_grid([0.1, 0.4], [60.1, 60.4]), tmp_path / "raw", out_path, lambda url, path: False)
    with xr.open_zarr(out_path, consolidated=False) as store:
        assert (store["elevation_mean"] == 0).all()
        assert (store["elevation_std"] == 0).all()


def test_read_dem_returns_the_store(tmp_path, tile_file):
    out_path = tmp_path / "dem.zarr"
    build_dem_store(make_grid([5.25, 5.75], [45.25, 45.75]), tmp_path / "raw", out_path, copy_tile(tile_file))
    assert "elevation_mean" in read_dem(out_path)


def test_read_dem_without_store_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        assert read_dem(tmp_path / "missing.zarr") is None
    assert "missing.zarr" in caplog.text


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_fetch_saves_through_a_part_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dem.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"tiff"))
    path = tmp_path / "tile.tif"
    assert http_fetch("https://example.test/tile.tif", path) is True
    assert path.read_bytes() == b"tiff"
    assert not path.with_name("tile.tif.part").exists()


def raise_http_error(code: int):
    def urlopen(url, timeout):
        raise urllib.error.HTTPError(url, code, "error", {}, None)

    return urlopen


def test_http_fetch_returns_false_on_404(tmp_path, monkeypatch):
    monkeypatch.setattr(dem.urllib.request, "urlopen", raise_http_error(404))
    path = tmp_path / "tile.tif"
    assert http_fetch("https://example.test/tile.tif", path) is False
    assert not path.exists()


def test_http_fetch_raises_other_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(dem.urllib.request, "urlopen", raise_http_error(500))
    with pytest.raises(urllib.error.HTTPError):
        http_fetch("https://example.test/tile.tif", tmp_path / "tile.tif")


def write_config(tmp_path: Path, dataset_path: Path) -> Path:
    config = {
        "data": {"processed_path": str(dataset_path)},
        "dem": {"raw_dir": str(tmp_path / "raw"), "out_path": str(tmp_path / "dem.zarr")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cmd_dem_refuses_an_existing_store_without_overwrite(tmp_path):
    (tmp_path / "dem.zarr").mkdir()
    args = argparse.Namespace(config=str(write_config(tmp_path, tmp_path / "seviri.zarr")), overwrite=False)
    with pytest.raises(FileExistsError, match="--overwrite"):
        cmd_dem(args)


def test_cmd_dem_writes_the_store(tmp_path, monkeypatch, tile_file):
    dataset_path = tmp_path / "seviri.zarr"
    make_grid([5.25, 5.75], [45.25, 45.75]).to_zarr(dataset_path, zarr_format=3, consolidated=False)
    monkeypatch.setattr(dem, "http_fetch", copy_tile(tile_file))
    args = argparse.Namespace(config=str(write_config(tmp_path, dataset_path)), overwrite=False)
    cmd_dem(args)
    assert read_dem(tmp_path / "dem.zarr") is not None
