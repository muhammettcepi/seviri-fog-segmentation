"""Copernicus GLO-90 elevation averaged onto the SEVIRI grid.

The DEM comes as 1 x 1 degree tiles named by their south-west corner. Ocean tiles do not exist; they count as
sea level. Block means of the tile pixels are assigned to the nearest SEVIRI pixel, whose mean and standard
deviation give the elevation and the relief inside the pixel.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from scipy.spatial import cKDTree

from fogseg.parallax import unit_vector

log = logging.getLogger("fogseg")

TILE_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
BLOCK_FACTOR = 10  # tile pixels per block side: 3 arcsec pixels give 30 arcsec blocks, about 1 km
DOWNLOAD_TIMEOUT_S = 120


def tile_name(lat: int, lon: int) -> str:
    """Name of the tile whose south-west corner is at `lat`, `lon`, e.g. `Copernicus_DSM_COG_30_N47_00_E008_00_DEM`."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tile_url(name: str) -> str:
    return TILE_URL.format(name=name)


def tiles_for_grid(lon: np.ndarray, lat: np.ndarray) -> list[tuple[int, int]]:
    """South-west corners (lat, lon) of all tiles touching the range of the grid."""
    lats = range(int(np.floor(np.min(lat))), int(np.floor(np.max(lat))) + 1)
    lons = range(int(np.floor(np.min(lon))), int(np.floor(np.max(lon))) + 1)
    return [(tile_lat, tile_lon) for tile_lat in lats for tile_lon in lons]


def http_fetch(url: str, path: Path) -> bool:
    """Download `url` to `path` through a `.part` file; False if the server has no such file (HTTP 404)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
    except urllib.error.HTTPError as err:
        partial.unlink(missing_ok=True)
        if err.code == 404:
            return False
        raise
    partial.replace(path)
    return True


def fetch_tile(lat: int, lon: int, raw_dir: Path, fetch: Callable[[str, Path], bool]) -> Path | None:
    """The cached tile file, downloaded with `fetch` if needed; None for a sea tile, remembered as `<name>.sea`."""
    name = tile_name(lat, lon)
    path = raw_dir / f"{name}.tif"
    sea_marker = raw_dir / f"{name}.sea"
    if path.exists():
        return path
    if sea_marker.exists():
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    if fetch(tile_url(name), path):
        return path
    sea_marker.touch()
    return None


def tile_samples(path: Path, factor: int = BLOCK_FACTOR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longitude, latitude and mean elevation of the `factor` x `factor` pixel blocks of a tile."""
    with rasterio.open(path) as file:
        elevation = file.read(1).astype("float64")
        transform = file.transform

    n_rows, n_cols = (size // factor for size in elevation.shape)
    blocks = elevation[: n_rows * factor, : n_cols * factor].reshape(n_rows, factor, n_cols, factor)
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    lon, lat = transform @ ((cols + 0.5) * factor, (rows + 0.5) * factor)
    return lon.ravel(), lat.ravel(), blocks.mean(axis=(1, 3)).ravel()


def grid_statistics(
    sample_lon: np.ndarray,
    sample_lat: np.ndarray,
    sample_elevation: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the samples nearest to each grid pixel; 0 and 0 where there are none.

    Samples farther from every pixel than the pixel spacing lie outside the grid and are dropped.
    """
    pixels = cKDTree(unit_vector(np.ravel(lon), np.ravel(lat)))
    spacing = pixels.query(pixels.data, k=2)[0][:, 1].max()
    _, nearest = pixels.query(
        unit_vector(np.ravel(sample_lon), np.ravel(sample_lat)), distance_upper_bound=spacing
    )
    inside = nearest < pixels.n
    nearest, elevation = nearest[inside], np.asarray(sample_elevation, dtype="float64")[inside]

    count = np.bincount(nearest, minlength=pixels.n)
    has_samples = count > 0
    mean = np.zeros(pixels.n)
    mean[has_samples] = (
        np.bincount(nearest, weights=elevation, minlength=pixels.n)[has_samples] / count[has_samples]
    )
    variance = np.zeros(pixels.n)
    squares = np.bincount(nearest, weights=(elevation - mean[nearest]) ** 2, minlength=pixels.n)
    variance[has_samples] = squares[has_samples] / count[has_samples]
    return mean.reshape(np.shape(lon)), np.sqrt(variance).reshape(np.shape(lon))


def build_dem_store(
    data: xr.Dataset, raw_dir: Path, out_path: Path, fetch: Callable[[str, Path], bool]
) -> None:
    """Write `elevation_mean` and `elevation_std` (metres) on the grid of `data` to the Zarr store `out_path`."""
    lon, lat = data["lon"].values, data["lat"].values
    tiles = tiles_for_grid(lon, lat)
    samples = []
    for i, (tile_lat, tile_lon) in enumerate(tiles, start=1):
        path = fetch_tile(tile_lat, tile_lon, raw_dir, fetch)
        log.info(
            "Tile %d/%d %s: %s", i, len(tiles), tile_name(tile_lat, tile_lon), "sea" if path is None else "ok"
        )
        if path is not None:
            samples.append(tile_samples(path))

    if samples:
        sample_lon, sample_lat, sample_elevation = (
            np.concatenate(parts) for parts in zip(*samples, strict=True)
        )
    else:
        sample_lon = sample_lat = sample_elevation = np.empty(0)
    mean, std = grid_statistics(sample_lon, sample_lat, sample_elevation, lon, lat)

    store = xr.Dataset(
        {
            "elevation_mean": (("y", "x"), mean.astype("float32"), {"units": "m"}),
            "elevation_std": (("y", "x"), std.astype("float32"), {"units": "m"}),
        },
        coords={
            "y": data["y"].values,
            "x": data["x"].values,
            "lon": (("y", "x"), lon, {"units": "degrees_east"}),
            "lat": (("y", "x"), lat, {"units": "degrees_north"}),
        },
        attrs={"source": "Copernicus GLO-90 DEM, 30 arcsec block means; sea level where no tile exists"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store.to_zarr(out_path, mode="w", zarr_format=3, consolidated=False)


def read_dem(path: Path) -> xr.Dataset | None:
    """The DEM store that `fogseg dem` wrote, or None with a warning if it is missing."""
    if not path.exists():
        log.warning("%s is missing; run `fogseg dem` first", path)
        return None
    with xr.open_zarr(path, consolidated=False) as store:
        return store.load()
