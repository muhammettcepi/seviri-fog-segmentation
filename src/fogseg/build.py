"""Stack the per-scene NetCDF files into one ML-ready Zarr store.

Result: one dataset with dimensions (time, channel, y, x), plus everything a user needs to work
with it without reading our code: pixel latitude/longitude, the solar zenith angle per scene (for
the night mask), units, channel wavelengths, which satellite and source file each scene came from,
and the processing version.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyorbital.astronomy import sun_zenith_angle
from pyresample.geometry import AreaDefinition

import fogseg
from fogseg.preprocess import GRID_FILE

SCENE_GLOB = "seviri_ceu_*.nc"
CHUNK_TIME = 8  # scenes per Zarr chunk: ~28 MB of float32 brightness temperatures
# No real Earth scene is this cold. Over very cold cloud tops (~210 K) the 3.9 µm signal is below
# the instrument noise, calibration then yields ~0 K; such values are set to NaN.
MIN_VALID_BT = 150.0


def grid_lonlats(grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every pixel centre, from the grid saved by `fogseg fetch`."""
    area = AreaDefinition(
        "ceu",
        "cropped SEVIRI grid",
        "ceu",
        grid.attrs["crs_wkt"],
        grid.sizes["x"],
        grid.sizes["y"],
        grid.attrs["area_extent"],
    )
    lons, lats = area.get_lonlats()
    return lons.astype("float32"), lats.astype("float32")


def load_scene(path: Path, grid: xr.Dataset, channels: list[str]) -> xr.Dataset:
    """One scene as a 1-element slice of the final dataset (time dimension of length 1)."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    if not (np.array_equal(ds["x"], grid["x"]) and np.array_equal(ds["y"], grid["y"])):
        raise ValueError(f"{path.name} is not on the reference grid")
    if list(ds["channel"].values) != channels:
        raise ValueError(f"{path.name} has channels {list(ds['channel'].values)}, expected {channels}")

    time = np.datetime64(dt.datetime.fromisoformat(ds.attrs["time"]), "ns")
    bt = ds["bt"].where(ds["bt"] >= MIN_VALID_BT)
    return xr.Dataset(
        {
            "bt": bt.expand_dims(time=[time]),
            "platform": ("time", [ds.attrs["platform"]]),
            "source_file": ("time", [ds.attrs["source_file"]]),
        }
    )


def add_sun_zenith(scenes: xr.Dataset, lons: np.ndarray, lats: np.ndarray) -> xr.Dataset:
    """Solar zenith angle per pixel and scene (degrees). > 90° = sun below the horizon."""
    angles = []
    for time in scenes["time"].values:
        utc_time = pd.Timestamp(time).to_pydatetime()
        angles.append(sun_zenith_angle(utc_time, lons, lats))
    sza = np.stack(angles).astype("float32")

    scenes["sza"] = (("time", "y", "x"), sza, {"units": "degree", "long_name": "solar zenith angle"})
    return scenes


def channel_wavelength(channel: str) -> float:
    """Central wavelength in µm from a SEVIRI channel name, e.g. "IR_108" -> 10.8."""
    digits = channel.split("_")[1]
    return int(digits) / 10


def build_zarr(
    interim_dir: Path,
    out_path: Path,
    channels: list[str],
    overwrite: bool = False,
    batch: int = 4 * CHUNK_TIME,
) -> int:
    """Stack all scenes in `interim_dir` into the Zarr store `out_path`. Returns the number of scenes.

    Scenes are appended in batches, so memory stays small however many scenes there are.
    The store is written under a temporary name and renamed at the end, so an interrupted run
    never leaves a half-finished dataset that looks complete.
    """
    files = sorted(interim_dir.glob(SCENE_GLOB))  # file names sort chronologically
    if not files:
        raise FileNotFoundError(f"No scenes matching {SCENE_GLOB} in {interim_dir}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; use overwrite=True (--overwrite) to rebuild")

    with xr.open_dataset(interim_dir / GRID_FILE) as grid:
        grid = grid.load()
    lons, lats = grid_lonlats(grid)

    tmp_path = out_path.with_name(out_path.name + ".part")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = grid.sizes["y"], grid.sizes["x"]
    encoding = {
        "bt": {"chunks": (CHUNK_TIME, len(channels), ny, nx)},
        "sza": {"chunks": (CHUNK_TIME, ny, nx)},
        # fixed time units, so later appended batches are stored exactly the same way
        "time": {"units": "minutes since 2000-01-01", "dtype": "int64"},
    }

    n_invalid = 0
    for start in range(0, len(files), batch):
        batch_files = files[start : start + batch]
        scenes = xr.concat([load_scene(f, grid, channels) for f in batch_files], dim="time")
        scenes = add_sun_zenith(scenes, lons, lats)
        n_invalid += int(scenes["bt"].isnull().sum())

        is_first_batch = start == 0
        if is_first_batch:
            scenes = scenes.assign_coords(
                lon=(("y", "x"), lons, {"units": "degrees_east"}),
                lat=(("y", "x"), lats, {"units": "degrees_north"}),
                wavelength=("channel", [channel_wavelength(ch) for ch in channels], {"units": "um"}),
            )
            # plain Zarr v3 (consolidated metadata is not part of the v3 standard)
            scenes.to_zarr(tmp_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
        else:
            scenes = scenes.drop_vars(["x", "y", "channel"])
            scenes.to_zarr(tmp_path, append_dim="time", consolidated=False)

    # Dataset-level metadata is written once at the end: appending batches would overwrite it
    zarr.open_group(tmp_path, mode="r+").attrs.update(
        {
            "title": "Meteosat SEVIRI night-time IR brightness temperatures, Central Europe",
            "source": "EUMETSAT Data Store, collection EO:EUM:DAT:MSG:HRSEVIRI (Level 1.5, 0 degree)",
            "processing": f"fogseg {fogseg.__version__}: satpy seviri_l1b_native, "
            "calibration=brightness_temperature, cropped on the native grid",
            "invalid_values": f"{n_invalid} brightness temperatures < {MIN_VALID_BT} K set to NaN",
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "area_id": grid.attrs["area_id"],
            "area_extent": [float(v) for v in grid.attrs["area_extent"]],
            "crs_wkt": grid.attrs["crs_wkt"],
            "region_bbox": [float(v) for v in grid.attrs["region_bbox"]],
        }
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    return len(files)
