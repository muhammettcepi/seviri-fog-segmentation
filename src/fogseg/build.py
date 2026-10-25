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
from fogseg.config import GRID_FILE, SCENE_GLOB

CHUNK_TIME = 8  # scenes per Zarr chunk: ~28 MB of float32 brightness temperatures
# No real Earth scene is this cold. Over very cold cloud tops (~210 K) the 3.9 µm signal is below
# the instrument noise, calibration then yields ~0 K; such values are set to NaN.
MIN_VALID_BT = 150.0
# satpy gives the nominal start of the 15-minute repeat cycle, but SEVIRI scans the disk from south
# to north, 3 lines per 0.6 s spin. Full-disk geometry from satpy's msg_seviri_fes_3km area:
SEVIRI_SOUTH_EDGE_Y = -5_567_248.283  # metres
SEVIRI_LINE_SPACING = 3_000.403  # metres
SECONDS_PER_LINE = 0.6 / 3


def variable_length_strings(dataset: xr.Dataset) -> xr.Dataset:
    """The same dataset, with fixed-width text variables turned into variable-length strings.

    Zarr v3 has no stable specification for fixed-width unicode: older zarr-python versions warn
    that such arrays may become unreadable. Variable-length strings are part of the specification.
    """
    converted = dataset
    for name, variable in dataset.variables.items():
        if variable.dtype.kind == "U":
            converted = converted.assign({name: variable.astype(object)})
    return converted


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
    return xr.Dataset(
        {
            "bt": ds["bt"].expand_dims(time=[time]),
            "platform": ("time", [ds.attrs["platform"]]),
            "source_file": ("time", [ds.attrs["source_file"]]),
        }
    )


def mask_invalid(scenes: xr.Dataset) -> tuple[xr.Dataset, int]:
    """Set physically impossible brightness temperatures to NaN. Returns the data and their count."""
    invalid = scenes["bt"] < MIN_VALID_BT
    scenes["bt"] = scenes["bt"].where(~invalid)
    return scenes, int(invalid.sum())


def check_time_order(times: pd.DatetimeIndex, previous_time: pd.Timestamp | None) -> None:
    """Scene times must be unique and increasing, also across batches."""
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError(f"Scene times are not unique and increasing: {list(times)}")
    if previous_time is not None and times[0] <= previous_time:
        raise ValueError(f"Scene at {times[0]} does not come after {previous_time}")


def region_scan_delay(y: np.ndarray) -> dt.timedelta:
    """Time from the nominal slot start until SEVIRI scans the middle of the region."""
    lines_from_south = (np.mean(y) - SEVIRI_SOUTH_EDGE_Y) / SEVIRI_LINE_SPACING
    return dt.timedelta(seconds=float(lines_from_south * SECONDS_PER_LINE))


def add_sun_zenith(
    scenes: xr.Dataset, lons: np.ndarray, lats: np.ndarray, scan_delay: dt.timedelta
) -> xr.Dataset:
    """Solar zenith angle per pixel and scene (degrees), at the time the region was scanned."""
    angles = []
    for time in scenes["time"].values:
        scan_time = pd.Timestamp(time).to_pydatetime() + scan_delay
        angles.append(sun_zenith_angle(scan_time, lons, lats))
    sza = np.stack(angles).astype("float32")

    delay_minutes = scan_delay.total_seconds() / 60
    attrs = {
        "units": "degree",
        "long_name": "solar zenith angle",
        "comment": f"computed at time + {delay_minutes:.1f} min, when SEVIRI scans this region; "
        "> 90 = sun below the horizon",
    }
    scenes["sza"] = (("time", "y", "x"), sza, attrs)
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
    The store is written under a temporary name and renamed at the end; if anything fails, the
    temporary store is deleted, so no half-finished dataset is ever left behind.
    """
    files = sorted(interim_dir.glob(SCENE_GLOB))
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
    try:
        n_invalid = write_batches(files, tmp_path, grid, channels, lons, lats, batch)
        write_metadata(tmp_path, grid, n_invalid)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    return len(files)


def write_batches(
    files: list[Path],
    store: Path,
    grid: xr.Dataset,
    channels: list[str],
    lons: np.ndarray,
    lats: np.ndarray,
    batch: int,
) -> int:
    """Write all scenes to `store`, `batch` scenes at a time. Returns the number of invalid values."""
    ny, nx = grid.sizes["y"], grid.sizes["x"]
    encoding = {
        "bt": {"chunks": (CHUNK_TIME, len(channels), ny, nx)},
        "sza": {"chunks": (CHUNK_TIME, ny, nx)},
        # fixed time units, so later appended batches are stored exactly the same way
        "time": {"units": "minutes since 2000-01-01", "dtype": "int64"},
    }

    scan_delay = region_scan_delay(grid["y"].values)
    n_invalid = 0
    previous_time = None
    for start in range(0, len(files), batch):
        batch_files = files[start : start + batch]
        scenes = xr.concat([load_scene(f, grid, channels) for f in batch_files], dim="time")

        times = pd.DatetimeIndex(scenes["time"].values)
        check_time_order(times, previous_time)
        previous_time = times[-1]

        scenes, batch_invalid = mask_invalid(scenes)
        n_invalid += batch_invalid
        scenes = add_sun_zenith(scenes, lons, lats, scan_delay)
        scenes = variable_length_strings(scenes)

        is_first_batch = start == 0
        if is_first_batch:
            scenes["time"].attrs["comment"] = "nominal start of the SEVIRI repeat cycle (UTC)"
            scenes = scenes.assign_coords(
                lon=(("y", "x"), lons, {"units": "degrees_east"}),
                lat=(("y", "x"), lats, {"units": "degrees_north"}),
                wavelength=("channel", [channel_wavelength(ch) for ch in channels], {"units": "um"}),
            )
            # Zarr v3 without consolidated metadata, which is not part of the v3 standard
            scenes.to_zarr(store, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
        else:
            # x, y and channel are already in the store; appending them again would silently
            # overwrite them with this batch's values
            scenes = scenes.drop_vars(["x", "y", "channel"])
            scenes.to_zarr(store, append_dim="time", consolidated=False)
    return n_invalid


def write_metadata(store: Path, grid: xr.Dataset, n_invalid: int) -> None:
    """Dataset-level metadata, written once at the end: appending batches would overwrite it."""
    zarr.open_group(store, mode="r+").attrs.update(
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
