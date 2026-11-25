"""ERA5 surface fields at the scene times, interpolated onto the SEVIRI grid.

One request per month asks the Copernicus Climate Data Store for the hours of that month's scenes. The files are
cached as `era5_<year>-<month>.nc`; the store holds one field per scene, taken at the nominal hour of the scene.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from fogseg.build import CHUNK_TIME
from fogseg.label_store import TIME_ENCODING

log = logging.getLogger("fogseg")

ERA5_VARIABLES = {
    "2m_temperature": "t2m",
    "skin_temperature": "skt",
    "2m_dewpoint_temperature": "d2m",
    "snow_depth": "sd",
}
UNITS = {"t2m": "K", "skt": "K", "d2m": "K", "sd": "m of water equivalent"}
AREA_MARGIN_DEG = 0.5
ERA5_GRID_DEG = 0.25


def era5_area(lon: np.ndarray, lat: np.ndarray, margin: float = AREA_MARGIN_DEG) -> list[float]:
    """The CDS area [north, west, south, east] around the grid, snapped outward to the ERA5 grid."""

    def snap(value: float, round_up: bool) -> float:
        steps = value / ERA5_GRID_DEG
        return float(np.ceil(steps) if round_up else np.floor(steps)) * ERA5_GRID_DEG

    return [
        snap(np.nanmax(lat) + margin, round_up=True),
        snap(np.nanmin(lon) - margin, round_up=False),
        snap(np.nanmin(lat) - margin, round_up=False),
        snap(np.nanmax(lon) + margin, round_up=True),
    ]


def has_cds_credentials() -> bool:
    """Whether a CDS key is configured (key file, or URL and key in the environment); never opens the file."""
    in_environment = "CDSAPI_URL" in os.environ and "CDSAPI_KEY" in os.environ
    return (Path.home() / ".cdsapirc").exists() or in_environment


def month_requests(times: np.ndarray, area: list[float], variables: list[str]) -> list[tuple[str, dict]]:
    """One CDS request per month with scenes, asking for the days and hours of that month's scenes."""
    scenes = pd.DataFrame({"month": pd.DatetimeIndex(times).strftime("%Y-%m")})
    scenes["day"] = pd.DatetimeIndex(times).strftime("%d")
    scenes["hour"] = pd.DatetimeIndex(times).strftime("%H:00")

    requests = []
    for month, month_scenes in scenes.groupby("month"):
        year, month_number = month.split("-")
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [year],
            "month": [month_number],
            "day": sorted(month_scenes["day"].unique()),
            "time": sorted(month_scenes["hour"].unique()),
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        requests.append((month, request))
    return requests


def month_file(raw_dir: Path, month: str) -> Path:
    return raw_dir / f"era5_{month}.nc"


def download_months(
    requests: list[tuple[str, dict]], raw_dir: Path, retrieve: Callable[[dict, Path], None]
) -> list[str]:
    """Download the months that have no file yet; the months that failed, logged and returned."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (month, request) in enumerate(requests, start=1):
        path = month_file(raw_dir, month)
        if path.exists():
            log.info("ERA5 %d/%d %s: cached", i, len(requests), month)
            continue

        partial = path.with_name(path.name + ".part")
        try:
            retrieve(request, partial)
            partial.replace(path)
            log.info("ERA5 %d/%d %s: downloaded", i, len(requests), month)
        except Exception as err:  # noqa: BLE001 - a failed month must not stop the other months
            partial.unlink(missing_ok=True)
            log.warning("ERA5 %s failed: %s", month, err)
            failed.append(month)
    return failed


def regrid(field: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (latitude, longitude) field to the points `lon`, `lat`; NaN outside it."""
    field = field.sortby(["latitude", "longitude"])
    interpolator = RegularGridInterpolator(
        (field["latitude"].values, field["longitude"].values), field.values, bounds_error=False
    )
    return interpolator(np.stack([lat, lon], axis=-1))


def scene_fields(times: np.ndarray, lon: np.ndarray, lat: np.ndarray, path: Path) -> dict[str, np.ndarray]:
    """The fields at the hours of `times` from one month file; NaN for a missing file or hour."""
    fields = {name: np.full((len(times), *lon.shape), np.nan, dtype="float32") for name in UNITS}
    if not path.exists():
        return fields

    hours = times.astype("datetime64[h]").astype("datetime64[ns]")
    with xr.open_dataset(path) as raw:
        time_name = "valid_time" if "valid_time" in raw.coords else "time"
        file_index = {hour: i for i, hour in enumerate(raw[time_name].values.astype("datetime64[ns]"))}
        for scene, hour in enumerate(hours):
            if hour not in file_index:
                continue
            for name, values in fields.items():
                values[scene] = regrid(raw[name].isel({time_name: file_index[hour]}), lon, lat)
    return fields


def build_era5_store(
    times: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    raw_dir: Path,
    out_path: Path,
) -> None:
    """Write `t2m`, `skt`, `d2m` (K) and `sd` (m water equivalent) per scene on the grid to `out_path`."""
    partial = out_path.with_name(out_path.name + ".part")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)

    months = np.asarray(pd.DatetimeIndex(times).strftime("%Y-%m"))
    starts = np.flatnonzero(np.concatenate([[True], months[1:] != months[:-1]]))
    stops = np.append(starts[1:], len(times))
    encoding = {name: {"chunks": (CHUNK_TIME, *lon.shape)} for name in UNITS} | {"time": TIME_ENCODING}

    try:
        for start, stop in zip(starts, stops, strict=True):
            fields = scene_fields(times[start:stop], lon, lat, month_file(raw_dir, months[start]))
            batch = xr.Dataset(
                {
                    name: (("time", "y", "x"), values, {"units": UNITS[name]})
                    for name, values in fields.items()
                },
                coords={
                    "time": times[start:stop],
                    "y": y,
                    "x": x,
                    "lon": (("y", "x"), lon, {"units": "degrees_east"}),
                    "lat": (("y", "x"), lat, {"units": "degrees_north"}),
                },
                attrs={"source": "ERA5 single levels, bilinear to the SEVIRI grid, hour of the scene"},
            )
            if start == 0:
                batch.to_zarr(partial, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
            else:
                batch.drop_vars(["x", "y", "lon", "lat"]).to_zarr(
                    partial, append_dim="time", consolidated=False
                )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    if out_path.exists():
        shutil.rmtree(out_path)
    partial.rename(out_path)
