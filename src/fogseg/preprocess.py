"""Read a SEVIRI .nat file with satpy, crop it to the region and save it as a small NetCDF file."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import xarray as xr
from satpy import Scene

GRID_FILE = "grid.nc"
# The NetCDF/HDF5 C library is not thread-safe: two workers reading or writing .nc files at the
# same moment can crash the whole process (segmentation fault). Only one thread may do NetCDF I/O.
_nc_lock = threading.Lock()


def crop_scene(nat_path: Path, channels: list[str], bbox: list[float]) -> xr.Dataset:
    """Calibrated brightness temperatures (K) of `channels`, cropped to `bbox`, on the native grid.

    We keep SEVIRI's own geostationary grid (no resampling), so there are no interpolation
    artefacts and every scene cropped with the same bbox has exactly the same pixels.
    """
    scn = Scene(filenames=[str(nat_path)], reader="seviri_l1b_native")
    scn.load(channels, calibration="brightness_temperature", upper_right_corner="NE")
    ceu = scn.crop(ll_bbox=tuple(bbox))

    ref = ceu[channels[0]]
    area = ref.attrs["area"]
    bt = np.stack([ceu[ch].values.astype("float32") for ch in channels])

    return xr.Dataset(
        {"bt": (("channel", "y", "x"), bt, {"units": "K", "long_name": "brightness temperature"})},
        coords={"channel": channels, "y": ref["y"].values, "x": ref["x"].values},
        attrs={
            "time": ref.attrs["start_time"].isoformat(),
            "platform": ref.attrs["platform_name"],
            "source_file": nat_path.name,
            "area_id": area.area_id,
            "area_extent": list(area.area_extent),
            "crs_wkt": area.crs.to_wkt(),  # full projection description (a PROJ string is lossy)
            "region_bbox": list(bbox),
        },
    )


def save_scene(ds: xr.Dataset, out_path: Path) -> None:
    """Write compressed NetCDF via a temporary name, so half-written files never look finished."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".nc.part")
    with _nc_lock:
        ds.to_netcdf(tmp_path, encoding={"bt": {"zlib": True, "complevel": 4}})
    tmp_path.replace(out_path)


def check_grid(ds: xr.Dataset, out_dir: Path) -> None:
    """Save the pixel grid of the first scene; assert every later scene has exactly the same grid.

    This is what makes the scenes stackable into one (time, channel, y, x) dataset later.
    """
    grid_path = out_dir / GRID_FILE
    with _nc_lock:  # also stops two workers from both writing the first grid file
        if not grid_path.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            xr.Dataset(
                coords={"y": ds["y"], "x": ds["x"]},
                attrs={k: ds.attrs[k] for k in ("area_id", "area_extent", "crs_wkt", "region_bbox")},
            ).to_netcdf(grid_path)
            return
        with xr.open_dataset(grid_path) as grid:
            same = (
                grid.sizes["y"] == ds.sizes["y"]
                and grid.sizes["x"] == ds.sizes["x"]
                and np.allclose(grid.attrs["area_extent"], ds.attrs["area_extent"])
            )
    if not same:
        raise ValueError(f"Grid of {ds.attrs['source_file']} differs from {grid_path}")
