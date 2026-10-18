"""Shared test fixtures: tiny synthetic scenes, so tests need no internet, credentials or real data."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from fogseg.preprocess import check_grid, save_scene

# The real SEVIRI geostationary projection and our Central Europe crop, but with 4 x 6 big pixels
GEOS = CRS.from_proj4("+proj=geos +h=35785831 +a=6378169 +rf=295.488065897014 +lon_0=0 +units=m +no_defs")
AREA_EXTENT = [121516.0, 4082048.75, 1507702.0, 4805146.0]  # x_min, y_min, x_max, y_max in metres
NY, NX = 4, 6
CHANNELS = ["IR_039", "IR_108", "IR_120"]
TIMES = [dt.datetime(2023, 11, 21, 0), dt.datetime(2023, 11, 20, 21), dt.datetime(2023, 11, 21, 3)]


def make_scene(time: dt.datetime, value: float, x=None) -> xr.Dataset:
    """A fake cropped scene in exactly the format `fogseg fetch` writes. Every pixel = `value` K."""
    x0, y0, x1, y1 = AREA_EXTENT
    dx, dy = (x1 - x0) / NX, (y1 - y0) / NY
    xs = x0 + dx * (np.arange(NX) + 0.5) if x is None else x
    ys = y1 - dy * (np.arange(NY) + 0.5)  # north at the top, like upper_right_corner="NE"
    bt = np.full((len(CHANNELS), NY, NX), value, dtype="float32")
    return xr.Dataset(
        {"bt": (("channel", "y", "x"), bt, {"units": "K"})},
        coords={"channel": CHANNELS, "y": ys, "x": xs},
        attrs={"time": time.isoformat(), "platform": "Meteosat-10",
               "source_file": f"fake_{time:%Y%m%dT%H%M}.nat", "area_id": "msg_seviri_fes_3km",
               "area_extent": AREA_EXTENT, "crs_wkt": GEOS.to_wkt(), "region_bbox": [2, 43, 20, 55]},
    )


@pytest.fixture
def interim_dir(tmp_path: Path) -> Path:
    """A folder with grid.nc and three fake scenes (written out of time order on purpose)."""
    folder = tmp_path / "interim"
    for i, time in enumerate(TIMES):
        ds = make_scene(time, value=270.0 + i)
        check_grid(ds, folder)
        save_scene(ds, folder / f"seviri_ceu_{time:%Y%m%dT%H%M}.nc")
    return folder
