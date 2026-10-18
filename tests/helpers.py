"""Tiny synthetic scenes, so tests need no internet, credentials or real data."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from fogseg.dynamic_threshold import DynamicParams
from fogseg.labels import LabelParams

# The real SEVIRI geostationary projection and our Central Europe crop, but with 4 x 6 big pixels
GEOS = CRS.from_proj4("+proj=geos +h=35785831 +a=6378169 +rf=295.488065897014 +lon_0=0 +units=m +no_defs")
AREA_EXTENT = [121516.0, 4082048.75, 1507702.0, 4805146.0]  # x_min, y_min, x_max, y_max in metres
NY, NX = 4, 6
CHANNELS = ["IR_039", "IR_108", "IR_120"]
TIMES = [dt.datetime(2023, 11, 21, 0), dt.datetime(2023, 11, 20, 21), dt.datetime(2023, 11, 21, 3)]

# The provisional threshold values of the spec (the same as configs/default.yaml)
PROVISIONAL = LabelParams(
    night_min_sza=95.0,
    btd_fixed=2.5,
    warm_min_bt108=260.0,
    cirrus_max_split=2.0,
    smooth_max_std=1.5,
    uncertainty_band=0.5,
    min_blob_pixels=5,
)
PROVISIONAL_DYNAMIC = DynamicParams(
    bin_width=0.25,
    bin_range=(-2.0, 8.0),
    smoothing_bins=3,
    clear_window=(0.5, 3.0),
    cloud_window=(3.0, 8.0),
    min_pixels=2000,
    min_hump_share=0.05,
    max_valley_ratio=0.5,
    clamp=(1.5, 5.0),
)

LABEL_NY, LABEL_NX = 8, 10
DEEP_NIGHT_SZA = 120.0
LABEL_DATASET_CREATED = "2026-01-01T00:00:00+00:00"


def make_scene(time: dt.datetime, value: float, x=None) -> xr.Dataset:
    """A fake cropped scene in exactly the format `fogseg fetch` writes. Every pixel = `value` K."""
    x0, y0, x1, y1 = AREA_EXTENT
    dx = (x1 - x0) / NX
    dy = (y1 - y0) / NY
    xs = x0 + dx * (np.arange(NX) + 0.5) if x is None else x
    ys = y1 - dy * (np.arange(NY) + 0.5)  # north at the top, like upper_right_corner="NE"
    bt = np.full((len(CHANNELS), NY, NX), value, dtype="float32")
    return xr.Dataset(
        {"bt": (("channel", "y", "x"), bt, {"units": "K"})},
        coords={"channel": CHANNELS, "y": ys, "x": xs},
        attrs={
            "time": time.isoformat(),
            "platform": "Meteosat-10",
            "source_file": f"fake_{time:%Y%m%dT%H%M}.nat",
            "area_id": "msg_seviri_fes_3km",
            "area_extent": AREA_EXTENT,
            "crs_wkt": GEOS.to_wkt(),
            "region_bbox": [2, 43, 20, 55],
        },
    )


def make_label_dataset(path: Path, n_scenes: int = 3, scene_sza: dict[int, float] | None = None) -> Path:
    """A tiny dataset in the layout `fogseg build` writes, for the label tests.

    Each scene has a warm clear background (BTD 1.8 K) and a 5 x 5 fog patch (BTD 5.0 K) in the
    corner, all with a flat IR_108 field. `scene_sza` maps a scene index to its solar zenith
    angle; every other scene is in deep night.
    """
    times = np.datetime64("2023-11-20T21:00", "ns") + np.arange(n_scenes) * np.timedelta64(3, "h")

    bt = np.empty((n_scenes, len(CHANNELS), LABEL_NY, LABEL_NX), dtype="float32")
    bt[:, CHANNELS.index("IR_108")] = 270.0
    bt[:, CHANNELS.index("IR_120")] = 269.5
    bt[:, CHANNELS.index("IR_039")] = 270.0 - 1.8
    bt[:, CHANNELS.index("IR_039"), :5, :5] = 270.0 - 5.0

    sza = np.full((n_scenes, LABEL_NY, LABEL_NX), DEEP_NIGHT_SZA, dtype="float32")
    for scene, angle in (scene_sza or {}).items():
        sza[scene] = angle

    dataset = xr.Dataset(
        {
            "bt": (("time", "channel", "y", "x"), bt),
            "sza": (("time", "y", "x"), sza),
        },
        coords={
            "time": times,
            "channel": CHANNELS,
            "y": np.arange(LABEL_NY, dtype="float64"),
            "x": np.arange(LABEL_NX, dtype="float64"),
        },
        attrs={"created": LABEL_DATASET_CREATED},
    )
    dataset.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return path
