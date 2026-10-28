"""Tiny synthetic scenes, so tests need no internet, credentials or real data."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from fogseg.build import variable_length_strings
from fogseg.dynamic_threshold import DynamicParams
from fogseg.labels import LabelParams

# The real SEVIRI geostationary projection and our Central Europe crop, but with 4 x 6 big pixels
GEOS = CRS.from_proj4("+proj=geos +h=35785831 +a=6378169 +rf=295.488065897014 +lon_0=0 +units=m +no_defs")
AREA_EXTENT = [121516.0, 4082048.75, 1507702.0, 4805146.0]  # x_min, y_min, x_max, y_max in metres
NY, NX = 4, 6
CHANNELS = ["IR_039", "IR_108", "IR_120"]
TIMES = [dt.datetime(2023, 11, 21, 0), dt.datetime(2023, 11, 20, 21), dt.datetime(2023, 11, 21, 3)]

# Explicit threshold values for the tests; they do not follow configs/default.yaml
TEST_PARAMS = LabelParams(
    night_min_sza=95.0,
    btd_fixed=2.5,
    warm_min_bt108=260.0,
    cirrus_max_split=2.0,
    smooth_max_std=1.5,
    uncertainty_band=0.5,
    min_blob_pixels=5,
)
TEST_DYNAMIC = DynamicParams(
    bin_width=0.25,
    bin_range=(-2.0, 8.0),
    smoothing_bins=3,
    clear_window=(1.5, 4.75),
    offset=1.5,
    min_pixels=2000,
    clamp=(3.5, 6.0),
)

LABEL_NY, LABEL_NX = 8, 10
DEEP_NIGHT_SZA = 120.0
LABEL_DATASET_CREATED = "2026-01-01T00:00:00+00:00"
GRID_LON0, GRID_LAT0 = 10.0, 50.0
COLD_PIXEL = (7, 8)


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


def label_scenes(
    n_scenes: int,
    scene_sza: dict[int, float] | None,
    background_btd: float,
) -> xr.Dataset:
    """The scenes of `make_label_dataset`, before they are written."""
    times = np.datetime64("2023-11-20T21:00", "ns") + np.arange(n_scenes) * np.timedelta64(3, "h")

    bt = np.empty((n_scenes, len(CHANNELS), LABEL_NY, LABEL_NX), dtype="float32")
    bt[:, CHANNELS.index("IR_108")] = 270.0
    bt[:, CHANNELS.index("IR_120")] = 269.5
    bt[:, CHANNELS.index("IR_039")] = 270.0 - background_btd
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
    return dataset


def make_label_dataset(
    path: Path,
    n_scenes: int = 3,
    scene_sza: dict[int, float] | None = None,
    background_btd: float = 1.8,
) -> Path:
    """A tiny dataset in the layout `fogseg build` writes, for the label tests.

    Each scene has a warm clear background (BTD `background_btd` K) and a 5 x 5 fog patch
    (BTD 5.0 K) in the corner, all with a flat IR_108 field. `scene_sza` maps a scene index to
    its solar zenith angle; every other scene is in deep night.
    """
    dataset = label_scenes(n_scenes, scene_sza, background_btd)
    variable_length_strings(dataset).to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return path


def add_station_grid(dataset: xr.Dataset) -> xr.Dataset:
    """`lon` / `lat` on (y, x): the grid starts at `GRID_LON0`, `GRID_LAT0` with 0.1 degree spacing, north at the top."""
    lon, lat = np.meshgrid(GRID_LON0 + 0.1 * np.arange(LABEL_NX), GRID_LAT0 - 0.1 * np.arange(LABEL_NY))
    return dataset.assign_coords(
        lon=(("y", "x"), lon.astype("float32")),
        lat=(("y", "x"), lat.astype("float32")),
    )


def make_station_dataset(path: Path, n_scenes: int = 3) -> Path:
    """Like `make_label_dataset`, with `lon` / `lat` on (y, x) and one cold pixel at `COLD_PIXEL`.

    The cold pixel (IR_108 230 K) is high cloud: its label is 0 and a station on it is hidden.
    """
    dataset = label_scenes(n_scenes=n_scenes, scene_sza=None, background_btd=1.8)
    dataset["bt"].loc[{"channel": "IR_108", "y": float(COLD_PIXEL[0]), "x": float(COLD_PIXEL[1])}] = 230.0

    variable_length_strings(add_station_grid(dataset)).to_zarr(
        path, mode="w", zarr_format=3, consolidated=False
    )
    return path


# A background of 2.0 K fills one 0.25 K bin (bins from -2 K); the smoothing makes the bin and its two
# neighbours equal, and the peak search takes the first of them, the bin centred at 1.875 K.
PLANTED_PEAK = 1.875
PLANTED_PATCH_BTD = 5.0
PLANTED_STRIP_BTD = 3.6
PLANTED_FOG_PIXEL, PLANTED_STRIP_PIXEL, PLANTED_CLEAR_PIXEL = (2, 2), (6, 1), (3, 8)


def make_planted_dataset(path: Path, n_scenes: int = 3) -> Path:
    """Like `make_station_dataset`, built so that only some thresholds label it right.

    A clear background (BTD 2.0 K, peak bin `PLANTED_PEAK`), the 5 x 5 fog patch (BTD `PLANTED_PATCH_BTD`)
    and a 2 x 4 strip at an in-between BTD (`PLANTED_STRIP_BTD`) with a clear row between them. IR_108
    is flat, so the smoothness test never interferes.
    """
    dataset = label_scenes(n_scenes=n_scenes, scene_sza=None, background_btd=2.0)
    strip = {"channel": "IR_039", "y": slice(6.0, 7.0), "x": slice(0.0, 3.0)}
    dataset["bt"].loc[strip] = 270.0 - PLANTED_STRIP_BTD

    variable_length_strings(add_station_grid(dataset)).to_zarr(
        path, mode="w", zarr_format=3, consolidated=False
    )
    return path
