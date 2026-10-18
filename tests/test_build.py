"""Tests for stacking per-scene files into the Zarr dataset."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr
from helpers import CHANNELS, NX, NY, TIMES, make_scene
from pyorbital.astronomy import sun_zenith_angle

from fogseg.build import REGION_SCAN_DELAY, build_zarr
from fogseg.config import scene_filename
from fogseg.preprocess import save_scene

LATE_SLOT = dt.datetime(2023, 11, 21, 5)


def open_store(path):
    return xr.open_zarr(path, consolidated=False)


def test_build_stacks_all_scenes_in_time_order(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    n = build_zarr(interim_dir, out, CHANNELS, batch=2)  # batch=2 also exercises the append path

    ds = open_store(out)
    assert n == 3
    assert dict(ds["bt"].sizes) == {"time": 3, "channel": len(CHANNELS), "y": NY, "x": NX}
    assert list(ds["time"].values) == sorted(np.datetime64(t, "ns") for t in TIMES)
    # files were written with values 270, 271, 272 in TIMES order; check each landed at its time
    for i, t in enumerate(TIMES):
        assert float(ds["bt"].sel(time=np.datetime64(t, "ns")).mean()) == pytest.approx(270.0 + i)
    assert list(ds["platform"].values) == ["Meteosat-10"] * 3
    assert ds["source_file"].values[0] == "fake_20231120T2100.nat"


def test_build_adds_metadata_and_lonlat(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    ds = open_store(out)

    assert list(ds["wavelength"].values) == [3.9, 10.8, 12.0]
    assert ds["bt"].attrs["units"] == "K"
    assert ds["lat"].shape == (NY, NX)
    assert 40 < float(ds["lat"].min()) < float(ds["lat"].max()) < 60
    assert float(ds["lat"][0, 0]) > float(ds["lat"][-1, 0])  # row 0 is the northern edge
    assert "EO:EUM:DAT:MSG:HRSEVIRI" in ds.attrs["source"]


def test_sun_angle_is_computed_at_the_regional_scan_time(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    ds = open_store(out)

    scene = ds.sel(time=np.datetime64(TIMES[1], "ns"))
    lon, lat = float(scene["lon"][1, 2]), float(scene["lat"][1, 2])
    expected = sun_zenith_angle(TIMES[1] + REGION_SCAN_DELAY, lon, lat)
    assert float(scene["sza"][1, 2]) == pytest.approx(expected, abs=0.01)


def test_sun_angle_grows_towards_the_east_after_sunset(interim_dir, tmp_path):
    # At 21:00 UTC it is later in the solar day in the east, so the sun is further below the
    # horizon there. Swapped lat/lon or a mirrored grid would break this.
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    sza = open_store(out)["sza"].sel(time=np.datetime64(TIMES[1], "ns"))
    assert float(sza[1, -1]) > float(sza[1, 0]) + 5


def test_metadata_survives_appending_in_batches(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS, batch=1)  # every scene after the first is an append

    ds = open_store(out)
    assert list(ds["time"].values) == sorted(np.datetime64(t, "ns") for t in TIMES)
    assert "EO:EUM:DAT:MSG:HRSEVIRI" in ds.attrs["source"]
    assert ds.attrs["crs_wkt"].startswith("PROJCRS")
    assert len(ds.attrs["area_extent"]) == 4


def test_impossible_temperatures_become_nan_and_are_counted(interim_dir, tmp_path):
    # 3.9 µm over very cold cloud tops: signal below noise -> calibration gives ~0 K
    noisy = make_scene(LATE_SLOT, 270.0)
    noisy["bt"][0, 1, 2] = -2.9
    noisy["bt"][1, 0, 0] = np.nan  # already missing in the input: not an "invalid value"
    save_scene(noisy, interim_dir / scene_filename(LATE_SLOT))
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)

    ds = open_store(out)
    assert np.isnan(float(ds["bt"].sel(channel="IR_039").isel(time=-1, y=1, x=2)))
    assert int(ds["bt"].isnull().sum()) == 2
    assert ds.attrs["invalid_values"].startswith("1 ")


def test_build_refuses_to_overwrite_unless_asked(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    with pytest.raises(FileExistsError):
        build_zarr(interim_dir, out, CHANNELS)
    build_zarr(interim_dir, out, CHANNELS, overwrite=True)


def test_failed_build_leaves_nothing_behind(interim_dir, tmp_path):
    shifted = make_scene(LATE_SLOT, 280.0, x=np.arange(NX) * 1000.0)
    save_scene(shifted, interim_dir / scene_filename(LATE_SLOT))
    out = tmp_path / "data.zarr"

    # batch=1: the first scenes are already written when the shifted scene fails
    with pytest.raises(ValueError, match="reference grid"):
        build_zarr(interim_dir, out, CHANNELS, batch=1)
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


def test_build_rejects_unexpected_channels(interim_dir, tmp_path):
    with pytest.raises(ValueError, match="channels"):
        build_zarr(interim_dir, tmp_path / "data.zarr", ["IR_039", "IR_108", "IR_134"])


def test_build_rejects_duplicate_scene_times(interim_dir, tmp_path):
    # a file whose name says 05:00 but whose content says 21:00 (already present)
    duplicate = make_scene(TIMES[1], 280.0)
    save_scene(duplicate, interim_dir / scene_filename(LATE_SLOT))
    with pytest.raises(ValueError, match="unique and increasing"):
        build_zarr(interim_dir, tmp_path / "data.zarr", CHANNELS)
