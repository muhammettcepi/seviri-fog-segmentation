"""Tests for stacking per-scene files into the Zarr dataset."""

import numpy as np
import pytest
import xarray as xr
from conftest import CHANNELS, NX, NY, TIMES, make_scene

from fogseg.build import build_zarr
from fogseg.preprocess import save_scene


def test_build_stacks_all_scenes_in_time_order(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    n = build_zarr(interim_dir, out, CHANNELS, batch=2)  # batch=2 also exercises the append path

    ds = xr.open_zarr(out, consolidated=False)
    assert n == 3
    assert dict(ds["bt"].sizes) == {"time": 3, "channel": len(CHANNELS), "y": NY, "x": NX}
    assert list(ds["time"].values) == sorted(np.datetime64(t, "ns") for t in TIMES)
    # files were written with values 270, 271, 272 in TIMES order; check each landed at its time
    for i, t in enumerate(TIMES):
        assert float(ds["bt"].sel(time=np.datetime64(t, "ns")).mean()) == pytest.approx(270.0 + i)
    assert list(ds["platform"].values) == ["Meteosat-10"] * 3
    assert ds["source_file"].values[0] == "fake_20231120T2100.nat"


def test_build_adds_metadata_lonlat_and_sun_angle(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    ds = xr.open_zarr(out, consolidated=False)

    assert list(ds["wavelength"].values) == [3.9, 10.8, 12.0]
    assert ds["bt"].attrs["units"] == "K"
    assert ds["lat"].shape == (NY, NX) and 40 < float(ds["lat"].min()) < float(ds["lat"].max()) < 60
    assert float(ds["lat"][0, 0]) > float(ds["lat"][-1, 0])  # row 0 is the northern edge
    # November nights over Central Europe: the sun is far below the horizon in every pixel
    assert float(ds["sza"].min()) > 100
    assert "EO:EUM:DAT:MSG:HRSEVIRI" in ds.attrs["source"]


def test_metadata_survives_appending_in_batches(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS, batch=1)  # every scene after the first is an append
    ds = xr.open_zarr(out, consolidated=False)
    assert list(ds["time"].values) == sorted(np.datetime64(t, "ns") for t in TIMES)
    attrs = ds.attrs
    assert "EO:EUM:DAT:MSG:HRSEVIRI" in attrs["source"]
    assert attrs["crs_wkt"].startswith("PROJCRS") and len(attrs["area_extent"]) == 4


def test_impossible_temperatures_become_nan(interim_dir, tmp_path):
    # 3.9 µm over very cold cloud tops: signal below noise -> calibration gives ~0 K
    noisy = make_scene(TIMES[0].replace(hour=5), 270.0)
    noisy["bt"][0, 1, 2] = -2.9
    save_scene(noisy, interim_dir / "seviri_ceu_20231121T0500.nc")
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)

    ds = xr.open_zarr(out, consolidated=False)
    assert int(ds["bt"].isnull().sum()) == 1
    assert np.isnan(float(ds["bt"].sel(channel="IR_039").isel(time=-1, y=1, x=2)))
    assert ds.attrs["invalid_values"].startswith("1 ")


def test_build_refuses_to_overwrite_unless_asked(interim_dir, tmp_path):
    out = tmp_path / "data.zarr"
    build_zarr(interim_dir, out, CHANNELS)
    with pytest.raises(FileExistsError):
        build_zarr(interim_dir, out, CHANNELS)
    build_zarr(interim_dir, out, CHANNELS, overwrite=True)  # explicit rebuild works


def test_build_rejects_scene_on_a_different_grid(interim_dir, tmp_path):
    shifted = make_scene(TIMES[0].replace(hour=5), 280.0, x=np.arange(NX) * 1000.0)
    save_scene(shifted, interim_dir / "seviri_ceu_20231121T0500.nc")
    with pytest.raises(ValueError, match="reference grid"):
        build_zarr(interim_dir, tmp_path / "data.zarr", CHANNELS)
    assert not (tmp_path / "data.zarr").exists()  # nothing half-finished left behind
