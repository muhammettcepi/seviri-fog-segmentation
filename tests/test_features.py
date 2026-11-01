import numpy as np
import pandas as pd
import pytest
import xarray as xr
from helpers import TEST_PARAMS, add_station_grid
from test_experiment import tiny_season

from fogseg.features import (
    CHANNELS,
    FEATURE_GROUPS,
    StaticFields,
    case_features,
    feature_names,
    hours_after_21,
    night_feature_stack,
    night_features,
    open_era5,
    read_static,
    satellite_zenith,
)
from fogseg.label_store import LABEL_CHANNELS
from fogseg.labels import FOG, IGNORE, emissivity_39
from fogseg.validation import night_of

ALL_GROUPS = list(FEATURE_GROUPS)
SCENE_OF_CASE_1, SCENE_OF_CASE_2 = 3, 1
IR_108 = 270.0  # the flat IR_108 of the scenes of `tiny_season`


def night_inputs():
    """Two scenes of 4 x 4 pixels with a different value in every channel, scene and (row, column)."""
    rng = np.random.default_rng(0)
    channel_offset = np.arange(len(CHANNELS)).reshape(1, -1, 1, 1) * 1.5
    scene_offset = np.arange(2).reshape(-1, 1, 1, 1) * 0.7
    bt = 270.0 + channel_offset + scene_offset + rng.normal(0.0, 0.3, (2, len(CHANNELS), 4, 4))
    bt[:, CHANNELS.index("IR_039")] -= np.array([0.0, 1.0]).reshape(-1, 1, 1)
    times = np.array(["2023-11-20T21", "2023-11-21T00"], dtype="datetime64[ns]")
    static = StaticFields(
        latitude=np.linspace(45.0, 50.0, 16).reshape(4, 4),
        satellite_zenith=np.linspace(50.0, 60.0, 16).reshape(4, 4),
        elevation_mean=np.linspace(100.0, 400.0, 16).reshape(4, 4),
        elevation_std=np.linspace(5.0, 80.0, 16).reshape(4, 4),
    )
    era5 = {
        "t2m": 273.0 + rng.normal(0.0, 1.0, (2, 4, 4)),
        "skt": 272.0 + rng.normal(0.0, 1.0, (2, 4, 4)),
        "d2m": 271.0 + rng.normal(0.0, 1.0, (2, 4, 4)),
        "sd": rng.uniform(0.0, 0.2, (2, 4, 4)),
    }
    rule = np.zeros((2, 4, 4), dtype=np.uint8)
    rule[0, 1, 2] = FOG
    rule[1, 1, 2] = IGNORE
    return bt, times, static, era5, rule


def test_feature_names_follow_the_groups():
    assert len(feature_names(ALL_GROUPS)) == 29
    assert feature_names(["rule", "bt"]) == [f"bt_{c}" for c in CHANNELS] + ["rule_fog", "rule_ignore"]
    with pytest.raises(ValueError, match="era5"):
        feature_names(["radar"])


def test_satellite_zenith_by_hand():
    assert satellite_zenith(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert satellite_zenith(np.array([0.0]), np.array([50.0]))[0] == pytest.approx(57.3, abs=0.2)


def test_hours_after_21():
    times = np.array(
        ["2023-11-20T21", "2023-11-21T00", "2023-11-21T03", "2023-11-21T05"], dtype="datetime64[ns]"
    )
    assert hours_after_21(times).tolist() == [0, 3, 6, 8]


def test_stack_values_at_one_pixel():
    bt, times, static, era5, rule = night_inputs()
    stack = night_feature_stack(bt, times, static, era5, rule, ALL_GROUPS)
    at = dict(zip(feature_names(ALL_GROUPS), stack[:, :, 1, 2].T))
    ir039, ir108 = bt[:, CHANNELS.index("IR_039"), 1, 2], bt[:, CHANNELS.index("IR_108"), 1, 2]
    assert stack.dtype == np.float32
    assert stack.shape == (2, 29, 4, 4)
    assert np.allclose(at["d108_039"], ir108 - ir039)
    assert np.allclose(at["emissivity_39"], emissivity_39(ir039, ir108))
    assert np.allclose(at["night_max_btd"], (ir108 - ir039).max())
    assert np.isnan(at["ir108_change"][0]) and at["ir108_change"][1] == pytest.approx(ir108[1] - ir108[0])
    assert at["rule_fog"].tolist() == (rule[:, 1, 2] == FOG).tolist()
    assert at["rule_ignore"].tolist() == (rule[:, 1, 2] == IGNORE).tolist()
    assert np.allclose(at["t2m_minus_ir108"], era5["t2m"][:, 1, 2] - ir108)
    assert np.allclose(at["t2m_minus_d2m"], era5["t2m"][:, 1, 2] - era5["d2m"][:, 1, 2])
    assert np.allclose(at["snow_depth"], era5["sd"][:, 1, 2])
    assert np.allclose(at["hours_after_21"], [0.0, 3.0])
    assert np.allclose(at["elevation_mean"], static.elevation_mean[1, 2])


def test_texture_features_are_the_local_std_of_the_scene():
    bt, times, static, era5, rule = night_inputs()
    stack = night_feature_stack(bt, times, static, era5, rule, ["texture"])
    assert stack.shape == (2, 2, 4, 4)
    ir108 = bt[0, CHANNELS.index("IR_108")]
    centre = ir108[0:3, 1:4]
    assert stack[0, 0, 1, 2] == pytest.approx(centre.std(), rel=1e-4)


def test_era5_group_needs_era5():
    bt, times, static, _, rule = night_inputs()
    with pytest.raises(ValueError, match="fogseg era5"):
        night_feature_stack(bt, times, static, None, rule, ["bt", "era5"])


def test_terrain_group_needs_elevation():
    bt, times, static, era5, rule = night_inputs()
    flat = StaticFields(static.latitude, static.satellite_zenith, None, None)
    with pytest.raises(ValueError, match="terrain"):
        night_feature_stack(bt, times, flat, era5, rule, ["terrain"])


def test_groups_not_asked_for_are_not_needed():
    bt, times, static, _, rule = night_inputs()
    flat = StaticFields(static.latitude, static.satellite_zenith, None, None)
    stack = night_feature_stack(bt, times, flat, None, rule, ["bt", "geometry", "rule"])
    assert stack.shape == (2, 12, 4, 4)


def test_nan_channel_gives_nan_features_without_warnings(recwarn):
    bt, times, static, era5, rule = night_inputs()
    bt[0, CHANNELS.index("IR_108"), 1, 2] = np.nan
    stack = night_feature_stack(bt, times, static, era5, rule, ALL_GROUPS)
    assert np.isnan(stack[0, feature_names(ALL_GROUPS).index("d108_039"), 1, 2])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_pixel_missing_in_every_scene_has_no_night_maximum_and_no_warning(recwarn):
    bt, times, static, era5, rule = night_inputs()
    bt[:, CHANNELS.index("IR_108"), 1, 2] = np.nan
    stack = night_feature_stack(bt, times, static, era5, rule, ["time"])
    assert np.isnan(stack[:, 1, 1, 2]).all()
    assert not np.isnan(stack[:, 1, 0, 0]).any()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def tiny_teacher_season(tmp_path) -> tuple[xr.Dataset, pd.DataFrame, StaticFields]:
    """`tiny_season` with all eight channels and a lon / lat grid, and its `read_static` fields."""
    season, cases = tiny_season(tmp_path)
    bt = season["bt"].load().reindex(channel=CHANNELS)
    for index, channel in enumerate(CHANNELS):
        if channel not in LABEL_CHANNELS:
            bt.loc[{"channel": channel}] = 265.0 + index
    data = add_station_grid(xr.Dataset({"bt": bt, "sza": season["sza"].load()}))
    return data, cases, read_static(data, tmp_path / "no_dem.zarr", ["bt"])


def era5_store(
    data: xr.Dataset, without: np.datetime64 | None = None, nan_at: np.datetime64 | None = None
) -> xr.Dataset:
    """An ERA5 store with every scene of `data` but `without`, all NaN at `nan_at` like a missing hour.

    Every field is flat within a scene and grows with the scene's position in `data`.
    """
    times = data["time"].values
    kept = np.flatnonzero(times != without)
    shape = (len(kept), data.sizes["y"], data.sizes["x"])
    fields = {"t2m": 275.0, "skt": 273.0, "d2m": 272.0, "sd": 0.1}
    values = {
        name: np.broadcast_to((base + kept)[:, None, None], shape).astype("float32")
        for name, base in fields.items()
    }
    for field in values.values():
        field[times[kept] == nan_at] = np.nan
    return xr.Dataset(
        {name: (("time", "y", "x"), field) for name, field in values.items()},
        coords={"time": times[kept]},
    )


def test_case_features_match_the_night_stack(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    groups = ["bt", "differences", "time", "rule"]
    table = case_features(data, cases, static, None, TEST_PARAMS, groups)
    night = np.flatnonzero(night_of(data["time"].values) == cases.loc[1, "night"])
    stack = night_features(data, night, static, None, TEST_PARAMS, groups).stack
    slot = list(data["time"].values[night]).index(cases.loc[1, "scene_time"])
    expected = stack[slot, :, cases.loc[1, "iy"], cases.loc[1, "ix"]]
    assert table.columns.tolist() == feature_names(groups) + ["has_era5", "trusted"]
    assert table.index.tolist() == cases.index.tolist()
    assert np.allclose(table.loc[1, feature_names(groups)].to_numpy(float), expected, equal_nan=True)


def test_night_features_carry_trust_and_rule_labels_without_the_rule_group(tmp_path):
    data, _, static = tiny_teacher_season(tmp_path)
    night = np.array([0, 1])
    result = night_features(data, night, static, None, TEST_PARAMS, ["bt"])
    assert result.stack.shape == (2, len(CHANNELS), 8, 10)
    assert result.has_era5.tolist() == [True, True]
    assert result.trusted.shape == result.rule_labels.shape == (2, 8, 10)
    assert result.trusted.all()
    assert result.rule_labels[0, 2, 2] == FOG and result.rule_labels[0, 7, 9] != FOG


def test_scene_missing_from_era5_gives_nan_and_no_era5(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    era5 = era5_store(data, without=cases.loc[0, "scene_time"])
    table = case_features(data, cases, static, era5, TEST_PARAMS, ["bt", "era5"])
    assert not table.loc[0, "has_era5"] and np.isnan(table.loc[0, "t2m_minus_ir108"])
    assert table.loc[1, "has_era5"] and table.loc[2, "has_era5"]
    assert table.loc[1, "t2m_minus_ir108"] == pytest.approx(275.0 + SCENE_OF_CASE_1 - IR_108)
    assert table.loc[2, "t2m_minus_ir108"] == pytest.approx(275.0 + SCENE_OF_CASE_2 - IR_108)
    assert table.loc[1, "snow_depth"] == pytest.approx(0.1 + SCENE_OF_CASE_1)


def test_scene_with_nan_filled_era5_fields_has_no_era5(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    era5 = era5_store(data, nan_at=cases.loc[0, "scene_time"])
    table = case_features(data, cases, static, era5, TEST_PARAMS, ["bt", "era5"])
    era5_columns = FEATURE_GROUPS["era5"]
    assert table["has_era5"].tolist() == [False, True, True]
    assert table.loc[0, era5_columns].isna().all()
    assert table.loc[2, "t2m_minus_ir108"] == pytest.approx(275.0 + SCENE_OF_CASE_2 - IR_108)
    assert table.loc[2, "t2m_minus_d2m"] == pytest.approx(3.0)


def test_scene_with_one_empty_era5_variable_has_no_era5(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    era5 = era5_store(data)
    era5["sd"].loc[{"time": cases.loc[0, "scene_time"]}] = np.nan
    table = case_features(data, cases, static, era5, TEST_PARAMS, ["era5"])
    assert table["has_era5"].tolist() == [False, True, True]
    assert table.loc[0, FEATURE_GROUPS["era5"]].isna().all()


def test_case_pixel_without_ir108_is_untrusted_and_has_nan_differences(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    scene = list(data["time"].values).index(cases.loc[0, "scene_time"])
    data["bt"][scene, CHANNELS.index("IR_108"), cases.loc[0, "iy"], cases.loc[0, "ix"]] = np.nan
    table = case_features(data, cases, static, None, TEST_PARAMS, ["differences"])
    assert table["trusted"].tolist() == [False, True, True]
    assert np.isnan(table.loc[0, "d108_039"]) and np.isfinite(table.loc[1, "d108_039"])


def test_read_static_takes_the_grid_geometry_and_the_dem_when_asked(tmp_path):
    data, _, _ = tiny_teacher_season(tmp_path)
    elevation = np.arange(80, dtype="float32").reshape(8, 10)
    dem = xr.Dataset(
        {"elevation_mean": (("y", "x"), elevation), "elevation_std": (("y", "x"), elevation / 2)}
    )
    dem.to_zarr(tmp_path / "dem.zarr", zarr_format=3, consolidated=False)
    flat = read_static(data, tmp_path / "dem.zarr", ["geometry"])
    hilly = read_static(data, tmp_path / "dem.zarr", ["terrain"])
    assert flat.elevation_mean is None and flat.elevation_std is None
    assert np.array_equal(flat.latitude, data["lat"].values)
    assert flat.satellite_zenith.shape == (8, 10)
    assert np.array_equal(hilly.elevation_mean, elevation)
    assert np.array_equal(hilly.elevation_std, elevation / 2)


def test_missing_stores_name_their_command(tmp_path):
    data, _, _ = tiny_teacher_season(tmp_path)
    with pytest.raises(FileNotFoundError, match="fogseg dem"):
        read_static(data, tmp_path / "no_dem.zarr", ["terrain"])
    with pytest.raises(FileNotFoundError, match="fogseg era5"):
        open_era5(tmp_path / "no_era5.zarr", ["era5"])
    assert open_era5(tmp_path / "no_era5.zarr", ["bt"]) is None


def test_case_of_a_missing_scene_is_an_error(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    cases.loc[0, "scene_time"] = pd.Timestamp("2030-01-01T21:00")
    with pytest.raises(ValueError, match="not in the dataset"):
        case_features(data, cases, static, None, TEST_PARAMS, ["bt"])
