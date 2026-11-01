import numpy as np
import pytest

from fogseg.features import (
    CHANNELS,
    FEATURE_GROUPS,
    StaticFields,
    feature_names,
    hours_after_21,
    night_feature_stack,
    satellite_zenith,
)
from fogseg.labels import FOG, IGNORE, emissivity_39

ALL_GROUPS = list(FEATURE_GROUPS)


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
