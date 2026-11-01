"""Features of the teacher: a stack per night, read at station pixels to train and everywhere to label."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.dem import read_dem
from fogseg.era5 import ERA5_VARIABLES
from fogseg.experiment import baseline_threshold
from fogseg.labels import FOG, IGNORE, LabelParams, emissivity_39, label_night, local_std, trusted_mask
from fogseg.parallax import EARTH_RADIUS_KM, SATELLITE_RADIUS_KM, unit_vector
from fogseg.validation import night_of

CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "bt": [f"bt_{channel}" for channel in CHANNELS],
    "differences": ["d108_039", "d108_120", "d087_108", "d120_134", "d108_062", "emissivity_39"],
    "texture": ["std3_ir108", "std3_btd"],
    "geometry": ["satellite_zenith", "latitude"],
    "terrain": ["elevation_mean", "elevation_std"],
    "era5": ["t2m_minus_ir108", "skt_minus_ir108", "t2m_minus_d2m", "snow_depth"],
    "time": ["hours_after_21", "night_max_btd", "ir108_change"],
    "rule": ["rule_fog", "rule_ignore"],
}


@dataclass(frozen=True)
class StaticFields:
    """2-D fields that stay the same from scene to scene; the elevation is None when `terrain` is unused."""

    latitude: np.ndarray
    satellite_zenith: np.ndarray
    elevation_mean: np.ndarray | None
    elevation_std: np.ndarray | None


def feature_names(groups: list[str]) -> list[str]:
    """The feature names of `groups`, in the order of `FEATURE_GROUPS`."""
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group {unknown[0]!r}; allowed groups are {', '.join(FEATURE_GROUPS)}"
        )
    return [name for group, names in FEATURE_GROUPS.items() if group in groups for name in names]


def satellite_zenith(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Angle in degrees between the local vertical and the direction to a satellite over 0° E."""
    vertical = unit_vector(lon, lat)
    to_satellite = np.array([SATELLITE_RADIUS_KM, 0.0, 0.0]) - EARTH_RADIUS_KM * vertical
    cosine = np.sum(vertical * to_satellite, axis=-1) / np.linalg.norm(to_satellite, axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def hours_after_21(times: np.ndarray) -> np.ndarray:
    """Hours since 21:00 UTC of the night; times before noon belong to the night that began the day before."""
    hour_of_day = (times - times.astype("datetime64[D]")) / np.timedelta64(1, "h")
    return np.where(hour_of_day >= 12, hour_of_day - 21, hour_of_day + 3)


def per_scene_local_std(field: np.ndarray) -> np.ndarray:
    """`local_std` of each scene of a (n_scenes, y, x) field."""
    return np.stack([local_std(scene) for scene in field])


def night_feature_stack(
    bt: np.ndarray,
    times: np.ndarray,
    static: StaticFields,
    era5: dict[str, np.ndarray] | None,
    rule_labels: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    """Features of the stored scenes of one night, float32 of shape (n_scenes, n_features, y, x).

    `bt` has shape (n_scenes, 8, y, x) in `CHANNELS` order, `times` is in time order, `era5` holds the
    (n_scenes, y, x) fields `t2m`, `skt`, `d2m` and `sd`. The night's features look only at the stored scenes.
    """
    names = feature_names(groups)
    if "era5" in groups and era5 is None:
        raise ValueError("the era5 feature group needs ERA5 fields; fetch them with fogseg era5")
    if "terrain" in groups and static.elevation_mean is None:
        raise ValueError("the terrain feature group needs the elevation fields of the static fields")

    band = {channel: bt[:, index] for index, channel in enumerate(CHANNELS)}
    ir039, ir108 = band["IR_039"], band["IR_108"]
    btd = ir108 - ir039
    features: dict[str, np.ndarray] = {}

    if "bt" in groups:
        features |= {f"bt_{channel}": values for channel, values in band.items()}
    if "differences" in groups:
        features |= {
            "d108_039": btd,
            "d108_120": ir108 - band["IR_120"],
            "d087_108": band["IR_087"] - ir108,
            "d120_134": band["IR_120"] - band["IR_134"],
            "d108_062": ir108 - band["WV_062"],
            "emissivity_39": emissivity_39(ir039, ir108),
        }
    if "texture" in groups:
        features |= {"std3_ir108": per_scene_local_std(ir108), "std3_btd": per_scene_local_std(btd)}
    if "era5" in groups:
        features |= {
            "t2m_minus_ir108": era5["t2m"] - ir108,
            "skt_minus_ir108": era5["skt"] - ir108,
            "t2m_minus_d2m": era5["t2m"] - era5["d2m"],
            "snow_depth": era5["sd"],
        }
    if "time" in groups:
        # a pixel that is missing in every scene has no maximum, and nanmax warns about that
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            night_max_btd = np.nanmax(btd, axis=0)
        ir108_change = np.full(ir108.shape, np.nan)
        ir108_change[1:] = np.diff(ir108, axis=0)
        features |= {
            "hours_after_21": hours_after_21(times)[:, np.newaxis, np.newaxis],
            "night_max_btd": night_max_btd,
            "ir108_change": ir108_change,
        }

    if "geometry" in groups:
        features |= {"satellite_zenith": static.satellite_zenith, "latitude": static.latitude}
    if "terrain" in groups:
        features |= {"elevation_mean": static.elevation_mean, "elevation_std": static.elevation_std}
    if "rule" in groups:
        features |= {"rule_fog": rule_labels == FOG, "rule_ignore": rule_labels == IGNORE}

    stack = np.empty((bt.shape[0], len(names), *bt.shape[2:]), dtype=np.float32)
    for position, name in enumerate(names):
        stack[:, position] = features[name]
    return stack


@dataclass(frozen=True)
class NightFeatures:
    """The features of one night's scenes with what the teacher needs to judge each pixel.

    `has_era5` is False for a scene that the ERA5 store lacks, `trusted` marks the pixels that the rule can
    label (all channels present, dark enough) and `rule_labels` are the rule's labels at its baseline
    threshold.
    """

    stack: np.ndarray
    has_era5: np.ndarray
    trusted: np.ndarray
    rule_labels: np.ndarray


def read_static(data: xr.Dataset, dem_path: Path, groups: list[str]) -> StaticFields:
    """The static fields of the dataset's grid; the DEM store is read only for `terrain`."""
    lon, lat = data["lon"].values, data["lat"].values
    elevation_mean = elevation_std = None
    if "terrain" in groups:
        dem = read_dem(dem_path)
        if dem is None:
            raise FileNotFoundError(f"{dem_path} is missing; run `fogseg dem` first")
        elevation_mean, elevation_std = dem["elevation_mean"].values, dem["elevation_std"].values
    return StaticFields(lat, satellite_zenith(lon, lat), elevation_mean, elevation_std)


def open_era5(era5_path: Path, groups: list[str]) -> xr.Dataset | None:
    """The ERA5 store, opened lazily; None if `groups` does not use it."""
    if "era5" not in groups:
        return None
    if not era5_path.exists():
        raise FileNotFoundError(f"{era5_path} is missing; run `fogseg era5` first")
    return xr.open_zarr(era5_path, consolidated=False)


def era5_fields(store: xr.Dataset, times: np.ndarray, shape: tuple[int, ...]) -> tuple[dict, np.ndarray]:
    """The ERA5 fields at `times`, NaN for a time the store lacks, and whether the store has each time."""
    position = pd.Index(store["time"].values).get_indexer(times)
    has_era5 = position >= 0
    fields = {name: np.full(shape, np.nan, dtype="float32") for name in ERA5_VARIABLES.values()}
    if has_era5.any():
        found = store[list(fields)].isel(time=position[has_era5]).load()
        for name, values in fields.items():
            values[has_era5] = found[name].values
    return fields, has_era5


def night_features(
    data: xr.Dataset,
    scenes: np.ndarray,
    static: StaticFields,
    era5_store: xr.Dataset | None,
    params: LabelParams,
    groups: list[str],
) -> NightFeatures:
    """Features, trust and rule labels of the dataset scenes `scenes`, which are one night in time order."""
    night = data[["bt", "sza"]].isel(time=scenes).load()
    bt = night["bt"].sel(channel=CHANNELS).values
    sza = night["sza"].values
    times = night["time"].values
    ir039, ir108, ir120 = (bt[:, CHANNELS.index(channel)] for channel in ("IR_039", "IR_108", "IR_120"))

    thresholds = np.full(len(scenes), baseline_threshold(params))
    rule_labels = label_night(ir039, ir108, ir120, sza, thresholds, params)
    trusted = trusted_mask(ir039, ir108, ir120, sza, params.night_min_sza)

    era5, has_era5 = None, np.ones(len(scenes), dtype=bool)
    if "era5" in groups:
        era5, has_era5 = era5_fields(era5_store, times, ir108.shape)
    stack = night_feature_stack(bt, times, static, era5, rule_labels, groups)
    return NightFeatures(stack, has_era5, trusted, rule_labels)


def case_features(
    data: xr.Dataset,
    cases: pd.DataFrame,
    static: StaticFields,
    era5_store: xr.Dataset | None,
    params: LabelParams,
    groups: list[str],
) -> pd.DataFrame:
    """The features at each case's pixel, then `has_era5` and `trusted`; each case night is read once."""
    times = data["time"].values
    scene = pd.Index(times).get_indexer(cases["scene_time"])
    if (scene < 0).any():
        missing = cases["scene_time"].to_numpy()[scene < 0]
        raise ValueError(f"{len(missing)} case scenes are not in the dataset, the first is {missing[0]}")

    names = feature_names(groups)
    values = np.full((len(cases), len(names)), np.nan, dtype="float32")
    has_era5 = np.zeros(len(cases), dtype=bool)
    trusted = np.zeros(len(cases), dtype=bool)
    iy, ix = cases["iy"].to_numpy(), cases["ix"].to_numpy()
    case_night, dataset_night = night_of(times[scene]), night_of(times)
    for night in np.unique(case_night):
        in_night = np.flatnonzero(dataset_night == night)
        rows = np.flatnonzero(case_night == night)
        slot = np.searchsorted(in_night, scene[rows])
        result = night_features(data, in_night, static, era5_store, params, groups)
        values[rows] = result.stack[slot, :, iy[rows], ix[rows]]
        has_era5[rows] = result.has_era5[slot]
        trusted[rows] = result.trusted[slot, iy[rows], ix[rows]]

    table = pd.DataFrame(values, index=cases.index, columns=names)
    return table.assign(has_era5=has_era5, trusted=trusted)
