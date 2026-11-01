"""Features of the teacher: a stack per night, read at station pixels to train and everywhere to label."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from fogseg.labels import FOG, IGNORE, emissivity_39, local_std
from fogseg.parallax import EARTH_RADIUS_KM, SATELLITE_RADIUS_KM, unit_vector

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
