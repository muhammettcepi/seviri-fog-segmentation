"""Choose the scenes to judge by eye before the thresholds are tuned.

About 15 training-season scenes that cover the cases the labels can get wrong: much fog, no fog,
high cloud, twilight, fallback thresholds, and the rest spread over the season. The choice depends
only on the data, so it is the same every time. Neighbouring slots of one night look alike, so no
night supplies more than one scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from fogseg.build import CHUNK_TIME

TRAINING_SEASON = "2023-24"
HIGH_CLOUD_MAX_BT108 = 240.0  # K
BATCH_SCENES = 4 * CHUNK_TIME
NIGHT_SPLIT = pd.Timedelta(hours=12)  # moves the small hours back to the evening they belong to

N_FOGGY = 3
N_CLEAR = 3
N_HIGH_CLOUD = 3
N_TWILIGHT = 2
N_FALLBACK = 2
N_CHECK_SCENES = 15
GROUPS = ("foggy", "clear", "high_cloud", "twilight", "fallback")
SPREAD = "spread"
SCENE_TIME_FORMAT = "%Y-%m-%dT%H:%M"  # what `fogseg inspect --time` accepts


def season_of(time: pd.Timestamp) -> str:
    """Fog season of a scene: October 2023 to February 2024 is "2023-24".

    March maps back to the previous season on purpose: the dataset holds 2025-03-01 00, 03 and 05 UTC,
    the morning slots of the night of 28 February.
    """
    start_year = time.year if time.month >= 10 else time.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def night_of(times: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """The night of each scene, as the date of its evening: 21:00 on D and 00, 03, 05 UTC on D+1 give D."""
    return (pd.DatetimeIndex(times) - NIGHT_SPLIT).normalize()


def share(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """`part / whole` per scene, 0.0 where `whole` is 0."""
    result = np.zeros(part.shape, dtype=float)
    np.divide(part, whole, out=result, where=whole > 0)
    return result


def batch_fractions(scenes: xr.Dataset, night_min_sza: float) -> tuple[np.ndarray, np.ndarray]:
    """High-cloud share of the night pixels and twilight share of all pixels, per scene."""
    ir108 = scenes["bt"].sel(channel="IR_108").values
    sza = scenes["sza"].values
    scene_axes = (1, 2)

    is_night_with_bt = (sza > night_min_sza) & np.isfinite(ir108)
    is_high_cloud = is_night_with_bt & (ir108 < HIGH_CLOUD_MAX_BT108)
    high_cloud = share(is_high_cloud.sum(axis=scene_axes), is_night_with_bt.sum(axis=scene_axes))

    is_twilight = sza <= night_min_sza
    has_sza = np.isfinite(sza)
    twilight = share(is_twilight.sum(axis=scene_axes), has_sza.sum(axis=scene_axes))
    return high_cloud, twilight


def check_same_scenes(dataset: xr.Dataset, labels: xr.Dataset) -> None:
    """The label store must have been made from this dataset, scene for scene."""
    if not np.array_equal(dataset["time"].values, labels["time"].values):
        raise ValueError("The label store has other scenes than the dataset; run `fogseg label` again")


def scene_statistics(dataset: xr.Dataset, labels: xr.Dataset) -> pd.DataFrame:
    """One row per scene: time, season, fog, high-cloud and twilight fraction, threshold source."""
    check_same_scenes(dataset, labels)
    night_min_sza = float(labels.attrs["night_min_sza"])

    high_cloud_parts = []
    twilight_parts = []
    n_scenes = dataset.sizes["time"]
    for start in range(0, n_scenes, BATCH_SCENES):
        scenes = dataset.isel(time=slice(start, start + BATCH_SCENES))
        high_cloud, twilight = batch_fractions(scenes, night_min_sza)
        high_cloud_parts.append(high_cloud)
        twilight_parts.append(twilight)

    times = pd.DatetimeIndex(labels["time"].values)
    stats = pd.DataFrame(
        {
            "time": times,
            "season": [season_of(time) for time in times],
            "fog_fraction": labels["fog_fraction"].values,
            "high_cloud_fraction": np.concatenate(high_cloud_parts),
            "twilight_fraction": np.concatenate(twilight_parts),
            "threshold_source": [str(source) for source in labels["threshold_source"].values],
        }
    )
    return stats.sort_values("time").reset_index(drop=True)


def first_of_each_night(ordered: pd.DataFrame) -> pd.DataFrame:
    """The first row of every night in `ordered`, in the same order."""
    is_repeat = night_of(ordered["time"]).duplicated()
    return ordered[~is_repeat]


def without_nights_of(candidates: pd.DataFrame, picked: pd.DataFrame) -> pd.DataFrame:
    """The candidates from nights that `picked` has no scene in."""
    is_picked_night = night_of(candidates["time"]).isin(night_of(picked["time"]))
    return candidates[~is_picked_night]


def top_scenes(candidates: pd.DataFrame, column: str, n: int, highest: bool) -> pd.DataFrame:
    """The best scene of each of the `n` best nights, by `column` (highest or lowest); ties go to the earlier time."""
    ordered = candidates.sort_values([column, "time"], ascending=[not highest, True])
    return first_of_each_night(ordered).head(n)


def pick_group(candidates: pd.DataFrame, group: str) -> pd.DataFrame:
    """The scenes of one group, from the candidates no earlier group has taken."""
    if group == "foggy":
        return top_scenes(candidates, "fog_fraction", N_FOGGY, highest=True)
    if group == "clear":
        return top_scenes(candidates, "fog_fraction", N_CLEAR, highest=False)
    if group == "high_cloud":
        return top_scenes(candidates, "high_cloud_fraction", N_HIGH_CLOUD, highest=True)
    if group == "twilight":
        return top_scenes(candidates, "twilight_fraction", N_TWILIGHT, highest=True)
    if group == "fallback":
        fallback = candidates[candidates["threshold_source"] == "fallback"]
        return first_of_each_night(fallback.sort_values("time")).head(N_FALLBACK)
    raise ValueError(f"unknown group {group!r}")


def spread_scenes(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """The earliest scene of `n` nights spaced evenly in time (of all nights if there are fewer)."""
    ordered = first_of_each_night(candidates.sort_values("time"))
    n = min(n, len(ordered))
    if n <= 0:
        return ordered.head(0)
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
    return ordered.iloc[positions]


def select_check_scenes(stats: pd.DataFrame) -> pd.DataFrame:
    """The check scenes of the training season: rows of `stats` plus their `group`, sorted by time."""
    remaining = stats[stats["season"] == TRAINING_SEASON]
    selected = []
    for group in GROUPS:
        picked = pick_group(remaining, group)
        selected.append(picked.assign(group=group))
        remaining = without_nights_of(remaining, picked)

    n_picked = sum(len(picked) for picked in selected)
    spread = spread_scenes(remaining, N_CHECK_SCENES - n_picked)
    selected.append(spread.assign(group=SPREAD))

    chosen = pd.concat(selected)
    return chosen.sort_values("time").reset_index(drop=True)


def group_counts(selected: pd.DataFrame) -> dict[str, int]:
    """Number of selected scenes per group, in the order the groups are filled."""
    counts = {}
    for group in (*GROUPS, SPREAD):
        counts[group] = int((selected["group"] == group).sum())
    return counts


def write_check_scenes(selected: pd.DataFrame, mode: str, path: Path) -> None:
    """Write the selection as YAML: the label mode and each scene's time and group."""
    scenes = []
    for time, group in zip(selected["time"], selected["group"], strict=True):
        scenes.append({"time": time.strftime(SCENE_TIME_FORMAT), "group": group})
    document = {"labels": mode, "scenes": scenes}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
