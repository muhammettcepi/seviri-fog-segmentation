"""Tune the BTD offset and the fallback threshold: relabel the station cases per value and score each value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.build import CHUNK_TIME
from fogseg.dynamic_threshold import DynamicParams, scene_thresholds, usable_peak
from fogseg.label_store import LABEL_CHANNELS
from fogseg.labels import IGNORE, LabelParams, histogram_pixels, label_scene
from fogseg.scores import COUNT_NAMES, bootstrap_skill, counts_by_night, paired_csi_difference, skill

GRID_COLUMNS = ["value", "night", "label", "truth_fls"]
GRIDS = {("dynamic", "offset"), ("dynamic", "btd_fixed"), ("fixed", "btd_fixed")}
GRID_DECIMALS = 6


@dataclass(frozen=True)
class Decision:
    """The best value of a parameter against the current one, with the interval of their CSI difference."""

    parameter: str
    current: float
    best: float
    csi_current: float
    csi_best: float
    difference_low: float
    difference_high: float

    @property
    def changed(self) -> bool:
        """The best value is another one and its CSI is above the current one's for 95 % of the resamples."""
        return self.best != self.current and self.difference_low > 0


def grid_values(spec: list[float]) -> np.ndarray:
    """The values of a `[start, stop, step]` grid, `stop` included, rounded so they compare equal to decimals."""
    start, stop, step = spec
    if step <= 0 or stop < start:
        raise ValueError(f"A grid is [start, stop, step] with start <= stop and step > 0, got {spec}")
    n_steps = int(np.floor((stop - start) / step + 10**-GRID_DECIMALS))
    return np.round(start + step * np.arange(n_steps + 1), GRID_DECIMALS)


def grid_threshold(
    value: float,
    peak: float,
    parameter: str,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams,
) -> float:
    """BTD threshold of a scene when `parameter` takes `value`; the other parameters keep their settings."""
    if mode == "fixed":
        return value
    if parameter == "offset":
        return scene_thresholds(peak, value, params.btd_fixed, dynamic.clamp)
    return scene_thresholds(peak, dynamic.offset, value, dynamic.clamp)


def scene_grid_labels(
    channels: np.ndarray,
    sza: np.ndarray,
    at_scene: pd.DataFrame,
    values: np.ndarray,
    parameter: str,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams,
) -> pd.DataFrame:
    """The labels at the cases of one scene for every value; `channels` holds IR_039, IR_108 and IR_120."""
    ir039, ir108, ir120 = channels
    peak = float("nan")
    if mode == "dynamic":
        peak = usable_peak(histogram_pixels(ir039, ir108, ir120, sza, params), dynamic)

    labels = np.empty((len(values), len(at_scene)), dtype=np.uint8)
    for i, value in enumerate(values):
        threshold = grid_threshold(value, peak, parameter, mode, params, dynamic)
        labels[i] = label_scene(ir039, ir108, ir120, sza, threshold, params)[at_scene["iy"], at_scene["ix"]]

    return pd.DataFrame(
        {
            "value": np.repeat(values, len(at_scene)),
            "night": np.tile(at_scene["night"].to_numpy(), len(values)),
            "label": labels.ravel(),
            "truth_fls": np.tile(at_scene["truth_fls"].to_numpy(), len(values)),
        }
    )


def grid_labels(
    data: xr.Dataset,
    cases: pd.DataFrame,
    params: LabelParams,
    dynamic: DynamicParams,
    parameter: str,
    values: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    """The case rows repeated per value, with the label each scene gets at the case pixel for that value.

    `parameter` is "offset" (dynamic mode) or "btd_fixed", the fallback of the dynamic mode or the
    threshold of the fixed mode. Only scenes that have cases are labelled, read one Zarr time chunk
    at a time. Columns: `value`, `night`, `label`, `truth_fls`.
    """
    if (mode, parameter) not in GRIDS:
        raise ValueError(f"Cannot tune {parameter!r} in {mode!r} mode; the grids are {sorted(GRIDS)}")

    scene = pd.Index(data["time"].values).get_indexer(cases["scene_time"])
    if (scene < 0).any():
        raise ValueError("Some cases belong to scenes that are not in the dataset")
    located = cases.assign(scene=scene)

    frames = []
    for chunk, in_chunk in located.groupby(located["scene"] // CHUNK_TIME):
        first = chunk * CHUNK_TIME
        scenes = data.isel(time=slice(first, first + CHUNK_TIME))
        bt = scenes["bt"].sel(channel=LABEL_CHANNELS).values
        sza = scenes["sza"].values
        for scene_index, at_scene in in_chunk.groupby("scene"):
            i = scene_index - first
            frames.append(
                scene_grid_labels(bt[i], sza[i], at_scene, values, parameter, mode, params, dynamic)
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GRID_COLUMNS)


def score_row(value: float, night_counts: np.ndarray, n_ignored: int, n_samples: int, seed: int) -> dict:
    """Counts, scores and the CSI interval of one grid value."""
    counts = night_counts.sum(axis=0)
    point = skill(counts)
    csi_low, csi_high = bootstrap_skill(night_counts, n_samples, seed)["csi"]
    row = {"value": float(value), "n_cases": int(counts.sum())}
    row |= dict(zip(COUNT_NAMES, counts.tolist(), strict=True))
    row |= {
        "pod": float(point["pod"]),
        "far": float(point["far"]),
        "csi": float(point["csi"]),
        "csi_low": csi_low,
        "csi_high": csi_high,
        "bias": float(point["bias"]),
        "n_ignored": n_ignored,
    }
    return row


def decide(
    grid: pd.DataFrame, parameter: str, current: float, n_samples: int, seed: int
) -> tuple[pd.DataFrame, Decision]:
    """Score every value of a grid and pick the best one.

    The best value has the highest CSI; among equals the one closest to `current`. Whether it
    beats `current` is judged on the same resampled nights for both, so night effects cancel.
    A value whose labels leave no fog case to score has no CSI and ranks last. The table has one
    row per value; `n_ignored` counts the cases the value leaves undecided (label 255).
    """
    values = np.sort(grid["value"].unique())
    is_current = np.isclose(values, current)
    if not is_current.any():
        raise ValueError(f"The current {parameter} {current} is not one of the grid values {values.tolist()}")
    current_row = int(np.flatnonzero(is_current)[0])

    nights = np.unique(grid["night"])
    night_counts = []
    rows = []
    for value in values:
        at_value = grid[grid["value"] == value]
        label, truth = at_value["label"].to_numpy(), at_value["truth_fls"].to_numpy()
        night_counts.append(counts_by_night(at_value["night"].to_numpy(), label, truth, nights))
        n_ignored = int(np.sum(label == IGNORE))
        rows.append(score_row(value, night_counts[-1], n_ignored, n_samples, seed))
    table = pd.DataFrame(rows)

    csi = table["csi"].fillna(-1.0).to_numpy()
    distance_to_current = np.abs(values - values[current_row])
    best_row = int(np.lexsort((distance_to_current, -csi))[0])
    difference_low, difference_high = paired_csi_difference(
        night_counts[best_row], night_counts[current_row], n_samples, seed
    )
    decision = Decision(
        parameter=parameter,
        current=float(values[current_row]),
        best=float(values[best_row]),
        csi_current=float(table.loc[current_row, "csi"]),
        csi_best=float(table.loc[best_row, "csi"]),
        difference_low=difference_low,
        difference_high=difference_high,
    )
    return table, decision
