"""The held-out comparison of the U-Net with the teacher and with the rule, on the cases of the held-out set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.experiment import GRID_COLUMNS, SCORE_COLUMNS, block_counts, held_out_parts, value_scores
from fogseg.scores import COUNT_NAMES, paired_difference

SOURCES = ("rule", "teacher", "unet")
COMPARED_SCORES = ("pod", "far", "csi")
NOTE = "reported, not used for any choice"


def labels_at_cases(store: xr.Dataset, cases: pd.DataFrame) -> np.ndarray:
    """The label of the store at the scene and pixel of every case."""
    scene = pd.Index(store["time"].values).get_indexer(cases["scene_time"])
    if (scene < 0).any():
        name = store.encoding.get("source", "the label store")
        first = cases["scene_time"].to_numpy()[scene < 0][0]
        raise ValueError(f"{(scene < 0).sum()} case scenes are not in the store {name}, the first is {first}")
    at_cases = {
        "time": xr.DataArray(scene, dims="case"),
        "y": xr.DataArray(cases["iy"].to_numpy(), dims="case"),
        "x": xr.DataArray(cases["ix"].to_numpy(), dims="case"),
    }
    return store["label"].isel(at_cases).values.astype(np.uint8)


def case_grid(labels: np.ndarray, cases: pd.DataFrame) -> pd.DataFrame:
    """The labels of the cases as a grid of one value, so that the grid scoring applies to them."""
    return pd.DataFrame({"value": 0.0, "case": cases["case"].to_numpy(), "label": labels})[GRID_COLUMNS]


def held_out_table(held_out: pd.DataFrame, sources: dict[str, np.ndarray], n_folds: int) -> pd.DataFrame:
    """The counts and scores of every source on the whole held-out set and on each of its parts."""
    masks = held_out_parts(held_out, n_folds)
    rows = []
    for source, labels in sources.items():
        grid = case_grid(labels, held_out)
        for part, mask in masks.items():
            scores = value_scores(grid, held_out, mask).drop(columns="value").to_dict("records")[0]
            rows.append({"source": source, "part": part, **scores})
    return pd.DataFrame(rows, columns=["source", "part", *SCORE_COLUMNS[1:]])


def comparison(
    held_out: pd.DataFrame,
    sources: dict[str, np.ndarray],
    candidate: str,
    reference: str,
    stations_cfg: dict,
    season_start: np.datetime64,
) -> dict:
    """The POD gain and the FAR of the candidate against the reference on the whole held-out set.

    The intervals are paired bootstraps over the night blocks of the held-out cases.
    """
    everything = pd.Series(True, index=held_out.index)
    grids = {name: case_grid(sources[name], held_out) for name in (candidate, reference)}
    scores = {name: value_scores(grid, held_out, everything).iloc[0] for name, grid in grids.items()}
    counts = {
        name: block_counts(grid, held_out, 0.0, everything, season_start, stations_cfg["night_block_days"])
        for name, grid in grids.items()
    }
    samples, seed = stations_cfg["bootstrap_samples"], stations_cfg["seed"]
    pod_low, pod_high = paired_difference(counts[candidate], counts[reference], "pod", samples, seed)
    far_low, far_high = paired_difference(counts[candidate], counts[reference], "far", samples, seed)
    return {
        "pod_gain": float(scores[candidate]["pod"] - scores[reference]["pod"]),
        "pod_gain_low": pod_low,
        "pod_gain_high": pod_high,
        "far_candidate": float(scores[candidate]["far"]),
        "far_reference": float(scores[reference]["far"]),
        "far_difference_low": far_low,
        "far_difference_high": far_high,
    }


def differing_sources(table: pd.DataFrame, earlier: pd.DataFrame) -> list[str]:
    """The rule and teacher rows whose held-out counts differ between this table and the phase C table."""
    counts = list(COUNT_NAMES)
    now = table[table["part"] == "held_out"].set_index("source")[counts]
    before = earlier[earlier["part"] == "held_out"].set_index("rule")[counts]
    return [
        name
        for name in ("rule", "teacher")
        if not np.array_equal(now.loc[name].to_numpy(), before.loc[name].to_numpy())
    ]
