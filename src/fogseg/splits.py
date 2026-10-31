"""Frozen evaluation split of the tuning season: night blocks and station cells dealt to folds."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

NIGHT_BLOCK, STATION_CELL = "night_block", "station_cell"


def splits_path(data_dir: Path, season: str) -> Path:
    return data_dir / f"splits_{season}.csv"


def night_block_start(night: np.ndarray, season_start: np.datetime64, block_days: int) -> np.ndarray:
    """The first night of the block each night belongs to; blocks are `block_days` long from the season start."""
    days = (np.asarray(night, dtype="datetime64[D]") - np.datetime64(season_start, "D")).astype(int)
    return np.datetime64(season_start, "D") + (days // block_days) * np.timedelta64(block_days, "D")


def cell_grid_shape(bbox: list[float], cell_deg: float) -> tuple[int, int]:
    """Number of cell columns and rows that cover the bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return int(np.ceil((lon_max - lon_min) / cell_deg)), int(np.ceil((lat_max - lat_min) / cell_deg))


def station_cell(lon: np.ndarray, lat: np.ndarray, bbox: list[float], cell_deg: float) -> np.ndarray:
    """Cell ids like "c2_3" (column, row); a station outside the bbox gets the nearest edge cell."""
    n_cols, n_rows = cell_grid_shape(bbox, cell_deg)
    cols = np.clip(np.floor((np.asarray(lon) - bbox[0]) / cell_deg).astype(int), 0, n_cols - 1)
    rows = np.clip(np.floor((np.asarray(lat) - bbox[1]) / cell_deg).astype(int), 0, n_rows - 1)
    return np.array([f"c{col}_{row}" for col, row in zip(cols, rows, strict=True)], dtype=object)


def make_splits(
    season_start: np.datetime64, season_end: np.datetime64, bbox: list[float], stations_cfg: dict
) -> pd.DataFrame:
    """Folds of every night block and station cell: blocks in turn, cells in a seeded random order."""
    n_folds = stations_cfg["n_folds"]
    block_days = stations_cfg["night_block_days"]
    last_night = np.datetime64(season_end, "D")
    starts = np.arange(
        np.datetime64(season_start, "D"), last_night + np.timedelta64(1, "D"), np.timedelta64(block_days, "D")
    )
    blocks = pd.DataFrame(
        {
            "kind": NIGHT_BLOCK,
            "key": np.datetime_as_string(starts, unit="D"),
            "fold": np.arange(len(starts)) % n_folds + 1,
        }
    )

    n_cols, n_rows = cell_grid_shape(bbox, stations_cfg["station_cell_deg"])
    cell_ids = np.array([f"c{col}_{row}" for col in range(n_cols) for row in range(n_rows)])
    shuffled = np.random.default_rng(stations_cfg["seed"]).permutation(cell_ids)
    cells = pd.DataFrame(
        {"kind": STATION_CELL, "key": shuffled, "fold": np.arange(len(shuffled)) % n_folds + 1}
    )
    return pd.concat([blocks, cells], ignore_index=True)


def fold_of(keys: np.ndarray, splits: pd.DataFrame, kind: str) -> np.ndarray:
    """The fold of each key; a key that is not in the split file is an error naming it."""
    folds = splits[splits["kind"] == kind].set_index("key")["fold"]
    found = pd.Series(keys).map(folds)
    if found.isna().any():
        missing = sorted(set(np.asarray(keys)[found.isna().to_numpy()]))
        raise ValueError(f"The split file has no {kind} for {', '.join(missing)}; redraw it with --overwrite")
    return found.to_numpy().astype(int)


def assign_folds(
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    season_start: np.datetime64,
    bbox: list[float],
    stations_cfg: dict,
) -> pd.DataFrame:
    """The case table with the fold of its night block and of its station cell."""
    nights = pd.to_datetime(cases["night"]).values.astype("datetime64[D]")
    blocks = night_block_start(nights, season_start, stations_cfg["night_block_days"])
    cells = station_cell(
        cases["lon"].to_numpy(), cases["lat"].to_numpy(), bbox, stations_cfg["station_cell_deg"]
    )
    return cases.assign(
        night_fold=fold_of(np.datetime_as_string(blocks, unit="D"), splits, NIGHT_BLOCK),
        station_fold=fold_of(cells, splits, STATION_CELL),
    )


def held_out(cases: pd.DataFrame, n_folds: int) -> pd.Series:
    """Cases whose night or station is in the last fold; nothing may be fitted or chosen on them."""
    return (cases["night_fold"] == n_folds) | (cases["station_fold"] == n_folds)


def held_out_part(cases: pd.DataFrame, n_folds: int) -> pd.Series:
    """Where a case belongs: "selection", or "nights", "stations", "both" for what is unseen in a held-out case."""
    unseen_night = cases["night_fold"] == n_folds
    unseen_station = cases["station_fold"] == n_folds
    part = pd.Series("selection", index=cases.index)
    part[unseen_night] = "nights"
    part[unseen_station] = "stations"
    part[unseen_night & unseen_station] = "both"
    return part


def cv_folds(cases: pd.DataFrame, n_folds: int) -> Iterator[tuple[int, pd.Series, pd.Series]]:
    """Blocked cross-validation on the selection cases: (k, train, test) boolean masks for each fold k."""
    selection = ~held_out(cases, n_folds)
    for k in range(1, n_folds):
        in_k = (cases["night_fold"] == k) | (cases["station_fold"] == k)
        yield k, selection & ~in_k, selection & in_k
