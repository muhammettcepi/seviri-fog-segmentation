"""Validate the fog labels against station observations: truth rule, matching, and the case table."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.labels import IGNORE, LabelParams
from fogseg.scores import COUNT_NAMES, bootstrap_skill, contingency, counts_by_night, skill

log = logging.getLogger("fogseg")

SPEC_PATH = "docs/superpowers/specs/2026-10-28-station-validation-design.md"
SOURCES = ("metar", "synop")
INTERVAL_SCORES = ("pod", "far", "csi")
TRUTH_YES, TRUTH_NO, TRUTH_UNDEFINED = 1, 0, -1
EARTH_RADIUS_KM = 6371.0
STATION_COLUMNS = ["station", "source", "lat", "lon", "elevation_m"]
CASE_COLUMNS = [
    "scene_time",
    "night",
    "slot",
    "station",
    "source",
    "lat",
    "lon",
    "iy",
    "ix",
    "visibility_m",
    "ceiling_m",
    "truth_fls",
    "truth_fog",
    "label",
    "hidden",
    "threshold_source",
]


def truth_array(is_yes: np.ndarray, is_no: np.ndarray) -> np.ndarray:
    """1 where `is_yes`, 0 where `is_no`, -1 elsewhere."""
    truth = np.full(is_yes.shape, TRUTH_UNDEFINED, dtype=np.int8)
    truth[is_no] = TRUTH_NO
    truth[is_yes] = TRUTH_YES
    return truth


def fls_truth(visibility_m: np.ndarray, ceiling_m: np.ndarray, limit_m: float) -> np.ndarray:
    """Fog / low-stratus truth: yes if visibility or cloud ceiling is below the limit, no if both are known.

    A ceiling of infinity means no cloud layer and counts as known; NaN means unknown.
    """
    is_yes = (visibility_m < limit_m) | (ceiling_m < limit_m)
    is_no = (visibility_m >= limit_m) & (ceiling_m >= limit_m)
    return truth_array(is_yes, is_no)


def fog_truth(visibility_m: np.ndarray, limit_m: float) -> np.ndarray:
    """Fog-only truth: yes if the visibility is below the limit, undefined if it is missing."""
    return truth_array(visibility_m < limit_m, visibility_m >= limit_m)


def night_of(times: pd.Series | np.ndarray) -> np.ndarray:
    """The date of the evening a scene belongs to, so the 00:00 to 05:00 slots join the evening before."""
    shifted = np.asarray(times, dtype="datetime64[ns]") - np.timedelta64(12, "h")
    return shifted.astype("datetime64[D]")


def haversine_km(lon: np.ndarray, lat: np.ndarray, point_lon: float, point_lat: float) -> np.ndarray:
    """Great-circle distance in km from every (lon, lat) in degrees to one point."""
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    point_lon_rad, point_lat_rad = np.radians(point_lon), np.radians(point_lat)
    half_chord = (
        np.sin((lat_rad - point_lat_rad) / 2) ** 2
        + np.cos(lat_rad) * np.cos(point_lat_rad) * np.sin((lon_rad - point_lon_rad) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord))


def nearest_pixels(
    lon: np.ndarray,
    lat: np.ndarray,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
) -> pd.DataFrame:
    """Nearest pixel centre of each station on the 2-D `lon` / `lat` grid: row `iy`, column `ix`, distance in km."""
    rows = []
    for point_lon, point_lat in zip(station_lon, station_lat, strict=True):
        distance = haversine_km(lon, lat, point_lon, point_lat)
        iy, ix = np.unravel_index(np.argmin(distance), distance.shape)
        rows.append((iy, ix, distance[iy, ix]))
    return pd.DataFrame(rows, columns=["iy", "ix", "distance_km"])


def select_stations(
    stations: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    max_distance_km: float,
    max_elevation_m: float,
) -> pd.DataFrame:
    """Stations close to a pixel centre and low enough, with that pixel added; a missing elevation is kept."""
    pixels = nearest_pixels(lon, lat, stations["lon"].values, stations["lat"].values)
    located = pd.concat([stations.reset_index(drop=True), pixels], axis="columns")
    is_close = located["distance_km"] <= max_distance_km
    is_low = (located["elevation_m"] <= max_elevation_m) | located["elevation_m"].isna()
    return located[is_close & is_low].reset_index(drop=True)


def match_observations(
    observations: pd.DataFrame,
    scene_times: np.ndarray,
    scan_delay_min: float,
    tolerance_min: float,
) -> pd.DataFrame:
    """Per station and scene, the observation closest to the scan time of the scene, within the tolerance.

    The scan time is the nominal scene time plus `scan_delay_min`. Scenes without an observation in
    reach give no row; the result has the observation columns plus `scene_time`.
    """
    stations = observations[["station", "source"]].drop_duplicates()
    scenes = pd.DataFrame({"scene_time": pd.to_datetime(scene_times)})
    wanted = stations.merge(scenes, how="cross")
    wanted["scan_time"] = (wanted["scene_time"] + pd.Timedelta(minutes=scan_delay_min)).astype(
        observations["time"].dtype
    )

    matched = pd.merge_asof(
        wanted.sort_values("scan_time"),
        observations.sort_values("time"),
        left_on="scan_time",
        right_on="time",
        by=["station", "source"],
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )
    has_observation = matched.dropna(subset=["time"]).drop(columns="scan_time")
    # Unmatched scenes held NaN, which turned integer columns such as `iy` into floats
    return has_observation.astype(observations.dtypes.to_dict()).reset_index(drop=True)


def deduplicate(cases: pd.DataFrame) -> pd.DataFrame:
    """One case per scene and pixel: METAR before SYNOP (exact cloud base), then by station id."""
    ordered = cases.sort_values(["source", "station"], kind="stable")
    return ordered.drop_duplicates(["scene_time", "iy", "ix"]).reset_index(drop=True)


def station_cases(
    data: xr.Dataset,
    labels: xr.Dataset,
    observations: pd.DataFrame,
    stations_cfg: dict,
    params: LabelParams,
) -> pd.DataFrame:
    """The case table: truth of every matched station observation next to the label at its pixel.

    `hidden` marks cases whose pixel fails the labels' own warm-top and no-cirrus tests, so a
    high cloud covers the fog. `data` and `labels` must hold the same scenes.
    """
    stations = observations[STATION_COLUMNS].drop_duplicates(["station", "source"])
    selected = select_stations(
        stations,
        data["lon"].values,
        data["lat"].values,
        stations_cfg["max_distance_km"],
        stations_cfg["max_elevation_m"],
    )
    pixels = selected[["station", "source", "iy", "ix"]]
    matched = match_observations(
        observations.merge(pixels, on=["station", "source"]),
        labels["time"].values,
        stations_cfg["scan_delay_min"],
        stations_cfg["time_tolerance_min"],
    )

    scene_index = pd.Index(labels["time"].values).get_indexer(matched["scene_time"])
    at_cases = {
        "time": xr.DataArray(scene_index, dims="case"),
        "y": xr.DataArray(matched["iy"].values, dims="case"),
        "x": xr.DataArray(matched["ix"].values, dims="case"),
    }
    ir108 = data["bt"].sel(channel="IR_108").isel(at_cases).values
    ir120 = data["bt"].sel(channel="IR_120").isel(at_cases).values

    cases = matched.assign(
        night=night_of(matched["scene_time"]),
        slot=matched["scene_time"].dt.strftime("%H:%M"),
        truth_fls=fls_truth(
            matched["visibility_m"].values, matched["ceiling_m"].values, stations_cfg["truth_limit_m"]
        ),
        truth_fog=fog_truth(matched["visibility_m"].values, stations_cfg["truth_limit_m"]),
        label=labels["label"].isel(at_cases).values,
        hidden=(ir108 < params.warm_min_bt108) | (ir108 - ir120 >= params.cirrus_max_split),
        threshold_source=labels["threshold_source"].isel(time=at_cases["time"]).values.astype(object),
    )
    return cases[CASE_COLUMNS]


def season_bounds(cfg: dict, season: str) -> tuple[np.datetime64, np.datetime64]:
    """Noon of a season's first day to noon after its last, so each night keeps its 00:00 to 05:00 slots."""
    seasons = {entry["name"]: entry for entry in cfg["time"]["seasons"]}
    if season not in seasons:
        raise ValueError(f"Unknown season {season!r}; the config has {', '.join(seasons)}")
    noon = np.timedelta64(12, "h")
    first_day = np.datetime64(seasons[season]["start"], "D")
    last_day = np.datetime64(seasons[season]["end"], "D")
    return (
        (first_day + noon).astype("datetime64[ns]"),
        (last_day + np.timedelta64(1, "D") + noon).astype("datetime64[ns]"),
    )


def check_season_allowed(cfg: dict, season: str, final: bool) -> None:
    """Refuse every season but the tuning season unless `final`, so the test season stays unseen."""
    tune_season = cfg["stations"]["tune_season"]
    if season != tune_season and not final:
        raise PermissionError(
            f"Season {season} is not the tuning season {tune_season}. Score it once, at the very end, "
            f"with --final; see {SPEC_PATH}"
        )


def read_observations(data_dir: Path) -> pd.DataFrame:
    """The METAR and SYNOP tables that `fogseg stations` wrote; one of them may be missing."""
    tables = []
    for source in SOURCES:
        path = data_dir / f"observations_{source}.csv"
        if path.exists():
            tables.append(pd.read_csv(path, dtype={"station": str}, parse_dates=["time"]))
        else:
            log.warning("%s is missing; scoring without %s observations", path, source)
    if not tables:
        raise FileNotFoundError(f"No observation tables in {data_dir}; run `fogseg stations` first")
    return pd.concat(tables, ignore_index=True)


class View(NamedTuple):
    """One row of the score table: a group of cases, the truth it is scored against, and the hidden rule."""

    name: str
    group: str
    cases: pd.DataFrame
    truth_column: str = "truth_fls"
    keep_hidden: bool = False


def score_row(view: View, n_samples: int, seed: int) -> dict:
    """Counts, scores with bootstrap intervals, and the cases of the view that were not scored.

    `n_fls_hidden` counts all hidden cases of the view; the two ignored counts are taken among the
    cases that stay in, so no case is counted twice.
    """
    scored = view.cases if view.keep_hidden else view.cases[~view.cases["hidden"]]
    label, truth, night = (
        scored["label"].to_numpy(),
        scored[view.truth_column].to_numpy(),
        scored["night"].to_numpy(),
    )
    counts = contingency(label, truth)
    night_counts = counts_by_night(night, label, truth, np.unique(night))
    if len(night_counts) > 0:
        intervals = bootstrap_skill(night_counts, n_samples, seed)
    else:
        intervals = dict.fromkeys(INTERVAL_SCORES, (np.nan, np.nan))

    row = {"view": view.name, "group": view.group, "n_cases": int(counts.sum())}
    row |= dict(zip(COUNT_NAMES, counts.tolist(), strict=True))
    point = skill(counts)
    for name in INTERVAL_SCORES:
        low, high = intervals[name]
        row |= {name: float(point[name]), f"{name}_low": low, f"{name}_high": high}
    row["bias"] = float(point["bias"])

    is_ignored = label == IGNORE
    row["n_fls_ignored"] = int(np.sum(is_ignored & (truth == TRUTH_YES)))
    row["n_no_fls_ignored"] = int(np.sum(is_ignored & (truth == TRUTH_NO)))
    row["n_fls_hidden"] = int(np.sum(view.cases["hidden"] & (view.cases[view.truth_column] == TRUTH_YES)))
    return row


def score_table(cases: pd.DataFrame, lat_bands: list[float], n_samples: int, seed: int) -> pd.DataFrame:
    """POD, FAR, CSI and bias with bootstrap intervals, one row per view of the case table.

    Besides the pooled `main` row there are rows per source, threshold source, latitude band and slot,
    one against the fog-only truth, and one that keeps the hidden cases.
    """
    pooled = deduplicate(cases)
    views = [View("main", "all", pooled)]
    views += [View("source", source, deduplicate(cases[cases["source"] == source])) for source in SOURCES]
    views += [
        View("threshold_source", name, pooled[pooled["threshold_source"] == name])
        for name in sorted(pooled["threshold_source"].unique())
    ]
    views += [
        View("lat_band", f"{low:g}-{high:g}", pooled[(pooled["lat"] >= low) & (pooled["lat"] < high)])
        for low, high in itertools.pairwise(lat_bands)
    ]
    views += [View("slot", slot, pooled[pooled["slot"] == slot]) for slot in sorted(pooled["slot"].unique())]
    views += [
        View("fog_only", "all", pooled, truth_column="truth_fog"),
        View("with_hidden", "all", pooled, keep_hidden=True),
    ]
    return pd.DataFrame([score_row(view, n_samples, seed) for view in views])
