"""Diagnose why visible station fog is missed: the first labelling rule that fails, and what low-BTD misses look like."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.labels import FOG, OBSCURED, LabelParams, local_std
from fogseg.splits import read_splits, selection_cases
from fogseg.validation import TRUTH_NO, TRUTH_YES, night_of, read_case_table

LOW_BTD_K = 3.5
BAND_HIGH_K = 4.5
CAUSES = ["twilight", "no_data", "low_btd", "band", "texture", "small_blob_or_other"]
CATEGORICAL_VARIABLES = ["ceiling_amount", "auto", "slot", "fog_other_slot"]
NUMERIC_VARIABLES = ["ir108_minus_wv062", "t2m_minus_ir108", "elevation_diff_m"]
QUANTILES = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
TABLE_COLUMNS = ["section", "variable", "statistic", "fls_missed", "no_fls"]
UNKNOWN = "unknown"


def check_fixed_mode(mode: str) -> None:
    """Refuse labels of any mode but fixed, which the cause bands are set for."""
    if mode != "fixed":
        raise ValueError(
            f"The diagnosis needs the fixed labels, not {mode!r}: its cause bands at {LOW_BTD_K} and "
            f"{BAND_HIGH_K} K are defined around the fixed 4.0 K threshold"
        )


def read_selection_cases(
    cases_path: Path, split_path: Path, season_start: np.datetime64, bbox: list[float], stations_cfg: dict
) -> pd.DataFrame:
    """The selection cases of the table `fogseg validate` wrote, one per scene and pixel as in the score table."""
    splits = read_splits(split_path)
    return selection_cases(read_case_table(cases_path), splits, season_start, bbox, stations_cfg)


def scene_index(store: xr.Dataset, scene_times: pd.Series, name: str) -> np.ndarray:
    """Position of the scene of each case in the store; a scene the store does not have is an error."""
    index = pd.Index(store["time"].values).get_indexer(scene_times)
    if (index < 0).any():
        raise ValueError(f"Some cases are at scenes that the {name} does not have")
    return index


def fog_in_other_slot(labels: xr.Dataset, cases: pd.DataFrame) -> np.ndarray:
    """Per case, whether its pixel is fog in another scene of the same night."""
    label_times = labels["time"].values
    label_nights = night_of(label_times)
    case_times = pd.to_datetime(cases["scene_time"]).to_numpy()
    case_nights = night_of(case_times)
    iy, ix = cases["iy"].to_numpy(), cases["ix"].to_numpy()

    has_fog_elsewhere = np.zeros(len(cases), dtype=bool)
    for night in np.unique(case_nights):
        scenes = np.flatnonzero(label_nights == night)
        rows = np.flatnonzero(case_nights == night)
        is_fog = labels["label"].isel(time=scenes).values[:, iy[rows], ix[rows]] == FOG
        is_other_scene = label_times[scenes][:, None] != case_times[rows][None, :]
        has_fog_elsewhere[rows] = (is_fog & is_other_scene).any(axis=0)
    return has_fog_elsewhere


def case_features(
    data: xr.Dataset, labels: xr.Dataset, era5: xr.Dataset | None, cases: pd.DataFrame
) -> pd.DataFrame:
    """The cases with the measurements at their pixel that the diagnosis looks at; ERA5 columns are NaN without it.

    `btd` is IR_108 - IR_039 and `ir108_std3` the IR_108 texture in a 3 x 3 window, both as the labels see them.
    """
    iy, ix = cases["iy"].to_numpy(), cases["ix"].to_numpy()
    in_data = scene_index(data, cases["scene_time"], "dataset")
    in_era5 = None if era5 is None else scene_index(era5, cases["scene_time"], "ERA5 store")

    names = ["btd", "ir108_std3", "sza", "ir108_minus_wv062", "t2m_minus_ir108"]
    features = {name: np.full(len(cases), np.nan) for name in names}
    for scene in np.unique(in_data):
        rows = np.flatnonzero(in_data == scene)
        pixel = (iy[rows], ix[rows])
        ir039, ir108, wv062 = (
            data["bt"].isel(time=scene).sel(channel=channel).values
            for channel in ("IR_039", "IR_108", "WV_062")
        )
        features["btd"][rows] = (ir108 - ir039)[pixel]
        features["ir108_std3"][rows] = local_std(ir108)[pixel]
        features["sza"][rows] = data["sza"].isel(time=scene).values[pixel]
        features["ir108_minus_wv062"][rows] = (ir108 - wv062)[pixel]
        if in_era5 is not None:
            t2m = era5["t2m"].isel(time=in_era5[rows[0]]).values
            features["t2m_minus_ir108"][rows] = (t2m - ir108)[pixel]

    return cases.assign(**features, fog_other_slot=fog_in_other_slot(labels, cases))


def first_cause(cases: pd.DataFrame, params: LabelParams) -> pd.Series:
    """The first labelling rule, in the order the rules are checked, that explains why a case is not fog."""
    sza, btd, std = (cases[name].to_numpy() for name in ("sza", "btd", "ir108_std3"))
    conditions = [
        sza <= params.night_min_sza,
        np.isnan(btd),
        btd <= LOW_BTD_K,
        (btd < BAND_HIGH_K) & (std < params.smooth_max_std),
        std >= params.smooth_max_std,
    ]
    return pd.Series(np.select(conditions, CAUSES[:-1], default=CAUSES[-1]), index=cases.index)


def category_names(values: pd.Series) -> pd.Series:
    """The values as text, with a missing flag or an empty ceiling amount named `unknown`."""
    return values.map(lambda value: UNKNOWN if pd.isna(value) or value == "" else str(value))


def first_cause_rows(missed: pd.DataFrame, params: LabelParams) -> list[tuple]:
    """Count and share of the misses for every first cause, also the causes without a miss."""
    counts = first_cause(missed, params).value_counts().reindex(CAUSES, fill_value=0)
    shares = counts / counts.sum()
    rows = []
    for cause in CAUSES:
        rows.append(("first_cause", cause, "count", counts[cause], np.nan))
        rows.append(("first_cause", cause, "share", shares[cause], np.nan))
    return rows


def low_btd_rows(missed: pd.DataFrame, no_fls: pd.DataFrame) -> list[tuple]:
    """Side by side, the categories and quantiles of the visible night cases at a BTD of 3.5 K or less."""
    rows = []
    for variable in CATEGORICAL_VARIABLES:
        names = {"fls_missed": category_names(missed[variable]), "no_fls": category_names(no_fls[variable])}
        for value in sorted(set(names["fls_missed"]) | set(names["no_fls"])):
            shares = [(group_names == value).mean() for group_names in names.values()]
            rows.append(("low_btd", variable, f"share:{value}", *shares))
    for variable in NUMERIC_VARIABLES:
        for statistic, quantile in QUANTILES.items():
            rows.append(
                (
                    "low_btd",
                    variable,
                    statistic,
                    missed[variable].quantile(quantile),
                    no_fls[variable].quantile(quantile),
                )
            )
    return rows


def diagnosis_table(cases: pd.DataFrame, params: LabelParams) -> pd.DataFrame:
    """Why visible FLS reports are missed and, for the low-BTD misses, how they differ from no-FLS reports.

    A miss is a visible case with FLS truth that the label does not mark as fog, an ignored label included.
    The `low_btd` section looks at visible night cases with a BTD of `LOW_BTD_K` or less.
    """
    visible = cases[~cases["hidden"] & (cases["label"] != OBSCURED)]
    missed = visible[(visible["truth_fls"] == TRUTH_YES) & (visible["label"] != FOG)]
    no_fls = visible[visible["truth_fls"] == TRUTH_NO]

    def low_btd_night(group: pd.DataFrame) -> pd.DataFrame:
        return group[(group["sza"] > params.night_min_sza) & (group["btd"] <= LOW_BTD_K)]

    rows = first_cause_rows(missed, params) + low_btd_rows(low_btd_night(missed), low_btd_night(no_fls))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
