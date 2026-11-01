"""The teacher: a per-pixel classifier of station truth, its blocked out-of-fold probabilities and its p*."""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fogseg.experiment import (
    CV_COLUMNS,
    GRID_COLUMNS,
    SCORE_COLUMNS,
    matched_value,
    score_columns,
    value_scores,
    warn_at_grid_edge,
)
from fogseg.labels import FOG, IGNORE, NO_FOG
from fogseg.scores import COUNT_NAMES
from fogseg.splits import cv_folds
from fogseg.tuning import grid_values
from fogseg.validation import TRUTH_YES

log = logging.getLogger(__name__)

NOT_INPUTS = ["has_era5", "trusted"]
HGB_SETTINGS = [(0.05, 15), (0.05, 31), (0.05, 63), (0.1, 15), (0.1, 31), (0.1, 63)]
MODEL_SETTINGS: list[tuple[str, dict]] = [
    ("1 logistic C=0.1", {"model": "logistic", "C": 0.1}),
    ("2 logistic C=1", {"model": "logistic", "C": 1.0}),
    *[
        (
            f"{number} hgb lr={rate} leaves={leaves}",
            {"model": "hgb", "learning_rate": rate, "max_leaf_nodes": leaves},
        )
        for number, (rate, leaves) in enumerate(HGB_SETTINGS, start=3)
    ],
]


@dataclasses.dataclass(frozen=True)
class TeacherParams:
    """The `teacher:` section of the config."""

    model_path: str
    ignore_halfwidth: float
    idw_length_km: list[float]
    feature_groups: list[str]
    probability_grid: list[float]

    @classmethod
    def from_config(cls, teacher_cfg: dict) -> TeacherParams:
        return cls(**{field.name: teacher_cfg[field.name] for field in dataclasses.fields(cls)})


def make_model(options: dict, seed: int):
    """An unfitted classifier of a model setting's options."""
    if options["model"] == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=options["C"], max_iter=2000),
        )
    return HistGradientBoostingClassifier(
        learning_rate=options["learning_rate"],
        max_leaf_nodes=options["max_leaf_nodes"],
        max_iter=300,
        random_state=seed,
    )


def teacher_labels(p: np.ndarray, p_star: float, halfwidth: float) -> np.ndarray:
    """Fog from `p_star + halfwidth` up, no fog up to `p_star - halfwidth`, else ignored (also where NaN)."""
    no_fog_or_ignore = np.where(p <= p_star - halfwidth, NO_FOG, IGNORE)
    return np.where(p >= p_star + halfwidth, FOG, no_fog_or_ignore).astype(np.uint8)


def fitted_model(features: pd.DataFrame, is_fog: np.ndarray, options: dict, seed: int, where: str):
    """A model of `options` fitted on the rows of `features`; both truth classes must be present."""
    if is_fog.all() or not is_fog.any():
        raise ValueError(f"{where}: the training part has only one truth class")
    return make_model(options, seed).fit(features, is_fog)


def fold_probabilities(
    features: pd.DataFrame,
    cases: pd.DataFrame,
    options: dict,
    n_folds: int,
    seed: int,
    setting: str = "",
) -> pd.DataFrame:
    """The raw probability of each case in every fold whose test part holds it, from a model without it.

    Only cases with ERA5 and a trusted pixel are fitted and predicted; any other case has a NaN `raw`.
    """
    inputs = features.columns.difference(NOT_INPUTS, sort=False)
    is_usable = (features["has_era5"] & features["trusted"]).to_numpy()
    is_fog = (cases["truth_fls"] == TRUTH_YES).to_numpy()
    parts = []
    for fold, train, test in cv_folds(cases, n_folds):
        fit_rows = train.to_numpy() & is_usable
        model = fitted_model(
            features.loc[fit_rows, inputs],
            is_fog[fit_rows],
            options,
            seed,
            f"Setting {setting}, fold {fold}",
        )
        raw = np.full(len(cases), np.nan)
        predict_rows = test.to_numpy() & is_usable
        if predict_rows.any():
            raw[predict_rows] = model.predict_proba(features.loc[predict_rows, inputs])[:, 1]
        parts.append(
            pd.DataFrame({"fold": fold, "case": cases.loc[test, "case"], "raw": raw[test.to_numpy()]})
        )
    return pd.concat(parts, ignore_index=True)


def fit_calibrator(fold_probs: pd.DataFrame, cases: pd.DataFrame) -> IsotonicRegression:
    """The isotonic map from raw probability to the observed share of FLS, fitted on the finite `raw`."""
    finite = fold_probs[np.isfinite(fold_probs["raw"])]
    is_fog = cases["truth_fls"].to_numpy()[finite["case"].to_numpy()] == TRUTH_YES
    return IsotonicRegression(out_of_bounds="clip").fit(finite["raw"], is_fog)


def calibrated(fold_probs: pd.DataFrame, calibrator: IsotonicRegression) -> pd.DataFrame:
    """`fold_probs` with the calibrated probability `p`, NaN where `raw` is."""
    p = np.full(len(fold_probs), np.nan)
    finite = np.isfinite(fold_probs["raw"].to_numpy())
    p[finite] = calibrator.predict(fold_probs.loc[finite, "raw"])
    return fold_probs.assign(p=p)


def teacher_grid(fold_rows: pd.DataFrame, values: np.ndarray, halfwidth: float) -> pd.DataFrame:
    """The teacher label of the cases in `fold_rows` at every value, in `GRID_COLUMNS` form."""
    p = fold_rows["p"].to_numpy()
    labels = teacher_labels(p[None, :], values[:, None], halfwidth)
    return pd.DataFrame(
        {
            "value": np.repeat(values, len(p)),
            "case": np.tile(fold_rows["case"].to_numpy(), len(values)),
            "label": labels.ravel(),
        },
        columns=GRID_COLUMNS,
    )


def fold_value_scores(
    fold_probs: pd.DataFrame, cases: pd.DataFrame, values: np.ndarray, halfwidth: float
) -> dict[int, pd.DataFrame]:
    """The scores of every value on each fold's test part."""
    scores = {}
    for fold, rows in fold_probs.groupby("fold"):
        in_fold = cases["case"].isin(rows["case"])
        scores[fold] = value_scores(teacher_grid(rows, values, halfwidth), cases, in_fold)
    return scores


def summed_scores(fold_scores: list[pd.DataFrame]) -> pd.DataFrame:
    """`SCORE_COLUMNS` per value from the counts of the folds' scores added up."""
    counts = np.sum([scores[list(COUNT_NAMES)].to_numpy() for scores in fold_scores], axis=0)
    rows = [{"value": value, **score_columns(row)} for value, row in zip(fold_scores[0]["value"], counts)]
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def pooled_value_scores(
    fold_probs: pd.DataFrame, cases: pd.DataFrame, values: np.ndarray, halfwidth: float
) -> pd.DataFrame:
    """The scores of every value with the counts of the folds' test parts added up.

    `fold_probs` holds the calibrated `p`. A case in two test parts counts twice.
    """
    return summed_scores(list(fold_value_scores(fold_probs, cases, values, halfwidth).values()))


def cross_validate_setting(
    features: pd.DataFrame,
    cases: pd.DataFrame,
    options: dict,
    teacher_params: TeacherParams,
    stations_cfg: dict,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of one setting at its pooled p*, and its fold probabilities."""
    n_folds, halfwidth = stations_cfg["n_folds"], teacher_params.ignore_halfwidth
    raw = fold_probabilities(features, cases, options, n_folds, stations_cfg["seed"], name)
    fold_probs = calibrated(raw, fit_calibrator(raw, cases))

    values = grid_values(teacher_params.probability_grid)
    fold_scores = fold_value_scores(fold_probs, cases, values, halfwidth)
    far_target = stations_cfg["matched_far"]
    p_star, matched = matched_value(summed_scores(list(fold_scores.values())), far_target)
    if matched:
        warn_at_grid_edge(f"Setting {name}", p_star, values)
    else:
        log.warning("Setting %s: no probability reaches a FAR of %.3f; using %g", name, far_target, p_star)

    at_p_star = {fold: scores[scores["value"] == p_star] for fold, scores in fold_scores.items()}
    rows = [
        {"fold": fold, "matched": matched, **scores.to_dict("records")[0]}
        for fold, scores in at_p_star.items()
    ]
    pooled = summed_scores(list(at_p_star.values())).to_dict("records")[0]
    rows.append({"fold": "pooled", "matched": matched, **pooled})
    return pd.DataFrame(rows, columns=CV_COLUMNS), fold_probs


def cross_validate_settings(
    features: pd.DataFrame, cases: pd.DataFrame, teacher_params: TeacherParams, stations_cfg: dict
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The cross-validation table of every model setting, and each setting's out-of-fold probabilities.

    Per setting, p* is the calibrated probability matched to `matched_far` on the pooled test parts. Each
    fold row is scored at that p* on its test part, and the pooled row adds the folds up.
    """
    tables, probabilities = [], {}
    for name, options in MODEL_SETTINGS:
        table, probabilities[name] = cross_validate_setting(
            features, cases, options, teacher_params, stations_cfg, name
        )
        pooled = table.iloc[-1]
        log.info("Setting %s: p* %g, pooled POD %.3f, FAR %.3f", name, pooled.value, pooled.pod, pooled.far)
        tables.append(table.assign(variant=name, options=json.dumps(options)))
    return pd.concat(tables, ignore_index=True)[["variant", "options", *CV_COLUMNS]], probabilities
