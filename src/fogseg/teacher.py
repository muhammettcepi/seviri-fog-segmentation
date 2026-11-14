"""The teacher: a per-pixel classifier of station truth, its blocked out-of-fold probabilities and its p*."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xarray as xr
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
    KeepDecision,
    baseline_threshold,
    block_counts,
    check_split,
    config_record,
    held_out_parts,
    matched_value,
    score_columns,
    value_scores,
    variant_grid,
    warn_at_grid_edge,
)
from fogseg.features import StaticFields, case_features, feature_names
from fogseg.labels import FOG, IGNORE, NO_FOG, LabelParams
from fogseg.scores import COUNT_NAMES, paired_difference
from fogseg.splits import cv_folds, splits_path
from fogseg.tuning import grid_values
from fogseg.validation import TRUTH_YES, haversine_km

log = logging.getLogger(__name__)

NOT_INPUTS = ["has_era5", "trusted"]
TWIN_RADIUS_KM = 5.0
OOF_CASE_COLUMNS = [
    "case",
    "scene_time",
    "station",
    "lat",
    "lon",
    "iy",
    "ix",
    "night_fold",
    "station_fold",
    "truth_fls",
]
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


def model_columns(features: pd.DataFrame) -> pd.Index:
    """The columns a model takes as input: every feature column."""
    return features.columns.difference(NOT_INPUTS, sort=False)


def usable_rows(features: pd.DataFrame) -> np.ndarray:
    """Whether each row has ERA5 and a trusted pixel, the only rows a model is fitted or scored on."""
    return (features["has_era5"] & features["trusted"]).to_numpy()


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
    inputs, is_usable = model_columns(features), usable_rows(features)
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
    rows = [
        {"value": value, **score_columns(row)}
        for value, row in zip(fold_scores[0]["value"], counts, strict=True)
    ]
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def pooled_value_scores(
    fold_probs: pd.DataFrame, cases: pd.DataFrame, values: np.ndarray, halfwidth: float
) -> pd.DataFrame:
    """The scores of every value with the counts of the folds' test parts added up.

    `fold_probs` holds the calibrated `p`. A case in two test parts counts twice.
    """
    return summed_scores(list(fold_value_scores(fold_probs, cases, values, halfwidth).values()))


def setting_rows(
    fold_probs: pd.DataFrame,
    cases: pd.DataFrame,
    values: np.ndarray,
    halfwidth: float,
    far_target: float,
    name: str,
) -> pd.DataFrame:
    """The rows of a setting at its pooled p*: one per fold on its test part, then the pooled row."""
    fold_scores = fold_value_scores(fold_probs, cases, values, halfwidth)
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
    return pd.DataFrame(rows, columns=CV_COLUMNS)


def cross_validate_setting(
    features: pd.DataFrame,
    cases: pd.DataFrame,
    options: dict,
    teacher_params: TeacherParams,
    stations_cfg: dict,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of one setting at its pooled p*, and its fold probabilities."""
    raw = fold_probabilities(features, cases, options, stations_cfg["n_folds"], stations_cfg["seed"], name)
    fold_probs = calibrated(raw, fit_calibrator(raw, cases))
    values = grid_values(teacher_params.probability_grid)
    halfwidth = teacher_params.ignore_halfwidth
    return setting_rows(fold_probs, cases, values, halfwidth, stations_cfg["matched_far"], name), fold_probs


def cross_validate_settings(
    features: pd.DataFrame, cases: pd.DataFrame, teacher_params: TeacherParams, stations_cfg: dict
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The cross-validation table of every model setting, and each setting's out-of-fold probabilities.

    Per setting, p* is the calibrated probability matched to `matched_far` on the pooled test parts. Each
    fold row is scored at that p* on its test part, and the pooled row adds the folds up.
    """
    tables, probabilities = [], {}
    for name, options in MODEL_SETTINGS:
        started = time.perf_counter()
        table, probabilities[name] = cross_validate_setting(
            features, cases, options, teacher_params, stations_cfg, name
        )
        pooled = table.iloc[-1]
        log.info(
            "Setting %s: p* %g, pooled POD %.3f, FAR %.3f, %.1f s",
            name,
            pooled.value,
            pooled.pod,
            pooled.far,
            time.perf_counter() - started,
        )
        tables.append(table.assign(variant=name, options=json.dumps(options)))
    return pd.concat(tables, ignore_index=True)[["variant", "options", *CV_COLUMNS]], probabilities


@dataclasses.dataclass(frozen=True)
class TeacherModel:
    """A fitted teacher: classifier, calibrator and what labelling needs of them.

    `label_params` are the label settings its features were made with, `sklearn_version` the scikit-learn
    that fitted it, and `correction_km` the station-correction length, None until one is chosen.
    """

    setting: str
    options: dict
    model: object
    calibrator: IsotonicRegression
    feature_names: list[str]
    feature_groups: list[str]
    p_star: float
    ignore_halfwidth: float
    label_params: dict
    sklearn_version: str = sklearn.__version__
    correction_km: float | None = None


def fit_teacher(
    features: pd.DataFrame,
    cases: pd.DataFrame,
    setting: tuple[str, dict],
    fold_probs: pd.DataFrame,
    p_star: float,
    teacher_params: TeacherParams,
    label_params: LabelParams,
    seed: int,
) -> TeacherModel:
    """The teacher of a setting fitted on every selection case with ERA5 and a trusted pixel.

    The calibrator is the one of the setting's out-of-fold probabilities `fold_probs`.
    """
    name, options = setting
    columns = model_columns(features)
    fit_rows = usable_rows(features)
    is_fog = (cases["truth_fls"] == TRUTH_YES).to_numpy()
    model = fitted_model(features.loc[fit_rows, columns], is_fog[fit_rows], options, seed, f"Setting {name}")
    return TeacherModel(
        setting=name,
        options=options,
        model=model,
        calibrator=fit_calibrator(fold_probs, cases),
        feature_names=list(columns),
        feature_groups=list(teacher_params.feature_groups),
        p_star=p_star,
        ignore_halfwidth=teacher_params.ignore_halfwidth,
        label_params=dataclasses.asdict(label_params),
    )


def predict_probability(teacher: TeacherModel, rows: np.ndarray) -> np.ndarray:
    """The calibrated probability of FLS of every row, which has the teacher's features as columns."""
    raw = teacher.model.predict_proba(pd.DataFrame(rows, columns=teacher.feature_names))[:, 1]
    return teacher.calibrator.predict(raw)


def save_teacher(teacher: TeacherModel, path: Path) -> None:
    """Write the teacher to a `.part` file and move it into place, so a crash leaves no partial `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    joblib.dump(teacher, part)
    part.replace(path)


def load_teacher(path: Path, groups: list[str]) -> TeacherModel:
    """The teacher saved at `path`, which must have been fitted on the features of `groups`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `fogseg teacher` first")
    teacher = joblib.load(path)
    expected = feature_names(groups)
    if teacher.feature_names != expected:
        raise ValueError(
            f"The teacher in {path} was fitted on the features {teacher.feature_names}, but the feature "
            f"groups {groups} give {expected}; run `fogseg teacher` again"
        )
    if teacher.sklearn_version != sklearn.__version__:
        log.warning(
            "The teacher in %s was fitted with scikit-learn %s, but %s is installed",
            path,
            teacher.sklearn_version,
            sklearn.__version__,
        )
    return teacher


def oof_table(fold_probs: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """The selection cases with their mean calibrated out-of-fold probability `p_oof`, NaN if unpredicted."""
    p_oof = fold_probs.groupby("case")["p"].mean()
    return cases[OOF_CASE_COLUMNS].assign(p_oof=cases["case"].map(p_oof).to_numpy())


def idw_weights(
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    feed_lon: np.ndarray,
    feed_lat: np.ndarray,
    length_km: float,
    min_km: float = 0.0,
) -> np.ndarray:
    """The weight exp(-d / L) of every feeder (columns) for every target (rows).

    It is 0 from 3 L on, and for a feeder closer than `min_km`.
    """
    distance = haversine_km(feed_lon[None, :], feed_lat[None, :], target_lon[:, None], target_lat[:, None])
    near_enough = (distance >= min_km) & (distance < 3 * length_km)
    return np.where(near_enough, np.exp(-distance / length_km), 0.0)


def weighted_shift(weights: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Each target's shift: the weighted residuals over one plus the weights, which damps a lone feeder."""
    return weights @ residuals / (1 + weights.sum(axis=1))


def idw_shift(
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    feed_lon: np.ndarray,
    feed_lat: np.ndarray,
    residuals: np.ndarray,
    length_km: float,
) -> np.ndarray:
    """The inverse-distance weighted shift of each target from the residuals of the feeders around it."""
    return weighted_shift(idw_weights(target_lon, target_lat, feed_lon, feed_lat, length_km), residuals)


def corrected_probability(
    targets: pd.DataFrame, p: np.ndarray, feeders: pd.DataFrame, length_km: float, twin_km: float = 0.0
) -> np.ndarray:
    """`p` shifted by the residuals of other stations' feeders in the same scene, clipped to [0, 1].

    Feeders closer than `twin_km` to a target never feed it.
    """
    shift = np.zeros(len(targets))
    feeder_rows = feeders.groupby("scene_time").indices
    for scene, rows in targets.groupby("scene_time").indices.items():
        if scene not in feeder_rows:
            continue
        target, feeder = targets.iloc[rows], feeders.iloc[feeder_rows[scene]]
        weights = idw_weights(
            target["lon"].to_numpy(),
            target["lat"].to_numpy(),
            feeder["lon"].to_numpy(),
            feeder["lat"].to_numpy(),
            length_km,
            twin_km,
        )
        weights[target["station"].to_numpy()[:, None] == feeder["station"].to_numpy()[None, :]] = 0.0
        shift[rows] = weighted_shift(weights, feeder["residual"].to_numpy())
    return np.clip(p + shift, 0.0, 1.0)


def correction_feeders(oof: pd.DataFrame) -> pd.DataFrame:
    """The cases of `oof` with a probability, with their residual `truth_fls - p_oof`."""
    usable = oof[oof["p_oof"].notna()]
    return usable.assign(residual=(usable["truth_fls"] == TRUTH_YES) - usable["p_oof"])


def correction_fold_scores(
    rows: pd.DataFrame, p: np.ndarray, cases: pd.DataFrame, p_star: float, halfwidth: float
) -> pd.DataFrame:
    """The scores at `p_star` of one fold's test cases labelled from the probabilities `p`."""
    grid = teacher_grid(rows.assign(p=p), np.array([p_star]), halfwidth)
    return value_scores(grid, cases, cases["case"].isin(rows["case"]))


def cross_validate_correction(
    fold_probs: pd.DataFrame,
    cases: pd.DataFrame,
    oof: pd.DataFrame,
    p_star: float,
    halfwidth: float,
    lengths: list[float],
    n_folds: int,
) -> pd.DataFrame:
    """The scores at `p_star` without a correction (`"none"`) and with each length, per fold and pooled.

    A test case of fold k is corrected by the residuals of the other stations in its scene whose station
    fold is neither k nor the held-out `n_folds`, and that are not a twin within `TWIN_RADIUS_KM` of it.
    """
    feeders = correction_feeders(oof)
    scores = {"none": [], **{str(float(length)): [] for length in lengths}}
    for fold, rows in fold_probs.groupby("fold"):
        targets = cases.iloc[rows["case"].to_numpy()]
        fold_feeders = feeders[~feeders["station_fold"].isin([fold, n_folds])]
        p = {"none": rows["p"].to_numpy()}
        for length in lengths:
            p[str(float(length))] = corrected_probability(
                targets, p["none"], fold_feeders, length, TWIN_RADIUS_KM
            )
        for name, probability in p.items():
            fold_scores = correction_fold_scores(rows, probability, cases, p_star, halfwidth)
            scores[name].append(fold_scores.assign(fold=fold))
    tables = []
    for name, fold_scores in scores.items():
        pooled = summed_scores(fold_scores).assign(fold="pooled")
        tables.append(pd.concat([*fold_scores, pooled], ignore_index=True).assign(length_km=name))
    return pd.concat(tables, ignore_index=True)[["length_km", "fold", *SCORE_COLUMNS[1:]]]


def chosen_length(table: pd.DataFrame) -> float | None:
    """The length of the highest pooled POD at no higher FAR than without a correction, if it beats it.

    The shorter length wins a tie.
    """
    pooled = table[table["fold"].astype(str) == "pooled"].set_index("length_km")
    none = pooled.loc["none"]
    allowed = pooled.drop(index="none")
    allowed = allowed[allowed["far"] <= none["far"]]
    allowed = allowed.loc[sorted(allowed.index, key=float)]
    if allowed.empty or allowed["pod"].max() <= none["pod"]:
        return None
    return float(allowed["pod"].idxmax())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_record(
    season: str, mode: str, params: LabelParams, stations_cfg: dict, teacher_cfg: dict, data_dir: Path
) -> dict:
    """The configuration a teacher was fitted with: the phase B record, the `teacher:` section, and the
    sha256 of the case table and the split file it used."""
    inputs = {
        "cases_sha256": file_sha256(data_dir / f"cases_{mode}_{season}.csv"),
        "splits_sha256": file_sha256(splits_path(data_dir, season)),
    }
    return {**config_record(season, mode, params, stations_cfg), "teacher": teacher_cfg, "inputs": inputs}


def probability_of_cases(teacher: TeacherModel, features: pd.DataFrame) -> np.ndarray:
    """The teacher's probability of each case, NaN where it has no ERA5 or no trusted pixel."""
    p = np.full(len(features), np.nan)
    usable = usable_rows(features)
    if usable.any():
        p[usable] = predict_probability(teacher, features.loc[usable, model_columns(features)].to_numpy())
    return p


def keep_decision(
    candidate: str,
    reference: str,
    value: float,
    scores: pd.DataFrame,
    counts: dict[str, np.ndarray],
    stations_cfg: dict,
) -> KeepDecision:
    """The candidate against the reference: `scores` has a row of each, `counts` their block counts."""
    samples, seed = stations_cfg["bootstrap_samples"], stations_cfg["seed"]
    pod_low, pod_high = paired_difference(counts[candidate], counts[reference], "pod", samples, seed)
    far_low, far_high = paired_difference(counts[candidate], counts[reference], "far", samples, seed)
    return KeepDecision(
        winner=candidate,
        value=value,
        pod_gain=float(scores.loc[candidate, "pod"] - scores.loc[reference, "pod"]),
        pod_gain_low=pod_low,
        pod_gain_high=pod_high,
        far_winner=float(scores.loc[candidate, "far"]),
        far_baseline=float(scores.loc[reference, "far"]),
        far_difference_low=far_low,
        far_difference_high=far_high,
    )


def teacher_held_out(
    data: xr.Dataset,
    selection: pd.DataFrame,
    held_out: pd.DataFrame,
    teacher: TeacherModel,
    oof: pd.DataFrame,
    params: LabelParams,
    stations_cfg: dict,
    season_start: np.datetime64,
    static: StaticFields,
    era5_store: xr.Dataset | None,
) -> tuple[pd.DataFrame, KeepDecision, KeepDecision | None]:
    """Score the teacher, and the corrected teacher if it has a length, against the rule on held-out cases.

    The rule keeps the threshold of `params`; the teacher labels at its p*. The first decision is the teacher
    against the rule on all held-out cases. The second is the corrected teacher against the plain one on the
    `stations` part, whose stations the correction never saw; it is None without a correction length.
    The corrected teacher is shifted by the residuals of the selection cases of `oof`.
    """
    n_folds = stations_cfg["n_folds"]
    check_split(selection, held_out, n_folds)

    features = case_features(data, held_out, static, era5_store, params, teacher.feature_groups)
    probabilities = {"teacher": probability_of_cases(teacher, features)}
    if teacher.correction_km is not None:
        probabilities["corrected"] = corrected_probability(
            held_out, probabilities["teacher"], correction_feeders(oof), teacher.correction_km
        )
    rule_value = baseline_threshold(params)
    labellings = {"rule": (variant_grid(data, held_out, params, np.array([rule_value])), rule_value)}
    for name, p in probabilities.items():
        grid = teacher_grid(held_out.assign(p=p), np.array([teacher.p_star]), teacher.ignore_halfwidth)
        labellings[name] = (grid, teacher.p_star)

    masks = held_out_parts(held_out, n_folds)
    table = pd.DataFrame(
        [
            {"rule": name, "part": part, **value_scores(grid, held_out, mask).to_dict("records")[0]}
            for name, (grid, _) in labellings.items()
            for part, mask in masks.items()
        ]
    )

    block_days = stations_cfg["night_block_days"]

    def decide(candidate: str, reference: str, part: str) -> KeepDecision:
        counts = {}
        for name in (candidate, reference):
            grid, value = labellings[name]
            counts[name] = block_counts(grid, held_out, value, masks[part], season_start, block_days)
        scores = table[table["part"] == part].set_index("rule")
        return keep_decision(candidate, reference, teacher.p_star, scores, counts, stations_cfg)

    correction = decide("corrected", "teacher", "stations") if "corrected" in labellings else None
    return table, decide("teacher", "rule", "held_out"), correction
