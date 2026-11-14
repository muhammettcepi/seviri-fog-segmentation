from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import xarray as xr
import yaml
from helpers import TEST_PARAMS
from sklearn.isotonic import IsotonicRegression
from test_experiment import experiment_config
from test_features import teacher_dataset

from fogseg.build import variable_length_strings
from fogseg.cli import cmd_teacher
from fogseg.experiment import (
    CV_COLUMNS,
    RECORDED_STATIONS_KEYS,
    SCORE_COLUMNS,
    check_same_config,
    config_record,
    experiment_cases,
    matched_value,
)
from fogseg.features import feature_names, read_static
from fogseg.labels import FOG, IGNORE, NO_FOG, LabelParams
from fogseg.teacher import (
    MODEL_SETTINGS,
    OOF_CASE_COLUMNS,
    TWIN_RADIUS_KM,
    TeacherModel,
    TeacherParams,
    chosen_length,
    corrected_probability,
    cross_validate_correction,
    cross_validate_settings,
    fit_calibrator,
    fit_teacher,
    fold_probabilities,
    idw_shift,
    load_teacher,
    oof_table,
    pooled_value_scores,
    predict_probability,
    save_teacher,
    teacher_held_out,
    teacher_labels,
)
from fogseg.tuning import grid_values
from fogseg.validation import season_start

LOGISTIC = {"model": "logistic", "C": 1.0}
STATIONS_CFG = {"matched_far": 0.3, "n_folds": 5, "seed": 0}
TEACHER_PARAMS = TeacherParams(
    model_path="unused.joblib",
    ignore_halfwidth=0.1,
    idw_length_km=[25.0],
    feature_groups=["x", "noise"],
    probability_grid=[0.2, 0.8, 0.1],
)


def learnable_cases(held_out: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80 cases in folds 1-4 whose truth follows the feature `x`; the last `held_out` are in fold 5."""
    n = 80
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    truth = (x + rng.normal(scale=0.7, size=n) > 0).astype(int)
    cases = pd.DataFrame(
        {
            "case": np.arange(n),
            "truth_fls": truth,
            "night_fold": np.arange(n) % 4 + 1,
            "station_fold": np.arange(n) // 4 % 4 + 1,
        }
    )
    cases.loc[n - held_out :, "night_fold"] = 5
    features = pd.DataFrame(
        {"x": x, "noise": rng.normal(size=n), "has_era5": True, "trusted": True},
    )
    return features, cases


def hand_probabilities() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated probabilities with a pooled FAR of 0.30 at 0.40, 0.25 at 0.45 and 0.20 at 0.50."""
    p = [0.6] * 4 + [0.47] * 2 + [0.42] + [0.6, 0.47, 0.42] + [0.1] * 4
    truth = [1] * 7 + [0] * 3 + [0] * 4
    n = len(p)
    cases = pd.DataFrame({"case": np.arange(n), "truth_fls": truth, "night_fold": 1, "station_fold": 1})
    fold = np.arange(n) % 2 + 1
    fold_probs = pd.DataFrame({"fold": fold, "case": np.arange(n), "raw": p, "p": p})
    return fold_probs, cases


def test_teacher_labels_have_an_ignore_band():
    p = np.array([0.20, 0.35, 0.45, 0.60, np.nan])
    assert teacher_labels(p, 0.4, 0.1).tolist() == [NO_FOG, IGNORE, IGNORE, FOG, IGNORE]


def test_teacher_labels_are_uint8():
    assert teacher_labels(np.array([0.1, 0.9]), 0.5, 0.1).dtype == np.uint8


def test_settings_are_numbered_and_at_most_ten():
    assert len(MODEL_SETTINGS) <= 10
    assert [int(name.split()[0]) for name, _ in MODEL_SETTINGS] == list(range(1, len(MODEL_SETTINGS) + 1))


def test_params_are_read_from_the_config():
    cfg = {
        "model_path": "teacher.joblib",
        "ignore_halfwidth": 0.1,
        "idw_length_km": [25, 50],
        "feature_groups": ["bt"],
        "probability_grid": [0.1, 0.9, 0.01],
    }
    params = TeacherParams.from_config(cfg)
    assert params.model_path == "teacher.joblib"
    assert params.probability_grid == [0.1, 0.9, 0.01]


def test_a_case_never_predicts_itself():
    features, cases = learnable_cases()
    before = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    cases.loc[0, "truth_fls"] = 1 - cases.loc[0, "truth_fls"]
    after = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    assert np.allclose(before.loc[before["case"] == 0, "raw"], after.loc[after["case"] == 0, "raw"])


def test_a_case_in_two_test_folds_has_a_row_in_each():
    features, cases = learnable_cases()
    probs = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    in_two = (cases["night_fold"] != cases["station_fold"]).to_numpy()
    assert (probs.groupby("case").size() == np.where(in_two, 2, 1)).all()
    assert sorted(probs["fold"].unique()) == [1, 2, 3, 4]


def test_a_model_learns_the_informative_feature():
    features, cases = learnable_cases()
    for _, options in MODEL_SETTINGS:
        probs = fold_probabilities(features, cases, options, 5, 0)
        truth = cases["truth_fls"].to_numpy()[probs["case"].to_numpy()]
        assert probs.loc[truth == 1, "raw"].mean() > probs.loc[truth == 0, "raw"].mean() + 0.2


def test_a_training_part_with_one_truth_class_is_an_error():
    features, cases = learnable_cases()
    cases["truth_fls"] = 0
    with pytest.raises(ValueError, match="Setting 3 hgb, fold 1"):
        fold_probabilities(features, cases, LOGISTIC, 5, 0, "3 hgb")


@pytest.mark.parametrize("column", ["has_era5", "trusted"])
def test_cases_that_cannot_be_fitted_have_no_probability_and_do_not_train(column):
    features, cases = learnable_cases()
    features.loc[0, column] = False
    before = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    cases.loc[0, "truth_fls"] = 1 - cases.loc[0, "truth_fls"]
    probs = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    assert probs.loc[probs["case"] == 0, "raw"].isna().all()
    others = probs["case"] != 0
    assert np.allclose(probs.loc[others, "raw"], before.loc[others, "raw"])


def test_the_calibrator_maps_probabilities_to_the_observed_frequency():
    fold_probs, cases = hand_probabilities()
    calibrator = fit_calibrator(fold_probs, cases)
    assert calibrator.predict([0.1, 0.6]) == pytest.approx([0.0, 0.8])
    assert calibrator.predict([-1.0, 2.0]) == pytest.approx([0.0, 0.8])


def test_the_calibrator_ignores_probabilities_that_are_missing():
    fold_probs, cases = hand_probabilities()
    with_gaps = pd.concat(
        [fold_probs, pd.DataFrame({"fold": [1], "case": [0], "raw": [np.nan]})], ignore_index=True
    )
    with_gaps_fit, plain_fit = fit_calibrator(with_gaps, cases), fit_calibrator(fold_probs, cases)
    assert np.array_equal(with_gaps_fit.predict([0.3, 0.5]), plain_fit.predict([0.3, 0.5]))


def test_held_out_cases_never_enter():
    features, cases = learnable_cases(held_out=6)
    base, _ = cross_validate_settings(features, cases, TEACHER_PARAMS, STATIONS_CFG)
    cases.loc[cases["night_fold"] == 5, "truth_fls"] = 1
    flipped, _ = cross_validate_settings(features, cases, TEACHER_PARAMS, STATIONS_CFG)
    assert base.equals(flipped)


def test_p_star_is_the_matched_far_value():
    fold_probs, cases = hand_probabilities()
    scores = pooled_value_scores(fold_probs, cases, np.array([0.40, 0.45, 0.50]), 0.0)
    assert scores["far"].round(2).tolist() == [0.30, 0.25, 0.20]
    assert matched_value(scores, 0.28) == (0.45, True)


def test_the_pooled_scores_count_a_case_in_two_test_folds_twice():
    fold_probs, cases = hand_probabilities()
    again = pd.concat([fold_probs, fold_probs[fold_probs["case"] == 0].assign(fold=2)], ignore_index=True)
    once = pooled_value_scores(fold_probs, cases, np.array([0.5]), 0.0)
    twice = pooled_value_scores(again, cases, np.array([0.5]), 0.0)
    assert twice.loc[0, "n_cases"] == once.loc[0, "n_cases"] + 1
    assert twice.loc[0, "hits"] == once.loc[0, "hits"] + 1


def test_cases_without_a_probability_count_as_misses():
    fold_probs, cases = hand_probabilities()
    before = pooled_value_scores(fold_probs, cases, np.array([0.5]), 0.0).loc[0]
    fold_probs.loc[fold_probs["case"] == 0, ["raw", "p"]] = np.nan
    after = pooled_value_scores(fold_probs, cases, np.array([0.5]), 0.0).loc[0]
    assert after["hits"] == before["hits"] - 1
    assert after["misses"] == before["misses"] + 1
    assert after["n_cases"] == before["n_cases"]


@pytest.fixture(scope="module")
def cross_validated():
    features, cases = learnable_cases()
    table, probs = cross_validate_settings(features, cases, TEACHER_PARAMS, STATIONS_CFG)
    return table, probs, cases


def test_the_table_has_a_row_per_fold_and_a_pooled_row_for_each_setting(cross_validated):
    table, _, _ = cross_validated
    assert table.columns.tolist() == ["variant", "options", *CV_COLUMNS]
    for name, options in MODEL_SETTINGS:
        rows = table[table["variant"] == name]
        assert rows["fold"].astype(str).tolist() == ["1", "2", "3", "4", "pooled"]
        assert json.loads(rows["options"].iloc[0]) == options


def test_every_row_of_a_setting_carries_its_pooled_p_star(cross_validated):
    table, probs, _ = cross_validated
    for name, _ in MODEL_SETTINGS:
        rows = table[table["variant"] == name]
        assert rows["value"].nunique() == 1
        assert rows["matched"].nunique() == 1
        assert np.isclose(grid_values(TEACHER_PARAMS.probability_grid), rows["value"].iloc[0]).any()
        assert {"fold", "case", "raw", "p"} <= set(probs[name].columns)


def test_the_pooled_row_sums_the_folds(cross_validated):
    table, _, _ = cross_validated
    rows = table[table["variant"] == MODEL_SETTINGS[0][0]]
    folds, pooled = rows[rows["fold"] != "pooled"], rows[rows["fold"] == "pooled"].iloc[0]
    assert pooled["n_cases"] == folds["n_cases"].sum()
    assert pooled["hits"] == folds["hits"].sum()


def test_the_calibrated_probability_is_a_monotone_function_of_the_raw_one(cross_validated):
    _, probs, _ = cross_validated
    ordered = probs[MODEL_SETTINGS[1][0]].sort_values("raw")
    assert ordered["p"].is_monotonic_increasing
    assert ordered["p"].between(0, 1).all()


TEACHER_GROUPS = ["geometry", "rule"]
OOF_COLUMNS = [
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
    "p_oof",
]


def teacher_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`learnable_cases` with the feature names of `TEACHER_GROUPS`; `satellite_zenith` is informative."""
    features, cases = learnable_cases()
    features = features.rename(columns={"x": "satellite_zenith", "noise": "latitude"})
    features = features.assign(rule_fog=features["satellite_zenith"] > 0, rule_ignore=False)
    return features[[*feature_names(TEACHER_GROUPS), "has_era5", "trusted"]], cases


def fitted_teacher(features: pd.DataFrame | None = None, cases: pd.DataFrame | None = None):
    """The logistic teacher of `teacher_features` and the rows of its features."""
    if features is None:
        features, cases = teacher_features()
    probs = fold_probabilities(features, cases, LOGISTIC, 5, 0)
    params = dataclasses.replace(TEACHER_PARAMS, feature_groups=TEACHER_GROUPS)
    teacher = fit_teacher(features, cases, MODEL_SETTINGS[1], probs, 0.5, params, TEST_PARAMS, 0)
    return teacher, features[feature_names(TEACHER_GROUPS)].to_numpy()


def test_the_fitted_teacher_records_what_it_was_fitted_with():
    teacher, _ = fitted_teacher()
    assert teacher.setting == MODEL_SETTINGS[1][0]
    assert teacher.options == MODEL_SETTINGS[1][1]
    assert teacher.feature_names == feature_names(TEACHER_GROUPS)
    assert teacher.feature_groups == TEACHER_GROUPS
    assert teacher.p_star == 0.5
    assert teacher.ignore_halfwidth == TEACHER_PARAMS.ignore_halfwidth
    assert teacher.correction_km is None
    assert teacher.label_params == dataclasses.asdict(TEST_PARAMS)
    assert teacher.sklearn_version == sklearn.__version__


def test_the_probability_is_the_calibrated_one_and_follows_the_truth():
    features, cases = teacher_features()
    teacher, rows = fitted_teacher(features, cases)
    p = predict_probability(teacher, rows)
    raw = teacher.model.predict_proba(features[teacher.feature_names])[:, 1]
    assert np.allclose(p, teacher.calibrator.predict(raw))
    assert p.shape == (len(rows),)
    assert p[cases["truth_fls"] == 1].mean() > p[cases["truth_fls"] == 0].mean() + 0.2


def test_rows_with_missing_features_still_get_a_probability(recwarn):
    teacher, rows = fitted_teacher()
    rows = rows.copy()
    rows[0, :2] = np.nan
    assert np.isfinite(predict_probability(teacher, rows)).all()
    assert not recwarn.list


@pytest.mark.parametrize("column", ["has_era5", "trusted"])
def test_only_cases_with_era5_and_a_trusted_pixel_are_fitted(column):
    features, cases = teacher_features()
    features.loc[0, column] = False
    before, rows = fitted_teacher(features, cases)
    cases.loc[0, "truth_fls"] = 1 - cases.loc[0, "truth_fls"]
    after, _ = fitted_teacher(features, cases)
    assert np.allclose(predict_probability(before, rows), predict_probability(after, rows))


def test_the_teacher_takes_the_feature_columns_only():
    teacher, _ = fitted_teacher()
    assert teacher.model.n_features_in_ == len(feature_names(TEACHER_GROUPS))


def test_saved_teacher_predicts_like_the_fitted_one(tmp_path):
    teacher, rows = fitted_teacher()
    save_teacher(teacher, tmp_path / "teacher.joblib")
    loaded = load_teacher(tmp_path / "teacher.joblib", teacher.feature_groups)
    assert np.allclose(predict_probability(loaded, rows), predict_probability(teacher, rows))


def test_saving_leaves_no_partial_file(tmp_path):
    teacher, _ = fitted_teacher()
    save_teacher(teacher, tmp_path / "models" / "teacher.joblib")
    assert [path.name for path in (tmp_path / "models").iterdir()] == ["teacher.joblib"]


def test_loading_a_teacher_of_another_scikit_learn_version_warns(tmp_path, caplog):
    teacher, _ = fitted_teacher()
    save_teacher(dataclasses.replace(teacher, sklearn_version="0.0.1"), tmp_path / "teacher.joblib")
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        loaded = load_teacher(tmp_path / "teacher.joblib", teacher.feature_groups)
    assert loaded.sklearn_version == "0.0.1"
    assert "0.0.1" in caplog.text
    assert sklearn.__version__ in caplog.text


def test_loading_needs_the_same_features(tmp_path):
    teacher, _ = fitted_teacher()
    save_teacher(teacher, tmp_path / "teacher.joblib")
    with pytest.raises(ValueError, match="rule_fog"):
        load_teacher(tmp_path / "teacher.joblib", ["bt"])
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        load_teacher(tmp_path / "missing.joblib", ["bt"])


def oof_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four selection cases: one in two test parts, one never predicted, two in one part."""
    scenes = ["2023-11-20T21", "2023-11-20T21", "2023-11-21T00", "2023-11-21T00"]
    cases = pd.DataFrame(
        {
            "case": [0, 1, 2, 3],
            "scene_time": pd.to_datetime(scenes),
            "station": ["A", "B", "C", "D"],
            "lat": [45.0, 46.0, 47.0, 48.0],
            "lon": [5.0, 6.0, 7.0, 8.0],
            "iy": [1, 2, 3, 4],
            "ix": [5, 6, 7, 8],
            "night_fold": [1, 2, 3, 4],
            "station_fold": [2, 2, 3, 1],
            "truth_fls": [1, 0, 1, 0],
            "hidden": False,
        }
    )
    fold_probs = pd.DataFrame(
        {
            "fold": [1, 2, 2, 3, 1],
            "case": [0, 0, 1, 2, 3],
            "raw": [0.2, 0.5, np.nan, 0.6, 0.1],
            "p": [0.2, 0.4, np.nan, 0.8, 0.0],
        }
    )
    return fold_probs, cases


def test_the_oof_table_has_the_mean_probability_of_each_case():
    table = oof_table(*oof_inputs())
    assert table.columns.tolist() == OOF_COLUMNS
    assert table["case"].tolist() == [0, 1, 2, 3]
    assert table["p_oof"].iloc[0] == pytest.approx(0.3)
    assert np.isnan(table["p_oof"].iloc[1])
    assert table["p_oof"].iloc[2:].tolist() == pytest.approx([0.8, 0.0])


def test_the_oof_table_keeps_the_case_columns():
    fold_probs, cases = oof_inputs()
    assert oof_table(fold_probs, cases)[OOF_COLUMNS[:-1]].equals(cases[OOF_COLUMNS[:-1]])


def test_idw_shift_by_hand():
    zero = np.array([0.0])
    assert idw_shift(zero, zero, zero, zero, np.array([0.4]), 50.0)[0] == pytest.approx(0.2)
    far_lon = np.array([3.0])  # ~334 km at the equator, beyond 3 L = 150 km
    assert idw_shift(zero, zero, far_lon, zero, np.array([0.4]), 50.0)[0] == 0.0


def test_two_feeders_by_hand():
    zero = np.array([0.0])
    d = 50.0 / 111.195  # one L in degrees at the equator
    feed_lon, feed_lat = np.array([0.0, d]), np.array([0.0, 0.0])
    shift = idw_shift(zero, zero, feed_lon, feed_lat, np.array([0.4, -0.2]), 50.0)
    w = np.array([1.0, np.exp(-1.0)])
    assert shift[0] == pytest.approx((w @ [0.4, -0.2]) / (1 + w.sum()), rel=1e-3)


def test_idw_shift_is_zero_without_feeders():
    empty = np.array([])
    assert idw_shift(np.array([0.0, 1.0]), np.array([0.0, 1.0]), empty, empty, empty, 50.0).tolist() == [0, 0]


def feeder_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One target of station A; feeders: A in the same scene, B in another scene."""
    scene, other = pd.Timestamp("2023-11-20T21"), pd.Timestamp("2023-11-21T00")
    targets = pd.DataFrame({"scene_time": [scene], "lon": [0.0], "lat": [0.0], "station": ["A"]})
    feeders = pd.DataFrame(
        {
            "scene_time": [scene, other],
            "lon": [0.0, 0.0],
            "lat": [0.0, 0.0],
            "station": ["A", "B"],
            "residual": [1.0, 1.0],
        }
    )
    return targets, feeders


def test_own_station_and_other_scenes_never_feed():
    targets, feeders = feeder_tables()
    assert corrected_probability(targets, np.array([0.3]), feeders, 50.0)[0] == pytest.approx(0.3)


def test_a_feeder_of_another_station_in_the_scene_shifts_the_probability():
    targets, feeders = feeder_tables()
    feeders.loc[1, ["scene_time", "station"]] = [targets["scene_time"][0], "B"]
    assert corrected_probability(targets, np.array([0.3]), feeders, 50.0)[0] == pytest.approx(0.8)


def test_the_corrected_probability_is_clipped_and_a_missing_one_stays_missing():
    targets, feeders = feeder_tables()
    feeders.loc[0, ["station", "residual"]] = ["B", 0.1]
    targets = pd.concat([targets, targets], ignore_index=True)
    corrected = corrected_probability(targets, np.array([0.9, np.nan]), feeders.iloc[:1], 50.0)
    assert corrected[0] == pytest.approx(0.95)
    assert np.isnan(corrected[1])
    feeders["residual"] = 100.0
    assert corrected_probability(targets.iloc[:1], np.array([0.1]), feeders.iloc[:1], 50.0)[0] == 1.0
    feeders["residual"] = -100.0
    assert corrected_probability(targets.iloc[:1], np.array([0.9]), feeders.iloc[:1], 50.0)[0] == 0.0


def correction_cases() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One test case in fold 1 and three feeders: stations of folds 2, 1 and 5, in the same scene.

    The allowed feeder has residual 0; the ones in the test fold and in the held-out fold have residual 1.
    The stations are 15 km or more apart, beyond the twin radius.
    """
    n = 4
    cases = pd.DataFrame(
        {
            "case": np.arange(n),
            "scene_time": pd.Timestamp("2023-11-20T21"),
            "station": list("ABCD"),
            "lat": 45.0,
            "lon": 5.0 + 0.2 * np.arange(n),
            "night_fold": 1,
            "station_fold": [1, 2, 1, 5],
            "truth_fls": [1, 1, 1, 1],
        }
    )
    oof = cases.assign(p_oof=[0.45, 1.0, 0.0, 0.0])
    fold_probs = pd.DataFrame({"fold": [1], "case": [0], "raw": [0.45], "p": [0.45]})
    return fold_probs, cases, oof


def test_fold_feeders_skip_the_test_fold_and_the_held_out_stations():
    fold_probs, cases, oof = correction_cases()
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    pooled = table[table["fold"].astype(str) == "pooled"].set_index("length_km")
    assert pooled.loc["50.0", "hits"] == pooled.loc["none", "hits"] == 0


def test_a_feeder_with_a_missing_probability_never_feeds():
    fold_probs, cases, oof = correction_cases()
    oof["p_oof"] = [0.45, np.nan, np.nan, np.nan]
    oof.loc[1, "station_fold"] = 2
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    assert table["hits"].tolist() == [0, 0, 0, 0]


def test_a_feeder_of_an_allowed_fold_lifts_the_label():
    fold_probs, cases, oof = correction_cases()
    oof["p_oof"] = [0.45, 0.0, 0.0, 0.0]
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    assert table.columns.tolist() == ["length_km", "fold", *SCORE_COLUMNS[1:]]
    assert table["length_km"].tolist() == ["none"] * 2 + ["50.0"] * 2
    assert table["fold"].astype(str).tolist() == ["1", "pooled"] * 2
    assert table["hits"].tolist() == [0, 0, 1, 1]


def test_a_twin_station_within_the_twin_radius_never_feeds_the_correction():
    fold_probs, cases, oof = correction_cases()
    oof["p_oof"] = [0.45, 0.0, 0.0, 0.0]
    oof.loc[1, "lon"] = cases.loc[0, "lon"] + 0.01
    assert TWIN_RADIUS_KM > 1.0
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    assert table["hits"].tolist() == [0, 0, 0, 0]


def test_a_station_beyond_the_twin_radius_still_feeds_the_correction():
    fold_probs, cases, oof = correction_cases()
    oof["p_oof"] = [0.45, 0.0, 0.0, 0.0]
    oof.loc[1, "lon"] = cases.loc[0, "lon"] + 0.1
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    assert table["hits"].tolist() == [0, 0, 1, 1]


def correction_table(pod: dict, far: dict) -> pd.DataFrame:
    """Pooled rows of a correction table with the given POD and FAR per length."""
    rows = [
        {"length_km": name, "fold": "pooled", "pod": value, "far": far.get(name, far["none"])}
        for name, value in pod.items()
    ]
    return pd.DataFrame(rows)


def test_no_length_is_chosen_without_a_gain():
    table = correction_table(pod={"none": 0.40, "25.0": 0.39, "50.0": 0.40}, far={"none": 0.27, "25.0": 0.26})
    assert chosen_length(table) is None


def test_the_length_with_the_highest_pod_at_no_higher_far_is_chosen():
    pod = {"none": 0.40, "25.0": 0.42, "50.0": 0.45, "100.0": 0.50}
    table = correction_table(pod, far={"none": 0.27, "25.0": 0.27, "50.0": 0.27, "100.0": 0.30})
    assert chosen_length(table) == 50.0


def test_a_tie_in_pod_goes_to_the_shorter_length():
    table = correction_table(pod={"none": 0.40, "100.0": 0.45, "25.0": 0.45}, far={"none": 0.27})
    assert chosen_length(table) == 25.0


def phase_b_record() -> dict:
    return config_record("2023-24", "fixed", TEST_PARAMS, dict.fromkeys(RECORDED_STATIONS_KEYS, 1))


def test_a_changed_teacher_setting_is_named_when_the_configurations_are_compared():
    recorded = phase_b_record() | {"teacher": dataclasses.asdict(TEACHER_PARAMS)}
    check_same_config(json.loads(json.dumps(recorded)), recorded)
    changed = {**recorded, "teacher": {**recorded["teacher"], "ignore_halfwidth": 0.2}}
    with pytest.raises(ValueError, match=r"teacher\.ignore_halfwidth"):
        check_same_config(recorded, changed)
    with pytest.raises(ValueError, match=r"teacher\.model_path"):
        check_same_config(phase_b_record(), recorded)


def test_a_record_without_a_teacher_section_still_compares():
    check_same_config(phase_b_record(), json.loads(json.dumps(phase_b_record())))


def teacher_config(tmp_path: Path) -> dict:
    """`experiment_config` with all eight channels, a tiny DEM store and a teacher without ERA5."""
    cfg = experiment_config(tmp_path)
    store = Path(cfg["data"]["processed_path"])
    data = teacher_dataset(xr.open_zarr(store, consolidated=False))
    variable_length_strings(data).to_zarr(store, mode="w", zarr_format=3, consolidated=False)

    elevation = np.arange(80, dtype="float32").reshape(8, 10)
    dem = xr.Dataset(
        {"elevation_mean": (("y", "x"), elevation), "elevation_std": (("y", "x"), elevation / 2)}
    )
    dem.to_zarr(tmp_path / "dem.zarr", zarr_format=3, consolidated=False)
    cfg["dem"] |= {"out_path": str(tmp_path / "dem.zarr")}
    cfg["teacher"] |= {
        "model_path": str(tmp_path / "processed" / "teacher.joblib"),
        "feature_groups": ["bt", "differences", "geometry", "terrain", "rule"],
        "probability_grid": [0.2, 0.8, 0.2],
    }
    return cfg


def teacher_args(
    cfg: dict, season: str | None = None, mode: str = "fixed", held_out: bool = False
) -> argparse.Namespace:
    cfg = {**cfg, "labels": {**cfg["labels"], "mode": mode}}
    config_path = Path(cfg["stations"]["data_dir"]).parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, season=season, held_out=held_out)


def test_teacher_command_writes_its_outputs(tmp_path, caplog):
    cfg = teacher_config(tmp_path)
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_teacher(teacher_args(cfg))
    stations = tmp_path / "stations"
    table = pd.read_csv(stations / "teacher_cv.csv")
    assert table["variant"].nunique() == len(MODEL_SETTINGS)
    assert (tmp_path / "processed" / "teacher.joblib").exists()
    assert (stations / "teacher_config.json").exists()
    assert "winner" in caplog.text


def test_teacher_command_logs_each_setting_once_and_the_winner_once(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_teacher(teacher_args(teacher_config(tmp_path)))
    infos = [record.getMessage() for record in caplog.records if record.levelname == "INFO"]
    for name, _ in MODEL_SETTINGS:
        assert sum(message.startswith(f"Setting {name}:") for message in infos) == 1
    assert sum("winner" in message for message in infos) == 1


def test_teacher_command_writes_the_winners_out_of_fold_table_and_the_record(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    stations = tmp_path / "stations"
    oof = pd.read_csv(stations / "teacher_oof.csv")
    assert oof.columns.tolist() == OOF_COLUMNS
    assert oof["case"].tolist() == list(range(len(oof)))
    record = json.loads((stations / "teacher_config.json").read_text(encoding="utf-8"))
    assert record["teacher"] == cfg["teacher"]
    assert record["labels"]["btd_fixed"] == cfg["labels"]["btd_fixed"]
    teacher = load_teacher(Path(cfg["teacher"]["model_path"]), cfg["teacher"]["feature_groups"])
    pooled = pd.read_csv(stations / "teacher_cv.csv").query("fold == 'pooled'")
    assert pooled.loc[pooled["variant"] == teacher.setting, "value"].iloc[0] == teacher.p_star


def test_teacher_command_writes_the_correction_table_and_saves_the_chosen_length(tmp_path, caplog):
    cfg = teacher_config(tmp_path)
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_teacher(teacher_args(cfg))
    table = pd.read_csv(tmp_path / "stations" / "teacher_correction_cv.csv")
    lengths = ["none", *map(str, map(float, cfg["teacher"]["idw_length_km"]))]
    assert table["length_km"].drop_duplicates().tolist() == lengths
    teacher = load_teacher(Path(cfg["teacher"]["model_path"]), cfg["teacher"]["feature_groups"])
    pooled = table[table["fold"].astype(str) == "pooled"].set_index("length_km")
    if teacher.correction_km is None:
        assert "no correction" in caplog.text
    else:
        assert str(teacher.correction_km) in pooled.index


def test_teacher_refuses_the_test_season_and_dynamic_labels(tmp_path):
    cfg = teacher_config(tmp_path)
    with pytest.raises(PermissionError):
        cmd_teacher(teacher_args(cfg, season="2024-25"))
    with pytest.raises(ValueError, match="fixed or teacher"):
        cmd_teacher(teacher_args(cfg, mode="dynamic"))


def test_teacher_command_runs_in_teacher_mode(tmp_path):
    cmd_teacher(teacher_args(teacher_config(tmp_path), mode="teacher"))
    record = json.loads((tmp_path / "stations" / "teacher_config.json").read_text(encoding="utf-8"))
    assert record["mode"] == "fixed"


@pytest.mark.parametrize("name", ["teacher_held_out.csv", "teacher_held_out.json"])
def test_teacher_refuses_a_rerun_after_the_held_out_set_was_opened(tmp_path, name):
    cfg = teacher_config(tmp_path)
    (tmp_path / "stations" / name).write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match=name):
        cmd_teacher(teacher_args(cfg))
    assert not (tmp_path / "stations" / "teacher_cv.csv").exists()


class ConstantModel:
    """A classifier that gives every row the same probability of fog."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        fog = np.full(len(rows), self.probability)
        return np.column_stack([1 - fog, fog])


def held_out_run(tmp_path, teacher_hits_more=True, correction_km=None, probability=None):
    """`teacher_held_out` on the toy season with a teacher that gives every case one probability.

    By default it is 0.9 (all fog) or 0 (no fog); the rule is too strict to ever call fog.
    """
    cfg = teacher_config(tmp_path)
    selection, held_out = experiment_cases(cfg, "2023-24")
    groups = cfg["teacher"]["feature_groups"]
    teacher = TeacherModel(
        setting="toy",
        options={},
        model=ConstantModel(probability if probability is not None else 0.9 * teacher_hits_more),
        calibrator=IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0]),
        feature_names=feature_names(groups),
        feature_groups=groups,
        p_star=0.5,
        ignore_halfwidth=0.1,
        label_params=dataclasses.asdict(TEST_PARAMS),
        correction_km=correction_km,
    )
    oof = selection[OOF_CASE_COLUMNS].assign(p_oof=0.0)
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        static = read_static(data, Path(cfg["dem"]["out_path"]), groups)
        return teacher_held_out(
            data,
            selection,
            held_out,
            teacher,
            oof,
            dataclasses.replace(LabelParams.from_config(cfg["labels"]), btd_fixed=10.0),
            cfg["stations"],
            season_start(cfg, "2023-24"),
            static,
            None,
        )


def test_teacher_gain_is_teacher_minus_rule(tmp_path):
    table, decision, _ = held_out_run(tmp_path, teacher_hits_more=True)
    rows = table.set_index(["rule", "part"])
    gain = rows.loc[("teacher", "held_out"), "pod"] - rows.loc[("rule", "held_out"), "pod"]
    assert decision.pod_gain == pytest.approx(gain)
    assert gain > 0
    assert decision.winner == "teacher"
    assert decision.value == 0.5


def test_the_table_scores_every_labelling_on_every_part(tmp_path):
    table, _, _ = held_out_run(tmp_path, correction_km=50.0)
    assert set(table["rule"]) == {"rule", "teacher", "corrected"}
    assert set(table["part"]) == {"held_out", "nights", "stations", "both"}
    n_cases = table.set_index(["rule", "part"])["n_cases"]
    assert n_cases["teacher", "held_out"] == n_cases["teacher"].drop("held_out").sum()


def test_a_teacher_that_calls_no_fog_scores_no_hits(tmp_path):
    table, decision, _ = held_out_run(tmp_path, teacher_hits_more=False)
    teacher = table[(table["rule"] == "teacher") & (table["part"] == "held_out")].iloc[0]
    assert teacher["hits"] == 0
    assert teacher["misses"] > 0
    assert not decision.keep


def test_there_is_no_correction_decision_without_a_correction_length(tmp_path):
    table, _, correction = held_out_run(tmp_path)
    assert correction is None
    assert "corrected" not in set(table["rule"])


def test_correction_is_judged_on_unseen_stations_only(tmp_path):
    table, _, correction = held_out_run(tmp_path, correction_km=50.0)
    rows = table.set_index(["rule", "part"])
    assert correction is not None
    assert correction.winner == "corrected"
    assert rows.loc[("corrected", "nights"), "hits"] == rows.loc[("teacher", "nights"), "hits"]
    gain = rows.loc[("corrected", "stations"), "pod"] - rows.loc[("teacher", "stations"), "pod"]
    assert correction.pod_gain == pytest.approx(gain)


def test_feeders_lift_the_ignored_probabilities_of_unseen_stations_only(tmp_path):
    table, _, correction = held_out_run(tmp_path, correction_km=5000.0, probability=0.45)
    rows = table.set_index(["rule", "part"])
    assert rows.loc[("teacher", "held_out"), "hits"] == 0
    assert rows.loc[("corrected", "stations"), "hits"] > 0
    assert rows.loc[("corrected", "nights"), "hits"] == 0
    assert correction.pod_gain > 0


def test_mixed_up_selection_and_held_out_cases_are_refused(tmp_path):
    cfg = teacher_config(tmp_path)
    selection, held_out = experiment_cases(cfg, "2023-24")
    nothing = (None,) * 3
    with pytest.raises(ValueError, match="held-out cases include"):
        teacher_held_out(None, selection, selection, *nothing, cfg["stations"], None, None, None)
    with pytest.raises(ValueError, match="selection cases include"):
        teacher_held_out(None, held_out, held_out, *nothing, cfg["stations"], None, None, None)


def test_held_out_runs_once(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    cmd_teacher(teacher_args(cfg, held_out=True))
    with pytest.raises(FileExistsError, match="once"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_held_out_writes_the_table_and_the_decisions(tmp_path, caplog):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_teacher(teacher_args(cfg, held_out=True))
    table = pd.read_csv(tmp_path / "stations" / "teacher_held_out.csv")
    assert {"rule", "teacher"} <= set(table["rule"])
    record = json.loads((tmp_path / "stations" / "teacher_held_out.json").read_text(encoding="utf-8"))
    teacher = load_teacher(Path(cfg["teacher"]["model_path"]), cfg["teacher"]["feature_groups"])
    assert record["p_star"] == teacher.p_star
    assert record["setting"] == teacher.setting
    assert record["correction_km"] == teacher.correction_km
    assert record["teacher"]["winner"] == "teacher"
    assert record["config_changes"] == ({"labels.mode": "teacher"} if record["teacher"]["keep"] else {})
    assert (record["correction"] is None) == (teacher.correction_km is None)
    assert "Held-out" in caplog.text


def test_the_second_opening_is_refused_before_any_labelling(tmp_path):
    cfg = teacher_config(tmp_path)
    (tmp_path / "stations" / "teacher_held_out.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="once"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_held_out_refuses_the_test_season(tmp_path):
    with pytest.raises(PermissionError):
        cmd_teacher(teacher_args(teacher_config(tmp_path), season="2024-25", held_out=True))


def test_held_out_needs_the_teacher_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        cmd_teacher(teacher_args(teacher_config(tmp_path), held_out=True))


def test_held_out_needs_the_saved_model(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    Path(cfg["teacher"]["model_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_held_out_refuses_a_changed_config(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    cfg["teacher"]["ignore_halfwidth"] = 0.2
    with pytest.raises(ValueError, match="ignore_halfwidth"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_a_rerun_that_dies_leaves_no_record_for_held_out_to_trust(tmp_path, monkeypatch):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    config_path = tmp_path / "stations" / "teacher_config.json"
    assert config_path.exists()

    def fail(*args, **kwargs):
        raise RuntimeError("the cross-validation died")

    monkeypatch.setattr("fogseg.teacher.cross_validate_settings", fail)
    with pytest.raises(RuntimeError, match="died"):
        cmd_teacher(teacher_args(cfg))
    assert not config_path.exists()
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_the_record_fingerprints_the_case_table_and_the_split_file(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    record = json.loads((tmp_path / "stations" / "teacher_config.json").read_text(encoding="utf-8"))
    cases = tmp_path / "stations" / "cases_fixed_2023-24.csv"
    assert record["inputs"]["cases_sha256"] == hashlib.sha256(cases.read_bytes()).hexdigest()
    assert len(record["inputs"]["splits_sha256"]) == 64


@pytest.mark.parametrize("name", ["cases_fixed_2023-24.csv", "splits_2023-24.csv"])
def test_held_out_refuses_inputs_changed_since_the_teacher(tmp_path, name):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    path = tmp_path / "stations" / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"inputs\.(cases|splits)_sha256"):
        cmd_teacher(teacher_args(cfg, held_out=True))
    assert not (tmp_path / "stations" / "teacher_held_out.json").exists()


def test_held_out_corrects_with_the_saved_feeders_on_unseen_stations_only(tmp_path, monkeypatch):
    cfg = teacher_config(tmp_path)
    monkeypatch.setattr("fogseg.teacher.chosen_length", lambda table: 25.0)
    cmd_teacher(teacher_args(cfg))
    cmd_teacher(teacher_args(cfg, held_out=True))
    stations = tmp_path / "stations"
    record = json.loads((stations / "teacher_held_out.json").read_text(encoding="utf-8"))
    assert record["correction_km"] == 25.0
    assert record["correction"]["winner"] == "corrected"
    table = pd.read_csv(stations / "teacher_held_out.csv").set_index(["rule", "part"])
    for part in ("nights", "both"):
        assert table.loc[("corrected", part)].equals(table.loc[("teacher", part)])
