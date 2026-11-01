from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fogseg.experiment import CV_COLUMNS, matched_value
from fogseg.labels import FOG, IGNORE, NO_FOG
from fogseg.teacher import (
    MODEL_SETTINGS,
    TeacherParams,
    cross_validate_settings,
    fit_calibrator,
    fold_probabilities,
    pooled_value_scores,
    teacher_labels,
)
from fogseg.tuning import grid_values

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
