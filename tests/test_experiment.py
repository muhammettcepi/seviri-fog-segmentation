from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fogseg.experiment import (
    GRID_COLUMNS,
    block_counts,
    cross_validate,
    matched_value,
    scored_cases,
    value_scores,
)
from fogseg.validation import read_case_table

SEASON_START = np.datetime64("2023-10-01")


def small_cases(truth, nights=None, hidden=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truth_fls": truth,
            "night": pd.to_datetime(nights or ["2023-10-01"] * len(truth)),
            "hidden": hidden or [False] * len(truth),
        }
    )


def small_grid(value, labels) -> pd.DataFrame:
    return pd.DataFrame({"value": value, "case": np.arange(len(labels)), "label": labels})[GRID_COLUMNS]


def two_value_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four folds of 10 fog-free and 10 fog cases; value 3.0 has many false alarms in fold 1 only."""
    cases, rows = [], []
    for fold in range(1, 5):
        hits_3, false_alarms_3 = (2, 8) if fold == 1 else (10, 2)
        for truth, fog_at_3, fog_at_4 in ((0, false_alarms_3, 0), (1, hits_3, 5)):
            for i in range(10):
                cases.append({"truth_fls": truth, "hidden": False, "night_fold": fold, "station_fold": fold})
                rows.append((3.0, len(cases) - 1, int(i < fog_at_3)))
                rows.append((4.0, len(cases) - 1, int(i < fog_at_4)))
    return scored_cases(pd.DataFrame(cases)), pd.DataFrame(rows, columns=GRID_COLUMNS)


def test_scored_cases_drops_hidden_and_undefined_cases_and_numbers_the_rest():
    cases = small_cases([1, -1, 0, 1], hidden=[False, False, False, True])
    cases.index = [10, 11, 12, 13]

    scored = scored_cases(cases)

    assert scored["truth_fls"].tolist() == [1, 0]
    assert scored["case"].tolist() == [0, 1]
    assert scored.index.tolist() == [0, 1]


def test_value_scores_counts_an_ignored_label_as_no_and_follows_the_mask():
    cases = scored_cases(small_cases([1, 1, 0, 0]))
    grid = small_grid([1.0] * 4, [1, 255, 1, 0])

    everything = value_scores(grid, cases, pd.Series(True, index=cases.index)).iloc[0]
    first_three = value_scores(grid, cases, pd.Series([True, True, True, False])).iloc[0]

    assert everything[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [1, 1, 1, 1]
    assert everything["pod"] == 0.5 and everything["far"] == 0.5 and everything["n_cases"] == 4
    assert first_three[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [1, 1, 1, 0]


def test_value_scores_lists_every_value_of_the_grid_even_without_cases_in_the_mask():
    cases = scored_cases(small_cases([1, 0]))
    grid = pd.concat([small_grid([1.0, 1.0], [1, 0]), small_grid([2.0, 2.0], [0, 0])])

    scores = value_scores(grid, cases, pd.Series([False, False]))

    assert scores["value"].tolist() == [1.0, 2.0]
    assert scores["n_cases"].tolist() == [0, 0]
    assert scores["far"].isna().all()


def test_matched_value_is_the_highest_far_not_above_the_target():
    scores = pd.DataFrame({"value": [3.0, 4.0, 5.0], "far": [0.35, 0.27, 0.20], "pod": [0.7, 0.6, 0.5]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_skips_values_without_any_fog():
    scores = pd.DataFrame({"value": [4.0, 9.0], "far": [0.25, np.nan], "pod": [0.5, 0.0]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_falls_back_to_the_lowest_far():
    scores = pd.DataFrame({"value": [3.0, 4.0], "far": [0.5, 0.4], "pod": [0.8, 0.7]})
    assert matched_value(scores, 0.28) == (4.0, False)


def test_matched_value_breaks_a_tie_by_pod_and_then_by_the_lower_value():
    scores = pd.DataFrame(
        {"value": [3.0, 4.0, 5.0], "far": [0.25, 0.25, 0.25], "pod": [0.6, 0.7, 0.7]},
    )
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_refuses_scores_without_any_fog():
    scores = pd.DataFrame({"value": [4.0], "far": [np.nan], "pod": [0.0]})
    with pytest.raises(ValueError, match="any fog"):
        matched_value(scores, 0.28)


def test_each_fold_chooses_on_its_training_part():
    cases, grid = two_value_grid()  # value 3.0 reaches FAR <= 0.28 only on fold 1's training part
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    assert table.loc[1, "value"] == 3.0 and table.loc[2, "value"] == 4.0
    folds = table.drop(index="pooled")
    assert table.loc["pooled", "hits"] == folds["hits"].sum()
    assert folds["hits"].dtype.kind == "i" and folds["n_cases"].dtype.kind == "i"


def test_the_pooled_row_scores_the_summed_test_counts():
    cases, grid = two_value_grid()
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    pooled = table.loc["pooled"]
    # fold 1 tests value 3.0 (2 hits, 8 false alarms); folds 2 to 4 test value 4.0 (5 hits, 0 false alarms)
    assert pooled[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [17, 23, 8, 32]
    assert pooled["far"] == pytest.approx(8 / 25)
    assert pooled["n_cases"] == 80
    assert pooled["matched"]


@pytest.mark.parametrize("fold_column", ["night_fold", "station_fold"])
def test_held_out_cases_never_enter_the_cross_validation(fold_column):
    cases, grid = two_value_grid()
    fog_free_in_fold_4 = (cases["night_fold"] == 4) & (cases["truth_fls"] == 0)
    cases.loc[fog_free_in_fold_4, fold_column] = 5
    base = cross_validate(grid, cases, 5, 0.28)

    grid.loc[grid["case"].isin(cases.index[fog_free_in_fold_4]), "label"] = 1  # false alarms at both values

    assert cross_validate(grid, cases, 5, 0.28).equals(base)


def test_a_fold_that_falls_back_is_logged(caplog):
    cases, grid = two_value_grid()
    grid["label"] = 1  # every value has a FAR of 0.5
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    assert not table.loc[1, "matched"] and not table.loc["pooled", "matched"]
    assert "Fold 1" in caplog.text and "Fold 4" in caplog.text


def test_block_counts_has_one_row_per_night_block_in_order():
    cases = scored_cases(small_cases([1, 0, 1, 0], ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-03"]))
    grid = small_grid([1.0] * 4, [1, 1, 255, 0])
    everything = pd.Series(True, index=cases.index)

    counts = block_counts(grid, cases, 1.0, everything, SEASON_START, block_days=2)
    without_first = block_counts(
        grid, cases, 1.0, pd.Series([False, True, True, True]), SEASON_START, block_days=2
    )

    assert counts.tolist() == [[1, 0, 1, 0], [0, 1, 0, 1]]
    assert without_first.tolist() == [[0, 0, 1, 0], [0, 1, 0, 1]]


def test_read_case_table_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg validate"):
        read_case_table(tmp_path / "cases_fixed_2023-24.csv")
