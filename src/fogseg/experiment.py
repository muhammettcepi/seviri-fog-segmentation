"""Compare label variants at a matched false-alarm ratio: the value is chosen on training folds, scored on test folds."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fogseg.scores import COUNT_NAMES, contingency, counts_by_night, skill
from fogseg.splits import cv_folds, night_block_start
from fogseg.validation import TRUTH_NO, TRUTH_YES

log = logging.getLogger(__name__)

GRID_COLUMNS = ["value", "case", "label"]
SCORE_COLUMNS = ["value", *COUNT_NAMES, "pod", "far", "csi", "n_cases"]
CV_COLUMNS = ["fold", "value", "matched", *SCORE_COLUMNS[1:]]


def scored_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """The cases that are scored: not hidden and with a defined truth, numbered `case` 0..n-1 in a fresh index."""
    is_scored = ~cases["hidden"] & cases["truth_fls"].isin((TRUTH_NO, TRUTH_YES))
    scored = cases[is_scored].reset_index(drop=True)
    return scored.assign(case=np.arange(len(scored)))


def score_columns(counts: np.ndarray) -> dict:
    """The counts of one row with their POD, FAR, CSI and case count, as in `SCORE_COLUMNS` without the value."""
    scores = skill(counts)
    return {
        **dict(zip(COUNT_NAMES, counts.tolist(), strict=True)),
        "pod": float(scores["pod"]),
        "far": float(scores["far"]),
        "csi": float(scores["csi"]),
        "n_cases": int(counts.sum()),
    }


def value_scores(grid: pd.DataFrame, cases: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Full-count counts and scores of every grid value over the cases where `mask` is true."""
    chosen = grid[mask.to_numpy()[grid["case"].to_numpy()]]
    label = chosen["label"].to_numpy()
    truth = cases["truth_fls"].to_numpy()[chosen["case"].to_numpy()]
    rows = []
    for value in np.sort(grid["value"].unique()):
        at_value = (chosen["value"] == value).to_numpy()
        counts = contingency(label[at_value], truth[at_value], full_count=True)
        rows.append({"value": value, **score_columns(counts)})
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def matched_value(scores: pd.DataFrame, far_target: float) -> tuple[float, bool]:
    """The value with the highest FAR not above the target, and whether there is one.

    Ties go to the higher POD, then to the lower value. If every FAR is above the target, the value with the
    lowest FAR is returned with False. A value that labels no fog has no FAR and is never chosen.
    """
    defined = scores.dropna(subset=["far"])
    if defined.empty:
        raise ValueError("No value labels any fog in the cases the value is chosen on")
    within_target = defined[defined["far"] <= far_target]
    qualifies = not within_target.empty
    candidates = within_target if qualifies else defined
    ranked = candidates.sort_values(["far", "pod", "value"], ascending=[not qualifies, False, True])
    return float(ranked["value"].iloc[0]), qualifies


def cross_validate(grid: pd.DataFrame, cases: pd.DataFrame, n_folds: int, far_target: float) -> pd.DataFrame:
    """Per fold the value matched on the training part and its scores on the test part, then the pooled test folds.

    Cases of the held-out fold are in no training or test part.
    """
    rows, test_counts = [], []
    for fold, train, test in cv_folds(cases, n_folds):
        value, matched = matched_value(value_scores(grid, cases, train), far_target)
        if not matched:
            log.warning(
                "Fold %d: no value reaches a FAR of %.3f on its training part; using %g",
                fold,
                far_target,
                value,
            )
        at_value = grid[grid["value"] == value]
        scores = value_scores(at_value, cases, test).iloc[0]
        test_counts.append(scores[list(COUNT_NAMES)].to_numpy(dtype=int))
        rows.append({"fold": fold, "matched": matched, **scores.to_dict()})
    pooled = {
        "fold": "pooled",
        "value": np.nan,
        "matched": all(row["matched"] for row in rows),
        **score_columns(np.sum(test_counts, axis=0)),
    }
    return pd.DataFrame([*rows, pooled], columns=CV_COLUMNS)


def block_counts(
    grid: pd.DataFrame,
    cases: pd.DataFrame,
    value: float,
    mask: pd.Series,
    season_start: np.datetime64,
    block_days: int,
) -> np.ndarray:
    """Full-count counts of one value per night block over the cases where `mask` is true, shape (n_blocks, 4)."""
    in_mask = mask.to_numpy()
    block = night_block_start(cases["night"].to_numpy(), season_start, block_days)
    at_value = grid[grid["value"] == value]
    selected = at_value[in_mask[at_value["case"].to_numpy()]]
    in_case = selected["case"].to_numpy()
    return counts_by_night(
        block[in_case],
        selected["label"].to_numpy(),
        cases["truth_fls"].to_numpy()[in_case],
        np.unique(block[in_mask]),
        full_count=True,
    )
