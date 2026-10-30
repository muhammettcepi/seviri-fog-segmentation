"""Tests for the contingency scores and the bootstrap over nights."""

from __future__ import annotations

import warnings

import numpy as np

from fogseg.scores import (
    COUNT_NAMES,
    bootstrap_skill,
    contingency,
    counts_by_night,
    paired_csi_difference,
    score_summary,
    skill,
)


def test_contingency_counts_and_skips_ignore():
    label = np.array([1, 1, 0, 0, 255, 1], dtype=np.uint8)
    truth = np.array([1, 0, 1, 0, 1, -1], dtype=np.int8)

    assert contingency(label, truth).tolist() == [1, 1, 1, 1]


def test_skill_values():
    scores = skill(np.array([6, 2, 3, 89]))

    assert scores["pod"] == 0.75
    assert np.isclose(scores["far"], 1 / 3)
    assert np.isclose(scores["csi"], 6 / 11)
    assert np.isclose(scores["bias"], 9 / 8)


def test_skill_without_fog_cases_is_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = skill(np.array([0, 0, 0, 10]))

    assert all(np.isnan(value) for value in scores.values())


def test_skill_works_on_the_last_axis():
    scores = skill(np.array([[6, 2, 3, 89], [0, 0, 0, 10]]))

    assert scores["pod"].shape == (2,)
    assert scores["pod"][0] == 0.75
    assert np.isnan(scores["pod"][1])


def test_counts_by_night_keeps_night_order():
    night = np.array(["b", "b", "b"])
    label = np.array([1, 0, 1], dtype=np.uint8)
    truth = np.array([1, 0, 0], dtype=np.int8)

    counts = counts_by_night(night, label, truth, np.array(["a", "b"]))

    assert counts.tolist() == [[0, 0, 0, 0], [1, 0, 1, 1]]


def test_counts_by_night_without_nights_has_integer_dtype():
    counts = counts_by_night(np.array([]), np.array([]), np.array([]), np.array([]))

    assert counts.shape == (0, 4)
    assert np.issubdtype(counts.dtype, np.integer)


def synthetic_night_counts() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 20, size=(40, 4))


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    night_counts = synthetic_night_counts()

    first = bootstrap_skill(night_counts, n_samples=300, seed=0)
    second = bootstrap_skill(night_counts, n_samples=300, seed=0)

    assert first == second
    low, high = first["csi"]
    assert low <= skill(night_counts.sum(axis=0))["csi"] <= high


def test_bootstrap_resamples_whole_nights():
    night_counts = np.array([[3, 0, 0, 0], [0, 3, 0, 0]])

    intervals = bootstrap_skill(night_counts, n_samples=200, seed=0)

    assert set(intervals["pod"]) <= {0.0, 0.5, 1.0}


def test_bootstrap_without_fog_cases_gives_nan_bounds_without_warning():
    night_counts = np.array([[0, 0, 0, 5], [0, 0, 0, 7]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        intervals = bootstrap_skill(night_counts, n_samples=50, seed=0)

    assert all(np.isnan(bound) for interval in intervals.values() for bound in interval)


def test_paired_difference_of_identical_counts_is_zero():
    night_counts = synthetic_night_counts()

    assert paired_csi_difference(night_counts, night_counts, n_samples=200, seed=0) == (0.0, 0.0)


def test_score_summary_columns_follow_the_requested_intervals():
    night_counts = synthetic_night_counts()

    full = score_summary(night_counts, n_samples=50, seed=0)
    csi_only = score_summary(night_counts, n_samples=50, seed=0, interval_scores=("csi",))

    counts = ["n_cases", *COUNT_NAMES]
    assert list(full) == [
        *counts,
        *("pod", "pod_low", "pod_high", "far", "far_low", "far_high", "csi", "csi_low", "csi_high", "bias"),
    ]
    assert list(csi_only) == [*counts, "pod", "far", "csi", "csi_low", "csi_high", "bias"]
    assert full["n_cases"] == night_counts.sum()
    assert full["csi"] == csi_only["csi"]
    assert (full["csi_low"], full["csi_high"]) == (csi_only["csi_low"], csi_only["csi_high"])


def test_score_summary_without_nights_is_empty_with_nan_intervals():
    no_nights = counts_by_night(np.array([]), np.array([]), np.array([]), np.array([]))

    summary = score_summary(no_nights, n_samples=50, seed=0)

    assert summary["n_cases"] == 0
    assert all(isinstance(summary[name], int) for name in ("n_cases", *COUNT_NAMES))
    assert all(np.isnan(value) for name, value in summary.items() if name not in ("n_cases", *COUNT_NAMES))


def test_paired_difference_detects_a_better_system():
    night_counts = synthetic_night_counts()
    better = night_counts.copy()
    better[:, 0] += 20

    _, high = paired_csi_difference(night_counts, better, n_samples=200, seed=0)

    assert high < 0
