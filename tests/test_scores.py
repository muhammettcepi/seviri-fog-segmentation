"""Tests for the contingency scores and the bootstrap over nights."""

from __future__ import annotations

import warnings

import numpy as np

from fogseg.scores import bootstrap_skill, contingency, counts_by_night, paired_csi_difference, skill


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


def test_paired_difference_uses_the_same_nights_for_both_systems():
    night_counts = synthetic_night_counts()
    better = night_counts.copy()
    better[:, 0] += 20

    _, high = paired_csi_difference(night_counts, better, n_samples=200, seed=0)

    assert high < 0
