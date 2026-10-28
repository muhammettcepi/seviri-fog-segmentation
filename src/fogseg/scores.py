"""Contingency scores of satellite fog labels against station truth, with a bootstrap over nights."""

from __future__ import annotations

import numpy as np

COUNT_NAMES = ("hits", "misses", "false_alarms", "correct_negatives")
INTERVAL_PERCENTILES = (2.5, 97.5)


def contingency(label: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Counts in `COUNT_NAMES` order; cases with an ignored label (255) or undefined truth (-1) are skipped."""
    is_scored = np.isin(label, (0, 1)) & np.isin(truth, (0, 1))
    is_fog = label[is_scored] == 1
    is_fls = truth[is_scored] == 1
    return np.array(
        [
            np.sum(is_fog & is_fls),
            np.sum(~is_fog & is_fls),
            np.sum(is_fog & ~is_fls),
            np.sum(~is_fog & ~is_fls),
        ]
    )


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division, NaN where the denominator is 0."""
    out = np.full(np.shape(denominator), np.nan)
    return np.divide(numerator, denominator, out=out, where=denominator > 0)


def skill(counts: np.ndarray) -> dict[str, float]:
    """POD, FAR, CSI and frequency bias from counts in `COUNT_NAMES` order, along the last axis.

    A score is NaN where its denominator is 0, e.g. when a night or sample has no fog at all.
    """
    hits, misses, false_alarms, _ = np.moveaxis(np.asarray(counts), -1, 0)
    return {
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "bias": ratio(hits + false_alarms, hits + misses),
    }


def counts_by_night(
    night: np.ndarray, label: np.ndarray, truth: np.ndarray, nights: np.ndarray
) -> np.ndarray:
    """Counts per night as an array of shape (len(nights), 4), rows in the order of `nights`."""
    return np.array([contingency(label[night == one], truth[night == one]) for one in nights]).reshape(-1, 4)


def draw_nights(n_nights: int, n_samples: int, seed: int) -> np.ndarray:
    """Night indices of shape (n_samples, n_nights), drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n_nights, size=(n_samples, n_nights))


def interval(samples: np.ndarray) -> tuple[float, float]:
    """95 % interval of the bootstrap samples, ignoring NaN; both bounds NaN if every sample is NaN."""
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(valid, INTERVAL_PERCENTILES)
    return float(low), float(high)


def bootstrap_skill(night_counts: np.ndarray, n_samples: int, seed: int) -> dict[str, tuple[float, float]]:
    """95 % interval of each score from resampling whole nights, as cases on one night are not independent."""
    drawn = draw_nights(len(night_counts), n_samples, seed)
    sample_scores = skill(night_counts[drawn].sum(axis=1))
    return {name: interval(values) for name, values in sample_scores.items()}


def paired_csi_difference(
    counts_a: np.ndarray, counts_b: np.ndarray, n_samples: int, seed: int
) -> tuple[float, float]:
    """95 % interval of CSI(a) - CSI(b); both are scored on the same drawn nights, so night effects cancel."""
    drawn = draw_nights(len(counts_a), n_samples, seed)
    csi_a = skill(counts_a[drawn].sum(axis=1))["csi"]
    csi_b = skill(counts_b[drawn].sum(axis=1))["csi"]
    return interval(csi_a - csi_b)
