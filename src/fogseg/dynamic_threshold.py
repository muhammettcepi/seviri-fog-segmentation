"""Per-scene BTD threshold from the valley between the clear-ground and low-cloud histogram humps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicParams:
    """Histogram and plausibility settings of the dynamic threshold; BTD values in K."""

    bin_width: float
    bin_range: tuple[float, float]
    smoothing_bins: int
    clear_window: tuple[float, float]
    cloud_window: tuple[float, float]
    min_pixels: int
    min_hump_share: float
    max_valley_ratio: float
    clamp: tuple[float, float]

    @classmethod
    def from_config(cls, dynamic_cfg: dict) -> DynamicParams:
        """Read the `labels.dynamic` config block; YAML lists become tuples."""
        return cls(
            bin_width=dynamic_cfg["bin_width"],
            bin_range=tuple(dynamic_cfg["bin_range"]),
            smoothing_bins=dynamic_cfg["smoothing_bins"],
            clear_window=tuple(dynamic_cfg["clear_window"]),
            cloud_window=tuple(dynamic_cfg["cloud_window"]),
            min_pixels=dynamic_cfg["min_pixels"],
            min_hump_share=dynamic_cfg["min_hump_share"],
            max_valley_ratio=dynamic_cfg["max_valley_ratio"],
            clamp=tuple(dynamic_cfg["clamp"]),
        )


@dataclass(frozen=True)
class ThresholdChoice:
    """A BTD threshold and where it came from: "fixed", "dynamic" or "fallback"."""

    value: float
    source: str


def smooth_counts(counts: np.ndarray, smoothing_bins: int) -> np.ndarray:
    """Moving window over the histogram, keeping its length.

    The window is summed, not divided by its width: the scale cancels in every comparison
    here, and integer sums keep equal bins exactly equal, which the flat-valley rule needs.
    """
    window = np.ones(smoothing_bins)
    return np.convolve(counts, window, mode="same")


def bins_in_window(centres: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Boolean mask of the bins whose centre lies in [low, high)."""
    low, high = window
    return (centres >= low) & (centres < high)


def highest_bin_in(smoothed: np.ndarray, in_window: np.ndarray) -> int:
    """Index of the highest smoothed bin among the masked bins."""
    candidates = np.flatnonzero(in_window)
    return int(candidates[np.argmax(smoothed[candidates])])


def middle_of_lowest_bins(smoothed_between: np.ndarray) -> int:
    """Index of the middle bin among all bins sharing the minimum (a flat valley bottom)."""
    lowest = np.flatnonzero(smoothed_between == smoothed_between.min())
    return int(lowest[len(lowest) // 2])


def dynamic_threshold(btd: np.ndarray, fixed: float, params: DynamicParams) -> ThresholdChoice:
    """Threshold at the histogram valley between two humps, or `fixed` if there are not two."""
    fallback = ThresholdChoice(float(fixed), "fallback")

    samples = btd[np.isfinite(btd)]
    if samples.size < params.min_pixels:
        return fallback

    low, high = params.bin_range
    bin_count = round((high - low) / params.bin_width)
    counts, edges = np.histogram(samples, bins=bin_count, range=params.bin_range)
    centres = (edges[:-1] + edges[1:]) / 2
    smoothed = smooth_counts(counts, params.smoothing_bins)

    in_clear = bins_in_window(centres, params.clear_window)
    in_cloud = bins_in_window(centres, params.cloud_window)
    clear_share = counts[in_clear].sum() / samples.size
    cloud_share = counts[in_cloud].sum() / samples.size
    if min(clear_share, cloud_share) < params.min_hump_share:
        return fallback

    clear_peak = highest_bin_in(smoothed, in_clear)
    cloud_peak = highest_bin_in(smoothed, in_cloud)
    if cloud_peak - clear_peak < 2:
        return fallback

    between = smoothed[clear_peak + 1 : cloud_peak]
    valley = clear_peak + 1 + middle_of_lowest_bins(between)

    smaller_peak = min(smoothed[clear_peak], smoothed[cloud_peak])
    if smoothed[valley] > params.max_valley_ratio * smaller_peak:
        return fallback

    clamp_low, clamp_high = params.clamp
    value = float(np.clip(centres[valley], clamp_low, clamp_high))
    return ThresholdChoice(value, "dynamic")
