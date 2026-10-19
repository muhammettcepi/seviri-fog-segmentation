"""Per-scene BTD threshold: the clear-ground peak of the histogram plus a fixed offset.

On the training season low cloud is a shoulder on the clear-ground hump, not a second hump, so
there is rarely a valley to cut at. The clear-ground peak, however, is well defined and moves
from scene to scene; the fog / clear boundary lies a roughly constant offset above it.
"""

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
    offset: float
    min_pixels: int
    clamp: tuple[float, float]

    @classmethod
    def from_config(cls, dynamic_cfg: dict) -> DynamicParams:
        """Read the `labels.dynamic` config block; YAML lists become tuples."""
        return cls(
            bin_width=dynamic_cfg["bin_width"],
            bin_range=tuple(dynamic_cfg["bin_range"]),
            smoothing_bins=dynamic_cfg["smoothing_bins"],
            clear_window=tuple(dynamic_cfg["clear_window"]),
            offset=dynamic_cfg["offset"],
            min_pixels=dynamic_cfg["min_pixels"],
            clamp=tuple(dynamic_cfg["clamp"]),
        )


@dataclass(frozen=True)
class ThresholdChoice:
    """A BTD threshold and where it came from: "fixed", "dynamic" or "fallback"."""

    value: float
    source: str


def smooth_counts(counts: np.ndarray, smoothing_bins: int) -> np.ndarray:
    """Moving window over the histogram, keeping its length.

    The window is summed, not divided by its width: only the order of the bins matters here,
    and the sums stay integers.
    """
    window = np.ones(smoothing_bins)
    return np.convolve(counts, window, mode="same")


def bins_in_window(centres: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Boolean mask of the bins whose centre lies in [low, high)."""
    low, high = window
    return (centres >= low) & (centres < high)


def peak_in_window(
    samples: np.ndarray,
    window: tuple[float, float],
    bin_width: float,
    bin_range: tuple[float, float],
    smoothing_bins: int,
) -> float:
    """BTD (bin centre) of the highest smoothed-histogram bin inside `window`; NaN if there is no real peak.

    The histogram spans `bin_range`, which should reach beyond the window, so the smoothing at
    the window edge still sees the pixels next to it. The peak is not real when its bin is the
    first or last one inside the window: the hump then continues outside the window (or there
    is none). `samples` must be finite.
    """
    n_bins = round((bin_range[1] - bin_range[0]) / bin_width)
    counts, edges = np.histogram(samples, bins=n_bins, range=bin_range)
    centres = (edges[:-1] + edges[1:]) / 2
    smoothed = smooth_counts(counts, smoothing_bins)

    candidates = np.flatnonzero(bins_in_window(centres, window))
    top = int(np.argmax(smoothed[candidates]))
    if top in (0, len(candidates) - 1):
        return float("nan")
    return float(centres[candidates[top]])


def clear_ground_peak(samples: np.ndarray, params: DynamicParams) -> float:
    """BTD of the clear-ground hump of a scene; NaN if the hump is not inside `clear_window`."""
    return peak_in_window(
        samples, params.clear_window, params.bin_width, params.bin_range, params.smoothing_bins
    )


def dynamic_threshold(btd: np.ndarray, fixed: float, params: DynamicParams) -> ThresholdChoice:
    """Threshold `offset` above the clear-ground peak, or `fixed` if there is no usable peak."""
    fallback = ThresholdChoice(float(fixed), "fallback")

    samples = btd[np.isfinite(btd)]
    if samples.size < params.min_pixels:
        return fallback

    peak = clear_ground_peak(samples, params)
    if np.isnan(peak):
        return fallback

    clamp_low, clamp_high = params.clamp
    value = float(np.clip(peak + params.offset, clamp_low, clamp_high))
    return ThresholdChoice(value, "dynamic")
