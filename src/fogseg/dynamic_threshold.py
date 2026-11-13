"""Per-scene BTD threshold: the clear-ground peak of the histogram plus a fixed offset.

On the training season low cloud is a shoulder on the clear-ground hump, not a second hump, so
there is rarely a valley to cut at. The clear-ground peak, however, is well defined and moves
from scene to scene; the fog / clear boundary lies a roughly constant offset above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_WINDOW_BINS = 3  # the peak must not sit in the first or last bin of the window, so 3 bins is the least
BIN_COUNT_TOLERANCE = 1e-6  # float error of (range / width), e.g. 10 / 0.1 = 100.00000000000001


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

    def __post_init__(self) -> None:
        """Reject settings that would make the peak search meaningless."""
        n_bins = self.check_bins()
        centres = self.bin_range[0] + self.bin_width * (np.arange(n_bins) + 0.5)
        n_window_bins = int(bins_in_window(centres, self.clear_window).sum())
        if n_window_bins < MIN_WINDOW_BINS:
            raise ValueError(
                f"clear_window {self.clear_window} must hold at least {MIN_WINDOW_BINS} bins of "
                f"bin_range {self.bin_range}, it holds {n_window_bins}"
            )
        if self.smoothing_bins < 1 or self.smoothing_bins % 2 == 0:
            raise ValueError(f"smoothing_bins must be odd and at least 1, got {self.smoothing_bins}")
        if self.clamp[0] > self.clamp[1]:
            raise ValueError(f"clamp must be (low, high) with low <= high, got {self.clamp}")

    def check_bins(self) -> int:
        """Number of histogram bins; bin_width must be positive and fit a whole number of times into bin_range."""
        low, high = self.bin_range
        n_bins = (high - low) / self.bin_width if self.bin_width > 0 else float("nan")
        if not np.isfinite(n_bins) or n_bins < 1 or abs(n_bins - round(n_bins)) > BIN_COUNT_TOLERANCE:
            raise ValueError(
                f"bin_width {self.bin_width} must be positive and divide bin_range {self.bin_range}"
            )
        return round(n_bins)

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


def usable_peak(btd: np.ndarray, params: DynamicParams) -> float:
    """Clear-ground peak of a scene's BTD values; NaN if there are too few of them or no real peak."""
    # Only pixels inside bin_range reach the histogram (the upper edge belongs to the last bin), so
    # only they count towards min_pixels.
    low, high = params.bin_range
    samples = btd[np.isfinite(btd) & (btd >= low) & (btd <= high)]
    if samples.size < params.min_pixels:
        return float("nan")
    return clear_ground_peak(samples, params)


def scene_threshold(peak: float, offset: float, fallback: float, clamp: tuple[float, float]) -> float:
    """`offset` above the clear-ground `peak`, kept within `clamp`; `fallback` if there is no peak."""
    if np.isnan(peak):
        return float(fallback)
    return float(np.clip(peak + offset, *clamp))


def dynamic_threshold(btd: np.ndarray, fixed: float, params: DynamicParams) -> ThresholdChoice:
    """Threshold `offset` above the clear-ground peak, or `fixed` if there is no usable peak."""
    peak = usable_peak(btd, params)
    source = "fallback" if np.isnan(peak) else "dynamic"
    return ThresholdChoice(scene_threshold(peak, params.offset, fixed, params.clamp), source)
