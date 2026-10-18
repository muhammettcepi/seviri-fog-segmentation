"""Physics-based fog / low-stratus labels from night-time SEVIRI brightness temperatures."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

NO_FOG = 0
FOG = 1
IGNORE = 255


@dataclass(frozen=True)
class LabelParams:
    """Thresholds of the labelling rules; temperatures and differences in K, angles in degrees."""

    night_min_sza: float
    btd_fixed: float
    warm_min_bt108: float
    cirrus_max_split: float
    smooth_max_std: float
    uncertainty_band: float
    min_blob_pixels: int

    @classmethod
    def from_config(cls, labels_cfg: dict) -> LabelParams:
        """Read the thresholds from the `labels:` config section, ignoring its other keys."""
        field_names = [field.name for field in dataclasses.fields(cls)]
        return cls(**{name: labels_cfg[name] for name in field_names})


def trusted_mask(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    sza: np.ndarray,
    night_min_sza: float,
) -> np.ndarray:
    """Pixels that are dark enough (sza above the limit) and have all three channels present."""
    is_night = sza > night_min_sza
    has_all_channels = np.isfinite(ir039) & np.isfinite(ir108) & np.isfinite(ir120)
    return is_night & has_all_channels


def histogram_pixels(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    sza: np.ndarray,
    params: LabelParams,
) -> np.ndarray:
    """BTD (IR_108 - IR_039) of trusted, warm, non-cirrus pixels, as a 1-D array."""
    trusted = trusted_mask(ir039, ir108, ir120, sza, params.night_min_sza)
    keep = trusted & warm_and_not_cirrus(ir108, ir120, params)
    return (ir108 - ir039)[keep]


def warm_and_not_cirrus(ir108: np.ndarray, ir120: np.ndarray, params: LabelParams) -> np.ndarray:
    """Pixels warm enough to be a low cloud top and without a thin-cirrus split-window signature."""
    is_warm = ir108 >= params.warm_min_bt108
    is_thin_cirrus = (ir108 - ir120) >= params.cirrus_max_split
    return is_warm & ~is_thin_cirrus


def local_std(field: np.ndarray, size: int = 3) -> np.ndarray:
    """Standard deviation in a size x size window, ignoring NaN; NaN where a window has no valid value."""
    is_valid = np.isfinite(field)
    if not is_valid.any():
        return np.full(field.shape, np.nan)

    # Subtracting the global mean keeps E[x^2] - E[x]^2 from losing precision on values near 270 K.
    centred = np.where(is_valid, field - field[is_valid].mean(), 0.0)
    valid_fraction = ndimage.uniform_filter(is_valid.astype(float), size=size, mode="constant")
    mean = ndimage.uniform_filter(centred, size=size, mode="constant")
    mean_of_squares = ndimage.uniform_filter(centred**2, size=size, mode="constant")

    # A window with no valid pixel has a fraction of exactly zero; the half-pixel margin guards against
    # rounding noise in the filter.
    has_valid = valid_fraction > 0.5 / size**2
    safe_fraction = np.where(has_valid, valid_fraction, 1.0)
    variance = mean_of_squares / safe_fraction - (mean / safe_fraction) ** 2
    std = np.sqrt(np.clip(variance, 0.0, None))
    return np.where(has_valid, std, np.nan)


def fog_candidates(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    btd_threshold: float,
    params: LabelParams,
) -> np.ndarray:
    """Pixels passing all four fog tests: high BTD, warm top, no thin cirrus, smooth IR_108 texture."""
    has_high_btd = (ir108 - ir039) > btd_threshold
    is_smooth = local_std(ir108) < params.smooth_max_std
    return has_high_btd & warm_and_not_cirrus(ir108, ir120, params) & is_smooth


def small_blobs(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Pixels of 8-connected groups in `mask` that hold fewer than `min_pixels` pixels."""
    eight_connected = np.ones((3, 3), dtype=int)
    group_ids, _ = ndimage.label(mask, structure=eight_connected)
    group_sizes = np.bincount(group_ids.ravel())
    is_small_group = group_sizes < min_pixels
    is_small_group[0] = False  # group 0 is the background
    return is_small_group[group_ids]


def label_scene(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    sza: np.ndarray,
    btd_threshold: float,
    params: LabelParams,
) -> np.ndarray:
    """Label one scene: 0 not fog, 1 fog, 255 where the pixel is untrusted or the call is too uncertain."""
    trusted = trusted_mask(ir039, ir108, ir120, sza, params.night_min_sza)
    candidates = fog_candidates(ir039, ir108, ir120, btd_threshold, params) & trusted

    labels = np.full(ir108.shape, IGNORE, dtype=np.uint8)
    labels[trusted] = NO_FOG
    labels[candidates] = FOG

    btd = ir108 - ir039
    near_threshold = np.abs(btd - btd_threshold) < params.uncertainty_band
    labels[candidates & near_threshold] = IGNORE

    isolated = small_blobs(labels == FOG, params.min_blob_pixels)
    labels[isolated] = IGNORE
    return labels
