"""Physics-based fog / low-stratus labels from night-time SEVIRI brightness temperatures."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

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
    is_warm = ir108 >= params.warm_min_bt108
    is_thin_cirrus = (ir108 - ir120) >= params.cirrus_max_split
    keep = trusted & is_warm & ~is_thin_cirrus
    return (ir108 - ir039)[keep]
