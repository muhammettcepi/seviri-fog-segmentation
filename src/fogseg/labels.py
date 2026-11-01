"""Physics-based fog / low-stratus labels from night-time SEVIRI brightness temperatures."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

NO_FOG = 0
FOG = 1
OBSCURED = 2
IGNORE = 255

TEXTURES = ("ir108_std3", "btd_std3", "ir108_mad5", "blob", "none")
FOG_INDICES = ("btd", "emissivity")
SPLIT_WINDOWS = ("fixed", "kma")

# The IR_108 - IR_120 difference of a clear or low-cloud pixel grows with temperature along this line.
KMA_INTERCEPT = -37.48
KMA_SLOPE = 0.1329
KMA_HALFWIDTH_K = 1.0

WAVELENGTH_39_UM = 3.92
PLANCK_SECOND_CONSTANT_UM_K = 14387.77


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
    texture: str = "ir108_std3"
    fog_index: str = "btd"
    emissivity_max: float | None = None
    split_window: str = "fixed"
    temporal_relax: float = 0.0
    temporal_max_dt108: float = 2.0

    def __post_init__(self) -> None:
        options = {"texture": TEXTURES, "fog_index": FOG_INDICES, "split_window": SPLIT_WINDOWS}
        for name, allowed in options.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}, not {getattr(self, name)!r}")

    @classmethod
    def from_config(cls, labels_cfg: dict) -> LabelParams:
        """Read the thresholds from the `labels:` config section, ignoring its other keys.

        A key that is missing (an old config or label store) takes the default of its field.
        """
        field_names = [field.name for field in dataclasses.fields(cls)]
        return cls(**{name: labels_cfg[name] for name in field_names if name in labels_cfg})


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
    """Pixels warm enough to be a low cloud top and without a thin-cirrus split-window signature.

    The `fixed` split window calls a difference of `cirrus_max_split` or more cirrus; the `kma` one calls
    cirrus any difference farther than `KMA_HALFWIDTH_K` from the line `KMA_INTERCEPT + KMA_SLOPE * ir108`.
    """
    is_warm = ir108 >= params.warm_min_bt108
    split = ir108 - ir120
    if params.split_window == "kma":
        is_thin_cirrus = np.abs(split - (KMA_INTERCEPT + KMA_SLOPE * ir108)) > KMA_HALFWIDTH_K
    else:
        is_thin_cirrus = split >= params.cirrus_max_split
    return is_warm & ~is_thin_cirrus


def planck_39(bt: np.ndarray) -> np.ndarray:
    """Planck radiance at 3.9 µm of a black body at `bt` K, up to a constant factor."""
    return 1.0 / np.expm1(PLANCK_SECOND_CONSTANT_UM_K / (WAVELENGTH_39_UM * bt))


def emissivity_39(ir039: np.ndarray, ir108: np.ndarray) -> np.ndarray:
    """Effective 3.9 µm emissivity: the radiance relative to a black body at the IR_108 temperature."""
    return planck_39(ir039) / planck_39(ir108)


def emissivity_btd_threshold(ir108: np.ndarray, emissivity_max: float) -> np.ndarray:
    """The BTD (IR_108 - IR_039) at which a pixel with this IR_108 has the emissivity `emissivity_max`."""
    radiance = emissivity_max * planck_39(ir108)
    ir039 = PLANCK_SECOND_CONSTANT_UM_K / (WAVELENGTH_39_UM * np.log1p(1.0 / radiance))
    return ir108 - ir039


def btd_threshold_image(ir108: np.ndarray, threshold: float, params: LabelParams) -> float | np.ndarray:
    """The BTD threshold of each pixel: `threshold` in K, or the BTD that an emissivity `threshold` means."""
    if params.fog_index == "emissivity":
        return emissivity_btd_threshold(ir108, threshold)
    return threshold


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


def local_mad(field: np.ndarray, size: int = 5) -> np.ndarray:
    """Median absolute deviation in a size x size window; NaN wherever a missing value reaches the result."""
    is_missing = np.isnan(field)
    filled = np.where(is_missing, 0.0, field)
    deviation = np.abs(filled - ndimage.median_filter(filled, size=size, mode="nearest"))
    mad = ndimage.median_filter(deviation, size=size, mode="nearest")
    # The deviations of the neighbours depend on their own windows, so the missing values reach twice as far.
    is_reached = ndimage.maximum_filter(is_missing, size=2 * size - 1, mode="nearest")
    return np.where(is_reached, np.nan, mad)


def texture_ok(ir108: np.ndarray, btd: np.ndarray, region: np.ndarray, params: LabelParams) -> np.ndarray:
    """Pixels that pass the smoothness test chosen by `params.texture`.

    Only the `blob` test looks at `region`: it judges each 8-connected group of it by its median IR_108
    spread, and pixels outside the region fail.
    """
    limit = params.smooth_max_std
    if params.texture == "ir108_std3":
        return local_std(ir108) < limit
    if params.texture == "btd_std3":
        return local_std(btd) < limit
    if params.texture == "ir108_mad5":
        return local_mad(ir108) < limit
    if params.texture == "none":
        return np.ones(ir108.shape, dtype=bool)
    if params.texture == "blob":
        group_ids, group_count = ndimage.label(region, structure=np.ones((3, 3), dtype=int))
        group_spread = ndimage.median(local_std(ir108), group_ids, np.arange(1, group_count + 1))
        is_smooth_group = np.concatenate([[False], np.asarray(group_spread) < limit])
        return is_smooth_group[group_ids]
    raise ValueError(f"unknown texture {params.texture!r}")


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
    threshold: float,
    params: LabelParams,
) -> np.ndarray:
    """Label one scene: 0 not fog, 1 fog, 2 obscured, 255 untrusted or too uncertain to call.

    `threshold` is in the unit of `params.fog_index`: K of BTD, or an emissivity.

    A trusted pixel that is not warm or shows thin cirrus is obscured: fog may hide under that cloud.
    Among the others, a pixel that passes the smooth-top test is uncertain when its BTD lies within
    `uncertainty_band` K of the threshold on either side; small fog blobs are then found among the
    remaining fog pixels.
    """
    trusted = trusted_mask(ir039, ir108, ir120, sza, params.night_min_sza)
    is_candidate = trusted & warm_and_not_cirrus(ir108, ir120, params)
    is_obscured = trusted & ~is_candidate

    btd = ir108 - ir039
    margin = btd - btd_threshold_image(ir108, threshold, params)
    region = is_candidate & (margin > -params.uncertainty_band)
    fog_like = is_candidate & texture_ok(ir108, btd, region, params)

    is_uncertain = fog_like & (np.abs(margin) < params.uncertainty_band)
    is_fog = fog_like & (margin > 0) & ~is_uncertain

    labels = np.full(ir108.shape, IGNORE, dtype=np.uint8)
    labels[trusted] = NO_FOG
    labels[is_obscured] = OBSCURED
    labels[is_fog] = FOG
    labels[is_uncertain] = IGNORE

    isolated = small_blobs(labels == FOG, params.min_blob_pixels)
    labels[isolated] = IGNORE
    return labels


def label_night(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    sza: np.ndarray,
    thresholds: np.ndarray,
    params: LabelParams,
) -> np.ndarray:
    """Label the stored scenes of one night, shape (n_scenes, y, x), in time order.

    Each scene gets its `label_scene` labels. With `params.temporal_relax` above zero, a not-fog or ignored
    pixel that passes the smooth-top test and whose BTD is above the threshold or within `temporal_relax` K
    below it (ignored pixels above the threshold count too) becomes fog, once and without iterating, when
    the same pixel is fog in the previous or next stored scene and its IR_108 changed by less than
    `params.temporal_max_dt108`. Obscured and untrusted pixels never change.
    """
    n_scenes = len(ir108)
    strict = np.stack(
        [label_scene(ir039[s], ir108[s], ir120[s], sza[s], thresholds[s], params) for s in range(n_scenes)]
    )
    if params.temporal_relax <= 0:
        return strict

    labels = strict.copy()
    for s in range(n_scenes):
        trusted = trusted_mask(ir039[s], ir108[s], ir120[s], sza[s], params.night_min_sza)
        is_candidate = trusted & warm_and_not_cirrus(ir108[s], ir120[s], params)
        btd = ir108[s] - ir039[s]
        margin = btd - btd_threshold_image(ir108[s], thresholds[s], params)
        region = is_candidate & (margin > -max(params.uncertainty_band, params.temporal_relax))
        is_smooth = texture_ok(ir108[s], btd, region, params)
        near_miss = is_candidate & is_smooth & (margin > -params.temporal_relax)

        next_to_fog = np.zeros(ir108[s].shape, dtype=bool)
        for neighbour in (s - 1, s + 1):
            if 0 <= neighbour < n_scenes:
                steady_top = np.abs(ir108[s] - ir108[neighbour]) < params.temporal_max_dt108
                next_to_fog |= (strict[neighbour] == FOG) & steady_top

        labels[s, near_miss & next_to_fog & np.isin(strict[s], (NO_FOG, IGNORE))] = FOG
    return labels
