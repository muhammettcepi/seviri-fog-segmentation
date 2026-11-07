"""Model inputs, training targets, scene folds and station points of the U-Net, as numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.features import CHANNELS
from fogseg.labels import LabelParams, trusted_mask, warm_and_not_cirrus
from fogseg.splits import NIGHT_BLOCK, fold_of, night_block_start
from fogseg.teacher_labels import PROBABILITY_SCALE, UNDEFINED
from fogseg.validation import TRUTH_NO, TRUTH_YES, night_of, scenes_between, season_bounds

DIFFERENCES = [("IR_108", "IR_039"), ("IR_108", "IR_120"), ("IR_108", "WV_062")]
INPUT_CHANNELS = [
    *CHANNELS,
    *(f"{first}-{second}" for first, second in DIFFERENCES),
    "elevation",
    "sza",
    "trusted",
]
TARGETS = ("soft", "hard")


@dataclass(frozen=True)
class Normalization:
    """Mean and standard deviation of each standardized input channel."""

    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict) -> Normalization:
        return cls(np.array(values["mean"], dtype="float64"), np.array(values["std"], dtype="float64"))


def raw_inputs(
    bt: np.ndarray, sza: np.ndarray, elevation: np.ndarray, night_min_sza: float
) -> tuple[np.ndarray, np.ndarray]:
    """The 13 channels before standardization, `(n, 13, y, x)`, and the trusted mask `(n, y, x)`."""
    by_name = {name: bt[:, index] for index, name in enumerate(CHANNELS)}
    differences = [by_name[first] - by_name[second] for first, second in DIFFERENCES]
    elevation_stack = np.broadcast_to(elevation, sza.shape)
    raw = np.stack([*bt.transpose(1, 0, 2, 3), *differences, elevation_stack, sza], axis=1)
    trusted = trusted_mask(by_name["IR_039"], by_name["IR_108"], by_name["IR_120"], sza, night_min_sza)
    return raw.astype("float32"), trusted


def fit_normalization(raw: np.ndarray, trusted: np.ndarray) -> Normalization:
    """Mean and standard deviation of each channel over the trusted, finite pixels; a flat channel gets 1."""
    mean, std = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    for channel in range(raw.shape[1]):
        values = raw[:, channel][trusted].astype("float64")
        values = values[np.isfinite(values)]
        if values.size:
            mean[channel] = values.mean()
            std[channel] = values.std() or 1.0
    return Normalization(mean, std)


def model_inputs(raw: np.ndarray, trusted: np.ndarray, normalization: Normalization) -> np.ndarray:
    """Standardized `(n, 14, y, x)` float32 channels, 0 where untrusted or not finite, then the trusted flag."""
    standardized = (raw - normalization.mean[None, :, None, None]) / normalization.std[None, :, None, None]
    inputs = np.where(trusted[:, None] & np.isfinite(raw), standardized, 0.0)
    return np.concatenate([inputs, trusted[:, None]], axis=1).astype("float32")


def obscured_mask(bt: np.ndarray, trusted: np.ndarray, params: LabelParams) -> np.ndarray:
    """Trusted pixels that are cold or show thin cirrus: the rule's obscured class."""
    ir108, ir120 = bt[:, CHANNELS.index("IR_108")], bt[:, CHANNELS.index("IR_120")]
    return trusted & ~warm_and_not_cirrus(ir108, ir120, params)


def targets(
    label: np.ndarray, fog_probability: np.ndarray | None, target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Training target and 0 / 1 weight: the teacher probability (soft) or the 0 / 1 label (hard)."""
    if target == "hard":
        return label.astype("float32"), np.isin(label, (0, 1)).astype("float32")
    if fog_probability is None:
        raise ValueError(
            "the soft target needs a label store with fog_probability, such as labels_teacher.zarr"
        )
    defined = fog_probability != UNDEFINED
    return (fog_probability / PROBABILITY_SCALE).astype("float32"), defined.astype("float32")


def tune_season_scenes(times: np.ndarray, cfg: dict) -> np.ndarray:
    """Indices of the scenes of the tuning season."""
    season = cfg["stations"]["tune_season"]
    start, end = season_bounds(cfg, season)
    return scenes_between(times, start, end, season)


def scene_night_folds(
    times: np.ndarray, splits: pd.DataFrame, season_start: np.datetime64, block_days: int
) -> np.ndarray:
    """The fold of the night block of each scene."""
    blocks = night_block_start(night_of(times), season_start, block_days)
    return fold_of(np.datetime_as_string(blocks, unit="D"), splits, NIGHT_BLOCK)


def point_targets(cases: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """The cases in a scene of `times` with a yes or no truth: scene position, pixel and 1 / 0 truth."""
    scene = pd.Index(times).get_indexer(pd.to_datetime(cases["scene_time"]))
    truth = cases["truth_fls"].to_numpy()
    keep = (scene >= 0) & np.isin(truth, (TRUTH_YES, TRUTH_NO))
    return pd.DataFrame(
        {
            "scene": scene[keep],
            "iy": cases["iy"].to_numpy()[keep],
            "ix": cases["ix"].to_numpy()[keep],
            "truth": (truth[keep] == TRUTH_YES).astype("float32"),
        }
    )


def read_inputs(
    data: xr.Dataset, scenes: np.ndarray, elevation: np.ndarray, params: LabelParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw channels, trusted mask and obscured mask of the dataset scenes `scenes`."""
    chosen = data.isel(time=scenes)
    bt = chosen["bt"].sel(channel=CHANNELS).values
    raw, trusted = raw_inputs(bt, chosen["sza"].values, elevation, params.night_min_sza)
    return raw, trusted, obscured_mask(bt, trusted, params)


def read_targets(labels: xr.Dataset, scenes: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Target and weight of the label store scenes `scenes`."""
    chosen = labels.isel(time=scenes)
    probability = chosen["fog_probability"].values if "fog_probability" in chosen else None
    return targets(chosen["label"].values, probability, target)
