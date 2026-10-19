"""Write the fog label stores: one Zarr store per labelling mode, aligned with the dataset.

Each store holds, per scene, the label image (0 no fog, 1 fog, 255 ignore), the BTD threshold used
and where it came from, and the share of fog and ignored pixels. The thresholds and the version
that produced it are stored as attributes, so a label store explains itself.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import fogseg
from fogseg.build import CHUNK_TIME
from fogseg.dynamic_threshold import DynamicParams, ThresholdChoice, dynamic_threshold
from fogseg.labels import FOG, IGNORE, NO_FOG, LabelParams, histogram_pixels, label_scene

MODES = ("fixed", "dynamic")
LABEL_CHANNELS = ["IR_039", "IR_108", "IR_120"]
# Every batch must be written with the same string width, or appending would truncate a longer
# name; "fallback" is the longest threshold source.
SOURCE_DTYPE = "U8"
# Same fixed time units as the dataset, so appended batches are stored exactly like the first one.
TIME_ENCODING = {"units": "minutes since 2000-01-01", "dtype": "int64"}


def label_store_path(output_dir: Path, mode: str) -> Path:
    """Where the label store of a mode lives, e.g. `labels_fixed.zarr`."""
    return output_dir / f"labels_{mode}.zarr"


def check_mode(mode: str, dynamic: DynamicParams | None) -> None:
    """The mode must be known, and the dynamic mode needs its settings."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "dynamic" and dynamic is None:
        raise ValueError("mode 'dynamic' needs dynamic parameters (labels.dynamic in the config)")


def check_dataset(data: xr.Dataset, dataset_path: Path) -> None:
    """The dataset must have `bt` and `sza`, the three channels the labels need, and some scenes."""
    for name in ["bt", "sza"]:
        if name not in data:
            raise ValueError(f"{dataset_path} has no variable '{name}'")

    present = [str(channel) for channel in data["channel"].values]
    missing = [channel for channel in LABEL_CHANNELS if channel not in present]
    if missing:
        raise ValueError(f"{dataset_path} has no channel {missing}; it has {present}")

    if data.sizes["time"] == 0:
        raise ValueError(f"{dataset_path} contains no scenes")


def choose_threshold(
    ir039: np.ndarray,
    ir108: np.ndarray,
    ir120: np.ndarray,
    sza: np.ndarray,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams | None,
) -> ThresholdChoice:
    """BTD threshold of one scene: `btd_fixed` in fixed mode, the clear-ground peak plus an offset in dynamic mode."""
    if mode == "fixed":
        return ThresholdChoice(params.btd_fixed, "fixed")
    btd = histogram_pixels(ir039, ir108, ir120, sza, params)
    return dynamic_threshold(btd, params.btd_fixed, dynamic)


def label_fractions(labels: np.ndarray) -> tuple[float, float]:
    """Fog share of the decided pixels (NaN if there are none) and ignored share of all pixels."""
    n_fog = int(np.count_nonzero(labels == FOG))
    n_no_fog = int(np.count_nonzero(labels == NO_FOG))
    n_ignore = int(np.count_nonzero(labels == IGNORE))

    n_decided = n_fog + n_no_fog
    fog_fraction = n_fog / n_decided if n_decided > 0 else np.nan
    ignore_fraction = n_ignore / labels.size
    return fog_fraction, ignore_fraction


def label_batch(
    data: xr.Dataset,
    start: int,
    stop: int,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams | None,
) -> xr.Dataset:
    """Label scenes `start` to `stop` of the dataset; the result has the variables of the label store."""
    scenes = data.isel(time=slice(start, stop))
    bt = scenes["bt"].sel(channel=LABEL_CHANNELS).values
    sza = scenes["sza"].values

    n_scenes = bt.shape[0]
    labels = np.empty((n_scenes, bt.shape[2], bt.shape[3]), dtype=np.uint8)
    thresholds = np.empty(n_scenes, dtype=np.float32)
    sources = np.empty(n_scenes, dtype=SOURCE_DTYPE)
    fog_fractions = np.empty(n_scenes, dtype=np.float32)
    ignore_fractions = np.empty(n_scenes, dtype=np.float32)

    for i in range(n_scenes):
        ir039, ir108, ir120 = bt[i]
        choice = choose_threshold(ir039, ir108, ir120, sza[i], mode, params, dynamic)
        labels[i] = label_scene(ir039, ir108, ir120, sza[i], choice.value, params)
        thresholds[i] = choice.value
        sources[i] = choice.source
        fog_fractions[i], ignore_fractions[i] = label_fractions(labels[i])

    return xr.Dataset(
        {
            "label": (("time", "y", "x"), labels, {"long_name": "fog label: 0 no fog, 1 fog, 255 ignore"}),
            "btd_threshold": (("time",), thresholds, {"units": "K", "long_name": "BTD threshold used"}),
            "threshold_source": (("time",), sources, {"long_name": "fixed, dynamic or fallback"}),
            "fog_fraction": (("time",), fog_fractions, {"long_name": "fog share of the decided pixels"}),
            "ignore_fraction": (("time",), ignore_fractions, {"long_name": "ignored share of all pixels"}),
        },
        coords={"time": scenes["time"].values, "y": scenes["y"].values, "x": scenes["x"].values},
    )


def write_batches(
    data: xr.Dataset,
    store: Path,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams | None,
    batch: int,
) -> None:
    """Label all scenes and write them to `store`, `batch` scenes at a time."""
    n_scenes = data.sizes["time"]
    encoding = {
        "label": {"chunks": (CHUNK_TIME, data.sizes["y"], data.sizes["x"])},
        "btd_threshold": {"chunks": (CHUNK_TIME,)},
        "threshold_source": {"chunks": (CHUNK_TIME,)},
        "fog_fraction": {"chunks": (CHUNK_TIME,)},
        "ignore_fraction": {"chunks": (CHUNK_TIME,)},
        "time": TIME_ENCODING,
    }

    for start in range(0, n_scenes, batch):
        stop = min(start + batch, n_scenes)
        labelled = label_batch(data, start, stop, mode, params, dynamic)

        is_first_batch = start == 0
        if is_first_batch:
            labelled.to_zarr(store, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
        else:
            # x and y are already in the store; appending them again would overwrite them
            labelled = labelled.drop_vars(["x", "y"])
            labelled.to_zarr(store, append_dim="time", consolidated=False)


def store_attrs(
    data: xr.Dataset,
    dataset_path: Path,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams | None,
) -> dict:
    """Everything needed to reproduce the labels: the mode, all thresholds, and the source dataset."""
    attrs = {"mode": mode}
    attrs.update(dataclasses.asdict(params))
    if mode == "dynamic":
        for name, value in dataclasses.asdict(dynamic).items():
            attrs[f"dynamic_{name}"] = list(value) if isinstance(value, tuple) else value

    attrs["fogseg_version"] = fogseg.__version__
    attrs["source_dataset"] = str(dataset_path)
    attrs["source_created"] = data.attrs.get("created", "unknown")
    attrs["created"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return attrs


def write_metadata(store: Path, attrs: dict) -> None:
    """Store-level attributes, written once at the end: appending batches would overwrite them."""
    zarr.open_group(store, mode="r+").attrs.update(attrs)


def label_dataset(
    dataset_path: Path,
    out_path: Path,
    mode: str,
    params: LabelParams,
    dynamic: DynamicParams | None = None,
    overwrite: bool = False,
    batch: int = 32,
) -> int:
    """Label every scene of the dataset into the Zarr store `out_path`. Returns the number of scenes.

    Written like the dataset itself: under a temporary name, renamed at the end, and the temporary
    store is deleted if anything fails. An existing store stays untouched until the new one is complete.
    """
    check_mode(mode, dynamic)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; use overwrite=True (--overwrite) to replace it")

    with xr.open_zarr(dataset_path, consolidated=False) as data:
        check_dataset(data, dataset_path)

        tmp_path = out_path.with_name(out_path.name + ".part")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_batches(data, tmp_path, mode, params, dynamic, batch)
            write_metadata(tmp_path, store_attrs(data, dataset_path, mode, params, dynamic))
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        n_scenes = data.sizes["time"]

    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    return n_scenes


def summarize_store(store: Path) -> tuple[dict[str, int], float]:
    """Scenes per threshold source, and the mean fog fraction over the scenes that have one."""
    with xr.open_zarr(store, consolidated=False) as labels:
        sources = [str(source) for source in labels["threshold_source"].values]
        fog_fractions = labels["fog_fraction"].values

    source_counts = {source: sources.count(source) for source in sorted(set(sources))}
    has_fraction = fog_fractions[np.isfinite(fog_fractions)]
    mean_fog_fraction = float(has_fraction.mean()) if has_fraction.size else float("nan")
    return source_counts, mean_fog_fraction
