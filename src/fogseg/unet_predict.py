"""Dense U-Net predictions: the fog probability and label of every pixel of a season's scenes.

The store has the layout of the teacher's label store, without the threshold source. `fog_probability` is the
U-Net's probability times 250 as an integer, 255 where the pixel is untrusted or obscured.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fogseg.build import CHUNK_TIME
from fogseg.features import read_static
from fogseg.label_store import (
    TIME_ENCODING,
    check_dataset,
    label_fractions,
    store_attrs,
    write_metadata,
    write_store,
    write_store_atomically,
)
from fogseg.labels import LabelParams
from fogseg.teacher import file_sha256
from fogseg.teacher_labels import PROBABILITY_SCALE, UNDEFINED
from fogseg.unet import UNet
from fogseg.unet_data import model_inputs, read_inputs
from fogseg.unet_train import UNetModel, UNetParams, load_unet, network, predict_probabilities, unet_labels
from fogseg.validation import scenes_between, season_bounds

log = logging.getLogger(__name__)

NIGHT_BATCH = 32


def predict_batch(
    data: xr.Dataset,
    start: int,
    stop: int,
    model: UNetModel,
    net: UNet,
    elevation: np.ndarray,
    params: LabelParams,
    device: torch.device,
    batch_size: int,
) -> xr.Dataset:
    """Predict scenes `start` to `stop` of the dataset, which must hold whole nights."""
    scenes = np.arange(start, stop)
    raw, trusted, obscured = read_inputs(data, scenes, elevation, params)
    inputs = model_inputs(raw, trusted, model.normalization)
    p = predict_probabilities(net, inputs, device, batch_size)

    labels = unet_labels(p, obscured, model.p_star)
    undefined = np.isnan(p) | obscured
    scaled = np.rint(np.nan_to_num(p) * PROBABILITY_SCALE)
    probability = np.where(undefined, UNDEFINED, scaled).astype(np.uint8)

    fractions = np.array([label_fractions(scene) for scene in labels], dtype=np.float32)
    return xr.Dataset(
        {
            "label": (
                ("time", "y", "x"),
                labels,
                {"long_name": "fog label: 0 no fog, 1 fog, 2 obscured, 255 ignore"},
            ),
            "fog_probability": (
                ("time", "y", "x"),
                probability,
                {"long_name": f"U-Net probability of fog times {PROBABILITY_SCALE}, 255 undefined"},
            ),
            "fog_fraction": (("time",), fractions[:, 0], {"long_name": "fog share of the decided pixels"}),
            "ignore_fraction": (("time",), fractions[:, 1], {"long_name": "ignored share of all pixels"}),
            "obscured_fraction": (("time",), fractions[:, 2], {"long_name": "obscured share of all pixels"}),
        },
        coords={"time": data["time"].values[start:stop], "y": data["y"].values, "x": data["x"].values},
    )


def check_model_matches(model: UNetModel, data: xr.Dataset, model_path: Path) -> None:
    """The model must have been trained on this build of the dataset."""
    trained_on = model.label_attrs.get("source_created", "unknown")
    created = data.attrs.get("created", "unknown")
    if trained_on == created == "unknown":
        log.warning("Neither the model nor the dataset records a creation time; the match is not checked")
    if trained_on != created:
        raise ValueError(
            f"The model {model_path} was trained on the dataset created {trained_on}, but the dataset was "
            f"created {created}; run `fogseg train` again"
        )


def predict_dataset(
    dataset_path: Path, out_path: Path, cfg: dict, season: str, overwrite: bool, device: torch.device
) -> int:
    """Predict the scenes of `season` into the Zarr store `out_path`; returns the number of scenes."""
    unet_params = UNetParams.from_config(cfg["unet"])
    model_path = Path(unet_params.model_path)
    model = load_unet(model_path)
    params = LabelParams.from_config(model.label_attrs)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    net = network(model).to(device)

    def write(tmp_path: Path) -> int:
        with ExitStack() as stack:
            full = stack.enter_context(xr.open_zarr(dataset_path, consolidated=False))
            check_dataset(full, dataset_path)
            check_model_matches(model, full, model_path)
            start, end = season_bounds(cfg, season)
            data = full.isel(time=scenes_between(full["time"].values, start, end, season))
            elevation = read_static(full, Path(cfg["dem"]["out_path"]), ["terrain"]).elevation_mean

            def make_batch(first: int, last: int) -> xr.Dataset:
                return predict_batch(
                    data, first, last, model, net, elevation, params, device, unet_params.batch_size
                )

            shape = (CHUNK_TIME, data.sizes["y"], data.sizes["x"])
            encoding = {
                "label": {"chunks": shape},
                "fog_probability": {"chunks": shape},
                "fog_fraction": {"chunks": (CHUNK_TIME,)},
                "ignore_fraction": {"chunks": (CHUNK_TIME,)},
                "obscured_fraction": {"chunks": (CHUNK_TIME,)},
                "time": TIME_ENCODING,
            }
            write_store(data, tmp_path, make_batch, encoding, NIGHT_BATCH)
            attrs = {
                **store_attrs(full, dataset_path, "unet", params, None),
                "season": season,
                "unet_setting": model.setting,
                "unet_p_star": model.p_star,
                "unet_model_path": str(model_path),
                "unet_model_sha256": file_sha256(model_path),
                "torch_version": model.torch_version,
            }
            write_metadata(tmp_path, attrs)
            return data.sizes["time"]

    return write_store_atomically(out_path, overwrite, write)
