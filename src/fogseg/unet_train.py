"""Training, prediction, labels and the saved file of the U-Net."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fogseg.labels import FOG, IGNORE, NO_FOG, OBSCURED
from fogseg.unet import UNet
from fogseg.unet_data import INPUT_CHANNELS, Normalization

log = logging.getLogger(__name__)

EpochCallback = Callable[[int, float, float | None], None]


@dataclasses.dataclass(frozen=True)
class UNetParams:
    """The `unet:` section of the config."""

    model_path: str
    predictions_path: str
    tracking_uri: str
    artifact_dir: str
    experiment: str
    base_channels: int
    depth: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    point_weights: list[float]
    probability_grid: list[float]
    device: str

    @classmethod
    def from_config(cls, unet_cfg: dict) -> UNetParams:
        return cls(**{field.name: unet_cfg[field.name] for field in dataclasses.fields(cls)})


@dataclasses.dataclass(frozen=True)
class UNetModel:
    """A trained U-Net with what is needed to apply it to new scenes."""

    setting: str
    options: dict
    state_dict: dict
    normalization: Normalization
    p_star: float
    input_channels: list[str]
    base_channels: int
    depth: int
    label_attrs: dict
    torch_version: str
    fogseg_version: str


def unet_settings(point_weights: list[float]) -> list[tuple[str, dict]]:
    """The cross-validated settings: soft target, hard target, then the soft target with each point weight."""
    settings = [
        ("1 soft", {"target": "soft", "point_weight": 0.0}),
        ("2 hard", {"target": "hard", "point_weight": 0.0}),
    ]
    for number, weight in enumerate(point_weights, start=3):
        settings.append((f"{number} soft points {weight:g}", {"target": "soft", "point_weight": weight}))
    return settings


def choose_device(name: str) -> torch.device:
    """`auto` is the GPU if there is one, else the CPU with a warning."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("No GPU found; training on the CPU will be slow")
        return torch.device("cpu")
    return torch.device(name)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy over the pixels of weight 1, 0 if there are none."""
    safe_target = torch.where(weight > 0, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def point_bce(
    logits: torch.Tensor, batch: torch.Tensor, iy: torch.Tensor, ix: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """Mean binary cross-entropy at the points, each read from its own pixel of its own scene, 0 if none."""
    if len(truth) == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[batch, iy, ix], truth)


def batch_points(points: pd.DataFrame, scenes: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The points in the scenes of a batch as tensors: position in the batch, row, column and truth."""
    position = pd.Series(np.arange(len(scenes)), index=scenes)
    inside = points["scene"].isin(scenes).to_numpy()
    chosen = points[inside]
    batch = position.loc[chosen["scene"]].to_numpy()
    columns = (batch, chosen["iy"].to_numpy(), chosen["ix"].to_numpy())
    index = [torch.tensor(column, dtype=torch.long, device=device) for column in columns]
    return (*index, torch.tensor(chosen["truth"].to_numpy(), dtype=torch.float32, device=device))


def run_epoch(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    points: pd.DataFrame,
    point_weight: float,
    order: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """One pass over the scenes in `order`; the mean loss of its batches."""
    model.train()
    losses = []
    for start in range(0, len(order), batch_size):
        scenes = order[start : start + batch_size].numpy()
        x, y, w = (torch.as_tensor(array[scenes], device=device) for array in arrays)
        logits = model(x)
        loss = masked_bce(logits, y, w)
        if point_weight > 0:
            loss = loss + point_weight * point_bce(logits, *batch_points(points, scenes, device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def monitor_loss(
    model: UNet, arrays: tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int, device: torch.device
) -> float:
    """The mean masked cross-entropy of the model in eval mode, over batches."""
    model.eval()
    losses = []
    for start in range(0, len(arrays[0]), batch_size):
        x, y, w = (torch.as_tensor(array[start : start + batch_size], device=device) for array in arrays)
        losses.append(masked_bce(model(x), y, w).item())
    return float(np.mean(losses))


def train_model(
    inputs: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    points: pd.DataFrame,
    options: dict,
    params: UNetParams,
    seed: int,
    device: torch.device,
    monitor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    on_epoch: EpochCallback | None = None,
) -> UNet:
    """A U-Net trained on the scenes `inputs` (the `scene` of `points` indexes them), in eval mode."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = UNet(inputs.shape[1], params.base_channels, params.depth).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)
    for epoch in range(1, params.epochs + 1):
        order = torch.randperm(len(inputs), generator=generator)
        train_loss = run_epoch(
            model,
            optimizer,
            (inputs, target, weight),
            points,
            options["point_weight"],
            order,
            params.batch_size,
            device,
        )
        schedule.step()
        held_out_loss = None if monitor is None else monitor_loss(model, monitor, params.batch_size, device)
        if on_epoch is not None:
            on_epoch(epoch, train_loss, held_out_loss)
    return model.eval()


@torch.no_grad()
def predict_probabilities(
    model: UNet, inputs: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Fog probability of each pixel, NaN where the `trusted` channel is 0."""
    model.eval().to(device)
    parts = []
    for start in range(0, len(inputs), batch_size):
        x = torch.as_tensor(inputs[start : start + batch_size], device=device)
        parts.append(torch.sigmoid(model(x)).cpu().numpy())
    probabilities = np.concatenate(parts).astype("float32")
    probabilities[inputs[:, -1] == 0] = np.nan
    return probabilities


def unet_labels(p: np.ndarray, obscured: np.ndarray, p_star: float) -> np.ndarray:
    """Fog from `p_star` up, no fog below, ignored where `p` is NaN and obscured where the rule says so."""
    labels = np.where(p >= p_star, FOG, NO_FOG)
    labels = np.where(np.isnan(p), IGNORE, labels)
    return np.where(obscured, OBSCURED, labels).astype(np.uint8)


def network(model: UNetModel) -> UNet:
    """The U-Net of a saved model, with its weights, in eval mode on the CPU."""
    net = UNet(len(model.input_channels), model.base_channels, model.depth)
    net.load_state_dict(model.state_dict)
    return net.eval()


def save_unet(model: UNetModel, path: Path) -> None:
    """Write the model as plain tensors and values to a `.part` file and move it into place."""
    values = {field.name: getattr(model, field.name) for field in dataclasses.fields(model)}
    values["torch_version"] = str(model.torch_version)
    values["normalization"] = model.normalization.to_dict()
    values["state_dict"] = {name: tensor.cpu() for name, tensor in model.state_dict.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    torch.save(values, part)
    os.replace(part, path)


def load_unet(path: Path) -> UNetModel:
    """The model saved at `path`, which must use the input channels of this code."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `fogseg train` first")
    values = torch.load(path, map_location="cpu", weights_only=True)
    if values["input_channels"] != INPUT_CHANNELS:
        raise ValueError(
            f"The model in {path} takes the input channels {values['input_channels']}, but this code "
            f"builds {INPUT_CHANNELS}; run `fogseg train` again"
        )
    values["normalization"] = Normalization.from_dict(values["normalization"])
    return UNetModel(**values)
