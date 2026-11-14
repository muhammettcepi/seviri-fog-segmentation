"""Figures and the band check of docs/notes/unet.md, from the outputs of `fogseg train`, `predict` and `evaluate`.

    python docs/notes/figures/unet_figures.py --config configs/default.yaml

It reads the held-out cases, so run it only after `fogseg evaluate`; nothing it prints is used to choose anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mlflow.tracking import MlflowClient

from fogseg.config import load_config
from fogseg.experiment import experiment_cases, held_out_parts
from fogseg.inspect import draw_labels_panel, night_microphysics_rgb, scene_channels
from fogseg.label_store import label_store_path
from fogseg.labels import FOG
from fogseg.splits import held_out_part
from fogseg.validation import TRUTH_YES, night_of

mpl.use("Agg")

FIGURES = Path("docs/notes/figures")
SEASON = "2023-24"
EXAMPLE_SCENES = 2
UNSEEN_NIGHT_PARTS = ("nights", "both")
UNDEFINED = 255
PROBABILITY_SCALE = 250


def winner_fold_runs(client: MlflowClient, record: dict) -> list:
    """The CV runs of the winning setting, nested under the run of the `fogseg train` that saved the model."""
    parent = client.get_run(record["parent_run_id"])
    children = client.search_runs(
        [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )
    folds = [run for run in children if run.info.run_name.startswith(f"{record['winner']} fold")]
    return sorted(folds, key=lambda run: run.info.run_name)


def learning_curves(cfg: dict, record: dict) -> Path:
    client = MlflowClient(cfg["unet"]["tracking_uri"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in winner_fold_runs(client, record):
        for metric, style in (("train_loss", "-"), ("test_loss", "--")):
            history = client.get_metric_history(run.info.run_id, metric)
            steps = [point.step for point in history]
            values = [point.value for point in history]
            ax.plot(steps, values, style, label=f"{run.info.run_name} {metric}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Setting {record['winner']}: training and test-fold loss per CV fold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = FIGURES / "unet_learning_curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def at_cases(store: xr.Dataset, name: str, cases: pd.DataFrame) -> np.ndarray:
    scene = pd.Index(store["time"].values).get_indexer(pd.to_datetime(cases["scene_time"]))
    return store[name].values[scene, cases["iy"].to_numpy(), cases["ix"].to_numpy()]


def band_check(stores: dict[str, xr.Dataset], held_out: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Full-count POD and FAR of the U-Net, the teacher store, and the teacher decided everywhere at its p*."""
    teacher = stores["teacher"]
    probability = at_cases(teacher, "fog_probability", held_out)
    p_star = float(teacher.attrs["teacher_p_star"])
    says_fog = {
        "U-Net": at_cases(stores["U-Net"], "label", held_out) == FOG,
        "teacher store": at_cases(teacher, "label", held_out) == FOG,
        "teacher without its band": (probability != UNDEFINED) & (probability / PROBABILITY_SCALE >= p_star),
    }
    fog = held_out["truth_fls"].to_numpy() == TRUTH_YES
    rows = []
    for part, mask in held_out_parts(held_out, n_folds).items():
        in_part = mask.to_numpy()
        for source, fog_label in says_fog.items():
            hits = (fog_label & fog & in_part).sum()
            misses = (~fog_label & fog & in_part).sum()
            false_alarms = (fog_label & ~fog & in_part).sum()
            pod, far = hits / (hits + misses), false_alarms / (hits + false_alarms)
            rows.append({"part": part, "source": source, "pod": round(pod, 3), "far": round(far, 3)})
    return pd.DataFrame(rows)


def example_scenes(held_out: pd.DataFrame, n_folds: int) -> list[pd.Timestamp]:
    """The scene with the most station fog reports in each of the unseen nights that have the most."""
    unseen_night = held_out_part(held_out, n_folds).isin(UNSEEN_NIGHT_PARTS)
    reports = held_out[unseen_night & (held_out["truth_fls"] == TRUTH_YES)]
    per_scene = reports.groupby("scene_time").size().sort_values(ascending=False).reset_index()
    per_scene["night"] = night_of(per_scene["scene_time"].to_numpy())
    busiest = per_scene.drop_duplicates("night").head(EXAMPLE_SCENES)["scene_time"]
    return [pd.Timestamp(time) for time in busiest]


def draw_cases(ax, cases: pd.DataFrame) -> None:
    fog = cases["truth_fls"] == TRUTH_YES
    ax.scatter(cases.loc[fog, "ix"], cases.loc[fog, "iy"], s=14, facecolors="none", edgecolors="red")
    ax.scatter(cases.loc[~fog, "ix"], cases.loc[~fog, "iy"], s=3, c="black")


def example_map(data: xr.Dataset, stores: dict, cases: pd.DataFrame, time: pd.Timestamp) -> Path:
    scene = int(pd.Index(data["time"].values).get_loc(time))
    rgb = night_microphysics_rgb(*scene_channels(data, scene))
    at_scene = cases[pd.to_datetime(cases["scene_time"]) == time]
    fig, axes = plt.subplots(1, len(stores), figsize=(6 * len(stores), 4))
    for ax, (name, store) in zip(axes, stores.items(), strict=True):
        draw_labels_panel(ax, rgb, store["label"].sel(time=time).values)
        draw_cases(ax, at_scene)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(f"{time:%Y-%m-%d %H:%M} UTC, an unseen night; red rings: station fog, black dots: no fog")
    fig.tight_layout()
    path = FIGURES / f"unet_example_{time:%Y%m%dT%H%M}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)
    n_folds = cfg["stations"]["n_folds"]
    labels_dir = Path(cfg["labels"]["output_dir"])
    record = json.loads((Path(cfg["stations"]["data_dir"]) / "unet_config.json").read_text())
    stores = {
        "rule": xr.open_zarr(label_store_path(labels_dir, "fixed"), consolidated=False),
        "teacher": xr.open_zarr(label_store_path(labels_dir, "teacher"), consolidated=False),
        "U-Net": xr.open_zarr(cfg["unet"]["predictions_path"], consolidated=False),
    }
    _, held_out = experiment_cases(cfg, SEASON)

    print(learning_curves(cfg, record))
    print(band_check(stores, held_out, n_folds).to_string(index=False))
    data = xr.open_zarr(cfg["data"]["processed_path"], consolidated=False)
    for time in example_scenes(held_out, n_folds):
        print(example_map(data, stores, held_out, time))


if __name__ == "__main__":
    main()
