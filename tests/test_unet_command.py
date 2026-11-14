import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
import zarr
from test_teacher import teacher_config

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from fogseg.cli import cmd_train
from fogseg.label_store import label_dataset, label_store_path
from fogseg.labels import LabelParams
from fogseg.unet_train import load_unet
from fogseg.validation import night_of


def teacher_store(cfg: dict) -> Path:
    """A teacher-style store: the rule's labels of the tiny dataset with a probability of 200 or 25."""
    dataset, labels_dir = Path(cfg["data"]["processed_path"]), Path(cfg["labels"]["output_dir"])
    rule = label_store_path(labels_dir, "fixed")
    label_dataset(dataset, rule, "fixed", LabelParams.from_config(cfg["labels"]))
    with xr.open_zarr(rule, consolidated=False) as store:
        label = store["label"].load()
        attrs = {**store.attrs, "mode": "teacher", "teacher_setting": "3 logistic", "teacher_p_star": 0.4}
    probability = xr.where(label == 1, 200, xr.where(label == 0, 25, 255)).astype("uint8")
    path = label_store_path(labels_dir, "teacher")
    xr.Dataset({"label": label, "fog_probability": probability}, attrs=attrs).to_zarr(
        path, zarr_format=3, consolidated=False
    )
    return path


def unet_config(tmp_path: Path, mlflow_uri: str) -> dict:
    """`teacher_config` with a teacher-style label store and a tiny `unet:` section on the CPU."""
    cfg = teacher_config(tmp_path)
    cfg["labels"] |= {"mode": "teacher", "output_dir": str(tmp_path / "labels")}
    cfg["stations"] |= {"matched_far": 0.9}
    cfg["unet"] = {
        "model_path": str(tmp_path / "processed" / "unet.pt"),
        "predictions_path": str(tmp_path / "processed" / "predictions_unet.zarr"),
        "tracking_uri": mlflow_uri,
        "artifact_dir": str(tmp_path / "artifacts"),
        "experiment": "test",
        "base_channels": 4,
        "depth": 2,
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "point_weights": [0.25, 1.0],
        "probability_grid": [0.05, 0.95, 0.05],
        "device": "cpu",
    }
    teacher_store(cfg)
    return cfg


def train_args(cfg: dict, season: str | None = None) -> argparse.Namespace:
    config_path = Path(cfg["stations"]["data_dir"]).parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, season=season)


def test_train_command_writes_model_tables_and_runs(tmp_path, mlflow_uri):
    cfg = unet_config(tmp_path, mlflow_uri)
    cmd_train(train_args(cfg))
    stations = tmp_path / "stations"
    table = pd.read_csv(stations / "unet_cv.csv")
    assert table["variant"].nunique() == 4
    oof = pd.read_csv(stations / "unet_oof.csv")
    assert oof.columns.tolist() == [
        "case",
        "scene_time",
        "station",
        "night_fold",
        "station_fold",
        "truth_fls",
        "p",
    ]
    record = json.loads((stations / "unet_config.json").read_text(encoding="utf-8"))
    assert record["unet"] == cfg["unet"]
    assert record["labels_mode"] == "teacher"
    assert record["winner"] in set(table["variant"])
    assert set(record["inputs"]) == {"cases_sha256", "splits_sha256", "labels_created"}

    model = load_unet(Path(cfg["unet"]["model_path"]))
    assert model.setting == record["winner"]
    assert model.p_star == record["p_star"]
    assert model.label_attrs["teacher_setting"] == "3 logistic"

    client = mlflow.MlflowClient(mlflow_uri)
    runs = client.search_runs([client.get_experiment_by_name("test").experiment_id])
    nested = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == record["parent_run_id"]]
    assert len(runs) == len(nested) + 1
    names = [run.data.tags["mlflow.runName"] for run in nested]
    assert sum(name.endswith(("fold 1", "fold 2")) for name in names) == 4 * 2
    assert names.count("final") == 1
    assert [run.info.run_id for run in nested if run.data.tags["mlflow.runName"] == "final"] == [
        record["final_run_id"]
    ]
    artifacts = {item.path for item in client.list_artifacts(record["parent_run_id"])}
    assert artifacts == {"unet_cv.csv", "unet_oof.csv", "unet_config.json"}


@pytest.mark.parametrize("name", ["unet_held_out.csv", "unet_held_out.json"])
def test_train_refuses_after_the_held_out_look(tmp_path, mlflow_uri, name):
    cfg = unet_config(tmp_path, mlflow_uri)
    (tmp_path / "stations" / name).write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match=name):
        cmd_train(train_args(cfg))
    assert not (tmp_path / "stations" / "unet_cv.csv").exists()


def test_train_refuses_the_test_season(tmp_path, mlflow_uri):
    with pytest.raises(PermissionError):
        cmd_train(train_args(unet_config(tmp_path, mlflow_uri), season="2024-25"))


def test_train_refuses_a_label_store_of_another_dataset(tmp_path, mlflow_uri):
    cfg = unet_config(tmp_path, mlflow_uri)
    store = label_store_path(Path(cfg["labels"]["output_dir"]), "teacher")
    zarr.open_group(store, mode="r+").attrs.update({"source_created": "another build"})
    with pytest.raises(ValueError, match="fogseg label"):
        cmd_train(train_args(cfg))
    assert not Path(cfg["unet"]["model_path"]).exists()


def test_train_refuses_soft_settings_without_fog_probability(tmp_path, mlflow_uri):
    cfg = unet_config(tmp_path, mlflow_uri)
    store = label_store_path(Path(cfg["labels"]["output_dir"]), "teacher")
    del zarr.open_group(store, mode="r+")["fog_probability"]
    with pytest.raises(ValueError, match="fog_probability"):
        cmd_train(train_args(cfg))


def test_train_fits_only_scenes_of_the_selection_night_folds(tmp_path, mlflow_uri, monkeypatch):
    cfg = unet_config(tmp_path, mlflow_uri)
    seen = {}

    def capture(fit, *_):
        seen["fit"] = fit
        raise KeyboardInterrupt

    monkeypatch.setattr("fogseg.unet_cv.cross_validate_unet", capture)
    with pytest.raises(KeyboardInterrupt):
        cmd_train(train_args(cfg))
    fit = seen["fit"]
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        everything = data["time"].values
    held_out_night = night_of(everything) == np.datetime64("2023-11-22")
    assert held_out_night.any()
    assert not np.isin(fit.times, everything[held_out_night]).any()
    assert set(fit.night_folds) == {1, 2}
    assert len(fit.times) == (~held_out_night).sum()
    assert fit.usable.shape == fit.targets["soft"][0].shape
    assert fit.usable.dtype == bool
    assert fit.usable[fit.points["scene"], fit.points["iy"], fit.points["ix"]].all()
