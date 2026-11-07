import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
import zarr
from test_teacher import teacher_config

torch = pytest.importorskip("torch")

from fogseg.build import variable_length_strings
from fogseg.cli import cmd_predict
from fogseg.features import read_static
from fogseg.label_store import store_attrs
from fogseg.labels import FOG, IGNORE, OBSCURED, LabelParams
from fogseg.teacher import file_sha256
from fogseg.unet import UNet
from fogseg.unet_data import INPUT_CHANNELS, Normalization, model_inputs, read_inputs
from fogseg.unet_predict import predict_dataset
from fogseg.unet_train import UNetModel, load_unet, network, predict_probabilities, save_unet, unet_labels
from fogseg.validation import scenes_between, season_bounds

CPU = torch.device("cpu")
BASE_CHANNELS, DEPTH = 4, 2


def prepare_dataset(path: Path) -> None:
    """Move the last night to the test season; add a NaN pixel, a twilight pixel and a cold pixel."""
    with xr.open_zarr(path, consolidated=False) as store:
        data = store.load()
    channel = list(data["channel"].values).index("IR_108")
    data["bt"][0, channel, 0, 0] = np.nan
    data["sza"][1, 1, 1] = 90.0
    data["bt"][2, channel, 3, 3] = 250.0
    times = data["time"].values.copy()
    times[-2:] += np.timedelta64(366, "D")
    variable_length_strings(data.assign_coords(time=times)).to_zarr(
        path, mode="w", zarr_format=3, consolidated=False
    )


def save_model(cfg: dict, p_star: float, source_created: str = "unknown") -> None:
    torch.manual_seed(0)
    label_attrs = {**dataclasses.asdict(LabelParams.from_config(cfg["labels"])), "mode": "teacher"}
    model = UNetModel(
        setting="3 logistic",
        options={"target": "soft", "point_weight": 0.0},
        state_dict=UNet(len(INPUT_CHANNELS), BASE_CHANNELS, DEPTH).state_dict(),
        normalization=Normalization(np.full(13, 270.0), np.full(13, 10.0)),
        p_star=p_star,
        input_channels=INPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        depth=DEPTH,
        label_attrs={**label_attrs, "source_created": source_created},
        torch_version=str(torch.__version__),
        fogseg_version="test",
    )
    save_unet(model, Path(cfg["unet"]["model_path"]))


def predict_config(tmp_path: Path, p_star: float = 0.5, source_created: str = "unknown") -> dict:
    """A tiny season and a model with fresh weights; the third night is in the test season."""
    cfg = teacher_config(tmp_path)
    dataset_path = Path(cfg["data"]["processed_path"])
    prepare_dataset(dataset_path)
    cfg["unet"] = {
        "model_path": str(tmp_path / "processed" / "unet.pt"),
        "predictions_path": str(tmp_path / "processed" / "predictions_unet.zarr"),
        "tracking_uri": "unused",
        "artifact_dir": str(tmp_path / "artifacts"),
        "experiment": "test",
        "base_channels": BASE_CHANNELS,
        "depth": DEPTH,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "point_weights": [0.25, 1.0],
        "probability_grid": [0.05, 0.95, 0.05],
        "device": "cpu",
    }
    save_model(cfg, p_star, source_created)
    return cfg


def predict_args(
    cfg: dict, season: str | None = None, final: bool = False, overwrite: bool = False
) -> argparse.Namespace:
    config_path = Path(cfg["stations"]["data_dir"]).parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, season=season, final=final, overwrite=overwrite)


def run_predict(cfg: dict, season: str = "2023-24", overwrite: bool = False) -> xr.Dataset:
    out_path = Path(cfg["unet"]["predictions_path"])
    predict_dataset(Path(cfg["data"]["processed_path"]), out_path, cfg, season, overwrite, CPU)
    return xr.open_zarr(out_path, consolidated=False)


def season_inputs(cfg: dict, p_star: float):
    """The probabilities, obscured mask and labels of the season computed without the store."""
    model_path = Path(cfg["unet"]["model_path"])
    model = load_unet(model_path)
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        start, end = season_bounds(cfg, "2023-24")
        scenes = scenes_between(data["time"].values, start, end, "2023-24")
        elevation = read_static(data, Path(cfg["dem"]["out_path"]), ["terrain"]).elevation_mean
        raw, trusted, obscured = read_inputs(
            data, scenes, elevation, LabelParams.from_config(model.label_attrs)
        )
    inputs = model_inputs(raw, trusted, model.normalization)
    p = predict_probabilities(network(model), inputs, CPU, 2)
    return p, obscured, unet_labels(p, obscured, p_star)


def test_prediction_store_layout_and_attributes(tmp_path):
    cfg = predict_config(tmp_path)
    with run_predict(cfg) as store:
        assert store["label"].dims == ("time", "y", "x")
        assert store["label"].shape == (4, 8, 10)
        assert store["label"].dtype == store["fog_probability"].dtype == np.uint8
        assert set(store.data_vars) == {
            "label",
            "fog_probability",
            "fog_fraction",
            "ignore_fraction",
            "obscured_fraction",
        }
        assert store["fog_fraction"].dtype == np.float32
        attrs = store.attrs
    model_path = Path(cfg["unet"]["model_path"])
    assert attrs["mode"] == "unet" and attrs["season"] == "2023-24"
    assert attrs["unet_setting"] == "3 logistic" and attrs["unet_p_star"] == 0.5
    assert attrs["unet_model_path"] == str(model_path)
    assert attrs["unet_model_sha256"] == file_sha256(model_path)
    assert attrs["torch_version"] == str(torch.__version__)
    assert attrs["source_dataset"] == cfg["data"]["processed_path"]
    assert {"fogseg_version", "source_created", "created", "btd_fixed"} <= set(attrs)


def test_labels_follow_the_probabilities_of_the_store(tmp_path):
    cfg = predict_config(tmp_path)
    p, obscured, _ = season_inputs(cfg, 0.5)
    middle = np.rint(np.nanmedian(p[~obscured]) * 250) + 0.5
    save_model(cfg, float(middle / 250))
    p, obscured, expected = season_inputs(cfg, middle / 250)
    with run_predict(cfg, overwrite=True) as store:
        label, probability = store["label"].values, store["fog_probability"].values

    assert np.array_equal(label, expected)
    defined = probability != 255
    assert np.array_equal(defined, ~np.isnan(p) & ~obscured)
    assert np.array_equal(probability[defined], np.rint(p[defined] * 250))
    assert np.array_equal(label[defined], unet_labels(probability[defined] / 250, False, middle / 250))
    assert {FOG, 0} <= set(label[defined].tolist())
    untrusted = np.isnan(p) & ~obscured
    assert obscured.any() and untrusted.any() and np.isnan(p[0, 0, 0])
    assert (label[obscured] == OBSCURED).all() and (probability[obscured] == 255).all()
    assert (label[untrusted] == IGNORE).all() and (probability[untrusted] == 255).all()
    assert label[1, 1, 1] == IGNORE and label[0, 0, 0] == IGNORE


def test_an_obscured_pixel_stays_obscured_above_the_threshold(tmp_path):
    cfg = predict_config(tmp_path)
    p, obscured, _ = season_inputs(cfg, 0.5)
    assert obscured.any()
    save_model(cfg, float(p[obscured].min()))
    with run_predict(cfg, overwrite=True) as store:
        label, probability = store["label"].values, store["fog_probability"].values
    assert (label[obscured] == OBSCURED).all() and (probability[obscured] == 255).all()
    assert (label[~obscured & ~np.isnan(p)] == FOG).any()


def test_only_the_season_is_written(tmp_path):
    cfg = predict_config(tmp_path)
    with run_predict(cfg) as store:
        times = store["time"].values
    start, end = season_bounds(cfg, "2023-24")
    assert len(times) == 4 and ((times >= start) & (times < end)).all()


def test_the_test_season_needs_final(tmp_path):
    cfg = predict_config(tmp_path)
    with pytest.raises(PermissionError):
        cmd_predict(predict_args(cfg, season="2024-25"))
    assert not Path(cfg["unet"]["predictions_path"]).exists()
    cmd_predict(predict_args(cfg, season="2024-25", final=True))
    with xr.open_zarr(cfg["unet"]["predictions_path"], consolidated=False) as store:
        assert store.sizes["time"] == 2 and store.attrs["season"] == "2024-25"


def test_existing_store_needs_overwrite(tmp_path):
    cfg = predict_config(tmp_path)
    cmd_predict(predict_args(cfg))
    with pytest.raises(FileExistsError, match="overwrite"):
        cmd_predict(predict_args(cfg))
    cmd_predict(predict_args(cfg, overwrite=True))
    assert zarr.open_group(cfg["unet"]["predictions_path"], mode="r").attrs["mode"] == "unet"


def test_a_model_trained_on_another_dataset_is_refused(tmp_path):
    cfg = predict_config(tmp_path, source_created="2020-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="fogseg train"):
        run_predict(cfg)
    assert not Path(cfg["unet"]["predictions_path"]).exists()


def test_command_logs_the_scene_count_and_the_mean_fog_fraction(tmp_path, caplog):
    cfg = predict_config(tmp_path)
    with caplog.at_level("INFO", logger="fogseg"):
        cmd_predict(predict_args(cfg))
    assert "Predicted 4 scenes of 2023-24" in caplog.text
    assert cfg["unet"]["predictions_path"] in caplog.text


def test_store_attrs_layout_is_shared_with_the_label_stores(tmp_path):
    cfg = predict_config(tmp_path)
    with (
        run_predict(cfg) as store,
        xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data,
    ):
        rule = store_attrs(
            data, Path(cfg["data"]["processed_path"]), "fixed", LabelParams.from_config(cfg["labels"]), None
        )
        assert set(rule) <= set(store.attrs)
