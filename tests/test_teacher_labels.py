from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import xarray as xr
import yaml
from helpers import COLD_PIXEL, GRID_LAT0, GRID_LON0, TEST_PARAMS
from sklearn.isotonic import IsotonicRegression
from test_experiment import experiment_config, tiny_season
from test_features import era5_store, teacher_dataset

from fogseg.cli import cmd_label
from fogseg.features import case_features, night_features, read_static
from fogseg.label_store import night_runs
from fogseg.labels import OBSCURED
from fogseg.teacher import (
    MODEL_SETTINGS,
    TeacherModel,
    feature_names,
    load_teacher,
    make_model,
    probability_of_cases,
    save_teacher,
    teacher_labels,
)
from fogseg.teacher_labels import label_teacher_dataset
from fogseg.validation import season_bounds

N_NIGHTS = 3
LENGTH_KM = 25.0
FEEDER_PIXEL = (4, 4)


def write_toy_dataset(tmp_path: Path, nan_pixel: tuple[int, int] | None = None) -> xr.Dataset:
    """Three nights of two scenes with eight channels and a lon / lat grid, written to `season_8ch.zarr`."""
    toy_dir = tmp_path / "toy"
    toy_dir.mkdir()
    data = teacher_dataset(tiny_season(toy_dir, n_nights=N_NIGHTS)[0])
    data["bt"].loc[{"channel": "IR_108", "y": float(COLD_PIXEL[0]), "x": float(COLD_PIXEL[1])}] = 230.0
    if nan_pixel is not None:
        data["bt"].loc[{"channel": "IR_108", "y": float(nan_pixel[0]), "x": float(nan_pixel[1])}] = np.nan
    path = tmp_path / "season_8ch.zarr"
    data.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return xr.open_zarr(path, consolidated=False)


def night_spans(data: xr.Dataset) -> list[np.ndarray]:
    return [np.arange(start, stop) for start, stop in night_runs(data["time"].values)]


def fitted_toy_teacher(
    data: xr.Dataset, groups: list[str], era5: xr.Dataset | None, correction_km: float | None
) -> TeacherModel:
    """A logistic teacher that finds the fog patch in the corner of every scene."""
    static = read_static(data, Path("no_dem.zarr"), groups)
    names = feature_names(groups)
    in_patch = np.zeros((data.sizes["y"], data.sizes["x"]), dtype=bool)
    in_patch[:5, :5] = True
    rows, is_fog = [], []
    for scenes in night_spans(data):
        night = night_features(data, scenes, static, era5, TEST_PARAMS, groups)
        usable = night.trusted & night.has_era5[:, None, None]
        rows.append(np.moveaxis(night.stack, 1, -1)[usable])
        is_fog.append(np.broadcast_to(in_patch, usable.shape)[usable])
    features, is_fog = pd.DataFrame(np.concatenate(rows), columns=names), np.concatenate(is_fog)
    options = MODEL_SETTINGS[1][1]
    model = make_model(options, 0).fit(features, is_fog)
    raw = model.predict_proba(features)[:, 1]
    return TeacherModel(
        setting="toy",
        options=options,
        model=model,
        calibrator=IsotonicRegression(out_of_bounds="clip").fit(raw, is_fog),
        feature_names=names,
        feature_groups=groups,
        p_star=0.5,
        ignore_halfwidth=0.1,
        label_params=dataclasses.asdict(TEST_PARAMS),
        correction_km=correction_km,
    )


def feeder_table(data: xr.Dataset) -> pd.DataFrame:
    """One missed fog case in every scene at `FEEDER_PIXEL`: its residual of 1 moves the pixels around it."""
    scenes = data["time"].values
    return pd.DataFrame(
        {
            "scene_time": scenes,
            "station": "F",
            "lon": GRID_LON0 + 0.1 * FEEDER_PIXEL[1],
            "lat": GRID_LAT0 - 0.1 * FEEDER_PIXEL[0],
            "truth_fls": 1,
            "p_oof": 0.0,
        }
    )


def teacher_label_config(
    tmp_path: Path,
    groups: list[str] | None = None,
    correction_kept: bool | None = None,
    nan_pixel: tuple[int, int] | None = None,
    without_era5: np.datetime64 | None = None,
    **dataset_changes,
) -> dict:
    """A config of a tiny dataset, a saved toy teacher and, with `correction_kept`, a held-out record.

    The tuning season ends with the second night, so the third night is outside it. `correction_kept` None
    writes no held-out record at all.
    """
    groups = groups or ["differences"]
    cfg = experiment_config(tmp_path)
    data = write_toy_dataset(tmp_path, nan_pixel)
    cfg["data"]["processed_path"] = str(tmp_path / "season_8ch.zarr")
    cfg["time"]["seasons"][0] |= {"start": "2023-11-20", "end": "2023-11-21"}
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["dem"]["out_path"] = str(tmp_path / "no_dem.zarr")

    era5 = None
    if "era5" in groups:
        era5 = era5_store(data, without=without_era5)
        era5.to_zarr(tmp_path / "era5.zarr", mode="w", zarr_format=3, consolidated=False)
        cfg["era5"]["out_path"] = str(tmp_path / "era5.zarr")

    length = LENGTH_KM if correction_kept else None
    teacher = fitted_toy_teacher(data, groups, era5, length)
    cfg["teacher"] = {
        "model_path": str(tmp_path / "teacher.joblib"),
        "ignore_halfwidth": 0.1,
        "idw_length_km": [LENGTH_KM],
        "feature_groups": groups,
        "probability_grid": [0.2, 0.8, 0.1],
    }
    save_teacher(teacher, Path(cfg["teacher"]["model_path"]))

    data_dir = Path(cfg["stations"]["data_dir"])
    feeder_table(data).to_csv(data_dir / "teacher_oof.csv", index=False)
    if correction_kept is not None:
        record = {"correction": {"keep": correction_kept}, "correction_km": length}
        (data_dir / "teacher_held_out.json").write_text(json.dumps(record), encoding="utf-8")
    return cfg


def label_store(cfg: dict, out_path: Path) -> xr.Dataset:
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), out_path, cfg)
    return xr.open_zarr(out_path, consolidated=False)


def rule_and_trust(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The rule's labels and the trusted pixels of every scene of the config's dataset."""
    groups = ["differences"]
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        static = read_static(data, Path("no_dem.zarr"), groups)
        nights = [
            night_features(data, scenes, static, None, TEST_PARAMS, groups) for scenes in night_spans(data)
        ]
    return np.concatenate([n.rule_labels for n in nights]), np.concatenate([n.trusted for n in nights])


def corrected_scenes(plain: xr.Dataset, corrected: xr.Dataset) -> np.ndarray:
    """Whether the correction changed the probability anywhere in each scene."""
    return (plain["fog_probability"] != corrected["fog_probability"]).any(("y", "x")).to_numpy()


def tuning_season_scenes(store: xr.Dataset, cfg: dict) -> np.ndarray:
    start, end = season_bounds(cfg, cfg["stations"]["tune_season"])
    times = store["time"].values
    return (times >= start) & (times < end)


def test_teacher_store_layout(tmp_path):
    cfg = teacher_label_config(tmp_path)
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    labels = store["label"].values
    assert store["label"].dtype == np.uint8 and store["fog_probability"].dtype == np.uint8
    assert set(np.unique(labels)) <= {0, 1, 2, 255}
    assert store.attrs["mode"] == "teacher" and store.attrs["teacher_p_star"] == 0.5
    assert (labels[:, 0, 0] == 1).all() and (labels[:, 6, 0] == 0).all()
    assert set(store["threshold_source"].values) == {"teacher"}
    assert np.allclose(store["fog_fraction"].values, 25 / 79)
    assert not (tmp_path / "labels_teacher.zarr.part").exists()


def test_attrs_record_the_teacher(tmp_path):
    cfg = teacher_label_config(tmp_path)
    attrs = label_store(cfg, tmp_path / "labels_teacher.zarr").attrs
    assert attrs["teacher_setting"] == "toy" and attrs["teacher_ignore_halfwidth"] == 0.1
    assert attrs["teacher_feature_groups"] == ["differences"]
    assert attrs["teacher_correction_km"] is None
    assert attrs["teacher_model_path"] == cfg["teacher"]["model_path"]
    assert attrs["btd_fixed"] == TEST_PARAMS.btd_fixed
    assert attrs["teacher_sklearn_version"] == sklearn.__version__


def test_probability_follows_the_label(tmp_path):
    cfg = teacher_label_config(tmp_path)
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    labels, probability = store["label"].values, store["fog_probability"].values
    assert (probability[labels == 1] >= 150).all() and (probability[labels == 0] <= 100).all()


def test_untrusted_and_obscured_follow_the_rule(tmp_path):
    cfg = teacher_label_config(tmp_path, nan_pixel=(2, 2))
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    labels = store["label"].values
    rule, trusted = rule_and_trust(cfg)
    assert (rule == OBSCURED).any() and not trusted.all()
    assert np.array_equal(labels == 2, rule == 2) and (labels[~trusted] == 255).all()
    assert (store["fog_probability"].values[labels == 2] == 255).all()


def test_missing_model_stops_before_labelling(tmp_path):
    cfg = teacher_label_config(tmp_path)
    Path(cfg["teacher"]["model_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    assert not list(tmp_path.glob("labels_teacher*"))


def test_mismatched_model_stops_before_labelling(tmp_path):
    cfg = teacher_label_config(tmp_path)
    cfg["teacher"]["feature_groups"] = ["bt"]
    with pytest.raises(ValueError, match="feature"):
        label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    assert not list(tmp_path.glob("labels_teacher*"))


def test_a_teacher_fitted_with_other_label_settings_stops_before_labelling(tmp_path):
    cfg = teacher_label_config(tmp_path)
    cfg["labels"]["btd_fixed"] += 1.0
    with pytest.raises(ValueError, match="btd_fixed"):
        label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    assert not list(tmp_path.glob("labels_teacher*"))


def test_the_dense_label_of_a_case_is_the_label_the_case_level_path_gives(tmp_path):
    cfg = teacher_label_config(tmp_path)
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    times = store["time"].values
    pixels = [(0, 0), (1, 3), (3, 1), (6, 0), (6, 6)]
    cases = pd.DataFrame([{"scene_time": time, "iy": iy, "ix": ix} for time in times for iy, ix in pixels])
    groups = cfg["teacher"]["feature_groups"]
    teacher = load_teacher(Path(cfg["teacher"]["model_path"]), groups)
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        static = read_static(data, Path(cfg["dem"]["out_path"]), groups)
        features = case_features(data, cases, static, None, TEST_PARAMS, groups)
    p = probability_of_cases(teacher, features)
    expected = teacher_labels(p, teacher.p_star, teacher.ignore_halfwidth)

    scene = pd.Index(times).get_indexer(cases["scene_time"])
    dense = store["label"].values[scene, cases["iy"].to_numpy(), cases["ix"].to_numpy()]
    assert np.array_equal(dense, expected)
    assert set(expected) >= {0, 1}


def test_an_existing_store_needs_overwrite(tmp_path):
    cfg = teacher_label_config(tmp_path)
    out_path = tmp_path / "labels_teacher.zarr"
    label_store(cfg, out_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        label_teacher_dataset(Path(cfg["data"]["processed_path"]), out_path, cfg)
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), out_path, cfg, overwrite=True)


def test_correction_only_when_kept_and_only_in_the_tuning_season(tmp_path):
    cfg = teacher_label_config(tmp_path, correction_kept=True)
    corrected = label_store(cfg, tmp_path / "labels_corrected.zarr")
    (Path(cfg["stations"]["data_dir"]) / "teacher_held_out.json").unlink()
    plain = label_store(cfg, tmp_path / "labels_plain.zarr")

    in_season = tuning_season_scenes(corrected, cfg)
    assert in_season.tolist() == [True] * 4 + [False] * 2
    assert np.array_equal(corrected_scenes(plain, corrected), in_season)
    assert (
        corrected.attrs["teacher_correction_km"] == LENGTH_KM and plain.attrs["teacher_correction_km"] is None
    )


def test_a_dropped_correction_changes_nothing(tmp_path):
    cfg = teacher_label_config(tmp_path, correction_kept=False)
    dropped = label_store(cfg, tmp_path / "labels_dropped.zarr")
    (Path(cfg["stations"]["data_dir"]) / "teacher_held_out.json").unlink()
    plain = label_store(cfg, tmp_path / "labels_plain.zarr")
    assert not corrected_scenes(plain, dropped).any()
    assert dropped.attrs["teacher_correction_km"] is None


def test_a_missing_held_out_record_means_no_correction(tmp_path, caplog):
    cfg = teacher_label_config(tmp_path)
    with caplog.at_level(logging.INFO, logger="fogseg"):
        store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    assert store.attrs["teacher_correction_km"] is None
    assert "teacher_held_out.json" in caplog.text


def test_nan_channel_is_ignored_without_warnings(tmp_path, recwarn):
    cfg = teacher_label_config(tmp_path, nan_pixel=(2, 2))
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    assert (store["label"].values[:, 2, 2] == 255).all() and (
        store["fog_probability"].values[:, 2, 2] == 255
    ).all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_scene_without_era5_is_ignored_but_for_its_obscured_pixels(tmp_path):
    scene = np.datetime64("2023-11-21T21:00", "ns")
    cfg = teacher_label_config(tmp_path, groups=["differences", "era5"], without_era5=scene)
    store = label_store(cfg, tmp_path / "labels_teacher.zarr")
    missing = store["time"].values == scene
    labels, probability = store["label"].values, store["fog_probability"].values

    assert missing.sum() == 1
    assert (probability[missing] == 255).all()
    obscured = np.zeros(labels.shape[1:], dtype=bool)
    obscured[COLD_PIXEL] = True
    assert (labels[missing][0][~obscured] == 255).all() and (labels[missing][0][obscured] == 2).all()
    assert (labels[~missing][:, 0, 0] == 1).all()


def test_cli_writes_the_teacher_store(tmp_path, caplog):
    cfg = teacher_label_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_label(argparse.Namespace(config=config_path, mode="teacher", overwrite=False))

    store = xr.open_zarr(tmp_path / "labels" / "labels_teacher.zarr", consolidated=False)
    assert store.attrs["mode"] == "teacher" and "6 scenes" in caplog.text
    assert "teacher" in caplog.text
