import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from test_unet_command import train_args, unet_config
from test_unet_predict import predict_args

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from fogseg.cli import cmd_evaluate, cmd_predict, cmd_train
from fogseg.experiment import SCORE_COLUMNS, held_out_parts
from fogseg.unet_eval import case_grid, comparison, held_out_table, labels_at_cases

START = np.datetime64("2023-11-20")
STATIONS_CFG = {"night_block_days": 1, "n_folds": 5, "bootstrap_samples": 50, "seed": 0}


def small_store() -> xr.Dataset:
    times = np.array(["2023-11-20T01:00", "2023-11-21T01:00", "2023-11-22T01:00"], dtype="datetime64[ns]")
    label = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    return xr.Dataset({"label": (("time", "y", "x"), label)}, coords={"time": times})


def held_out_cases() -> pd.DataFrame:
    """Eight held-out cases on three nights: two in the nights part, two in the stations part, two in both."""
    nights = pd.to_datetime(["2023-11-20", "2023-11-20", "2023-11-21", "2023-11-21"] + ["2023-11-22"] * 4)
    return pd.DataFrame(
        {
            "case": np.arange(8),
            "scene_time": nights + pd.Timedelta(hours=1),
            "night": nights,
            "iy": [0, 1, 2, 3, 0, 1, 2, 3],
            "ix": [0, 1, 2, 3, 4, 0, 1, 2],
            "night_fold": [5, 5, 1, 1, 5, 5, 5, 5],
            "station_fold": [1, 1, 5, 5, 5, 5, 2, 2],
            "truth_fls": [1, 1, 1, 0, 1, 0, 0, 1],
        }
    )


def sources() -> dict[str, np.ndarray]:
    truth = held_out_cases()["truth_fls"].to_numpy().astype(np.uint8)
    return {"rule": np.zeros(8, np.uint8), "teacher": truth, "unet": np.ones(8, np.uint8)}


def test_labels_at_cases_read_their_pixel():
    cases = held_out_cases().iloc[[0, 3, 5]]
    labels = labels_at_cases(small_store(), cases)
    assert labels.dtype == np.uint8
    assert labels.tolist() == [0, 1 * 20 + 3 * 5 + 3, 2 * 20 + 1 * 5 + 0]


def test_labels_at_cases_refuse_a_scene_missing_from_the_store():
    cases = held_out_cases().assign(scene_time=pd.Timestamp("2023-12-01 01:00"))
    with pytest.raises(ValueError, match="not in the store"):
        labels_at_cases(small_store(), cases)


def test_case_grid_has_one_value_and_the_labels():
    grid = case_grid(np.arange(8, dtype=np.uint8), held_out_cases())
    assert grid.columns.tolist() == ["value", "case", "label"]
    assert (grid["value"] == 0.0).all()
    assert grid["case"].tolist() == list(range(8))


def test_table_has_three_sources_and_four_parts():
    cases = held_out_cases()
    table = held_out_table(cases, sources(), 5)
    assert table.columns.tolist() == ["source", "part", *SCORE_COLUMNS[1:]]
    assert set(table["source"]) == {"rule", "teacher", "unet"}
    assert set(table["part"]) == {"held_out", "nights", "stations", "both"}
    sizes = {part: int(mask.sum()) for part, mask in held_out_parts(cases, 5).items()}
    unet = table[table["source"] == "unet"].set_index("part")
    assert unet["n_cases"].to_dict() == sizes
    assert sizes["held_out"] == sizes["nights"] + sizes["stations"] + sizes["both"]
    assert unet.loc["held_out", "hits"] == 5
    assert unet.loc["held_out", "false_alarms"] == 3


def test_comparison_of_a_source_with_itself_is_zero():
    result = comparison(held_out_cases(), sources(), "unet", "unet", STATIONS_CFG, START)
    assert result["pod_gain"] == 0
    assert (result["pod_gain_low"], result["pod_gain_high"]) == (0, 0)
    assert (result["far_difference_low"], result["far_difference_high"]) == (0, 0)


def test_comparison_reports_the_gain_and_both_fars():
    result = comparison(held_out_cases(), sources(), "teacher", "rule", STATIONS_CFG, START)
    assert result["pod_gain"] == 1.0
    assert result["pod_gain_low"] <= 1.0 <= result["pod_gain_high"]
    assert result["far_candidate"] == 0.0
    assert np.isnan(result["far_reference"])


def prepare(tmp_path: Path, mlflow_uri: str) -> dict:
    """The tiny config with a case table carrying the rule's label, trained and predicted."""
    cfg = unet_config(tmp_path, mlflow_uri)
    cases_path = tmp_path / "stations" / "cases_fixed_2023-24.csv"
    cases = pd.read_csv(cases_path)
    cases.assign(label=np.arange(len(cases)) % 2).to_csv(cases_path, index=False)
    cmd_train(train_args(cfg))
    cmd_predict(predict_args(cfg))
    return cfg


def evaluate_args(cfg: dict, season: str | None = None) -> argparse.Namespace:
    config_path = Path(cfg["stations"]["data_dir"]).parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, season=season)


def test_evaluate_writes_once_and_refuses_a_rerun(tmp_path, mlflow_uri):
    cfg = prepare(tmp_path, mlflow_uri)
    cmd_evaluate(evaluate_args(cfg))
    stations = tmp_path / "stations"
    table = pd.read_csv(stations / "unet_held_out.csv")
    assert set(table["source"]) == {"rule", "teacher", "unet"}
    assert len(table) == 12
    record = json.loads((stations / "unet_held_out.json").read_text(encoding="utf-8"))
    config = json.loads((stations / "unet_config.json").read_text(encoding="utf-8"))
    assert set(record["comparisons"]) == {"unet_vs_teacher", "unet_vs_rule"}
    assert record["p_star"] == config["p_star"]
    assert record["setting"] == config["winner"]
    assert record["note"] == "reported, not used for any choice"
    assert len(record["model_sha256"]) == 64

    metrics = mlflow.MlflowClient(mlflow_uri).get_run(config["final_run_id"]).data.metrics
    assert {"held_out_unet_pod", "held_out_rule_far", "held_out_teacher_csi"} <= set(metrics)
    assert "unet_vs_teacher_pod_gain" in metrics

    with pytest.raises(FileExistsError, match="once"):
        cmd_evaluate(evaluate_args(cfg))


def test_evaluate_refuses_predictions_of_another_model(tmp_path, mlflow_uri):
    cfg = prepare(tmp_path, mlflow_uri)
    model_path = Path(cfg["unet"]["model_path"])
    model_path.write_bytes(model_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="fogseg predict"):
        cmd_evaluate(evaluate_args(cfg))
    assert not (tmp_path / "stations" / "unet_held_out.csv").exists()


def test_evaluate_refuses_a_split_file_changed_since_training(tmp_path, mlflow_uri):
    cfg = prepare(tmp_path, mlflow_uri)
    split_path = tmp_path / "stations" / "splits_2023-24.csv"
    split_path.write_text(split_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"splits_2023-24\.csv.*fogseg train"):
        cmd_evaluate(evaluate_args(cfg))
    assert not (tmp_path / "stations" / "unet_held_out.csv").exists()


def test_evaluate_warns_when_the_teacher_row_differs_from_phase_c(tmp_path, mlflow_uri, caplog):
    cfg = prepare(tmp_path, mlflow_uri)
    counts = {"hits": 999, "misses": 999, "false_alarms": 999, "correct_negatives": 999}
    earlier = pd.DataFrame([{"rule": name, "part": "held_out", **counts} for name in ("rule", "teacher")])
    earlier.to_csv(tmp_path / "stations" / "teacher_held_out.csv", index=False)
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cmd_evaluate(evaluate_args(cfg))
    assert "teacher_held_out.csv" in caplog.text
    assert "teacher" in caplog.text


def test_evaluate_refuses_without_the_model_or_the_predictions(tmp_path, mlflow_uri):
    cfg = unet_config(tmp_path, mlflow_uri)
    with pytest.raises(FileNotFoundError, match="fogseg train"):
        cmd_evaluate(evaluate_args(cfg))


def test_evaluate_refuses_the_test_season(tmp_path, mlflow_uri):
    with pytest.raises(PermissionError):
        cmd_evaluate(evaluate_args(unet_config(tmp_path, mlflow_uri), season="2024-25"))
