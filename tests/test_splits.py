import argparse
import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from fogseg.cli import cmd_split
from fogseg.config import load_config
from fogseg.splits import (
    assign_folds,
    cv_folds,
    held_out,
    held_out_part,
    make_splits,
    night_block_start,
    splits_path,
    station_cell,
)

STATIONS_CFG = {"night_block_days": 14, "station_cell_deg": 2.0, "n_folds": 5, "seed": 0}
BBOX = [2, 43, 20, 55]
SEASON = (np.datetime64("2023-10-01"), np.datetime64("2024-02-29"))


def test_night_block_start_is_the_first_night_of_the_two_week_block():
    nights = np.array(["2023-10-01", "2023-10-14", "2023-10-15", "2024-02-29"], dtype="datetime64[D]")
    starts = night_block_start(nights, np.datetime64("2023-10-01"), 14)
    expected = np.array(["2023-10-01", "2023-10-01", "2023-10-15", "2024-02-18"], dtype="datetime64[D]")
    assert (starts == expected).all()


def test_night_blocks_are_dealt_in_turn():
    splits = make_splits(*SEASON, BBOX, STATIONS_CFG)
    blocks = splits[splits["kind"] == "night_block"].sort_values("key")
    assert blocks["key"].iloc[0] == "2023-10-01"
    assert blocks["fold"].tolist()[:6] == [1, 2, 3, 4, 5, 1]


def test_station_cells_cover_the_bbox_and_are_balanced():
    cells = make_splits(*SEASON, BBOX, STATIONS_CFG)
    cells = cells[cells["kind"] == "station_cell"]
    assert len(cells) == 9 * 6
    assert cells["fold"].value_counts().max() - cells["fold"].value_counts().min() <= 1


def test_station_cells_are_dealt_by_the_seed():
    def folds(seed):
        splits = make_splits(*SEASON, BBOX, {**STATIONS_CFG, "seed": seed})
        return splits[splits["kind"] == "station_cell"].sort_values("key")["fold"].tolist()

    assert folds(0) == folds(0)
    assert folds(0) != folds(1)


def test_station_just_outside_the_bbox_gets_the_edge_cell():
    assert station_cell(np.array([1.95]), np.array([43.1]), BBOX, 2.0).tolist() == ["c0_0"]
    assert station_cell(np.array([20.0]), np.array([55.0]), BBOX, 2.0).tolist() == ["c8_5"]


def test_held_out_and_selection_are_disjoint_and_cv_never_trains_on_test():
    cases = pd.DataFrame({"night_fold": [1, 2, 5, 3, 1], "station_fold": [1, 5, 2, 3, 2]})
    assert held_out(cases, 5).tolist() == [False, True, True, False, False]
    assert held_out_part(cases, 5).tolist() == ["selection", "stations", "nights", "selection", "selection"]
    for k, train, test in cv_folds(cases, 5):
        assert not (train & test).any()
        assert not (train & held_out(cases, 5)).any()
        assert not (test & held_out(cases, 5)).any()
        assert ((cases["night_fold"] == k) | (cases["station_fold"] == k))[test].all()


def test_held_out_part_names_cases_unseen_in_both():
    cases = pd.DataFrame({"night_fold": [5], "station_fold": [5]})
    assert held_out_part(cases, 5).tolist() == ["both"]


def test_cv_folds_run_over_the_selection_folds():
    cases = pd.DataFrame({"night_fold": [1, 2, 3, 4], "station_fold": [4, 3, 2, 1]})
    assert [k for k, _, _ in cv_folds(cases, 5)] == [1, 2, 3, 4]


def test_assign_folds_reads_the_night_block_and_the_station_cell():
    splits = make_splits(*SEASON, BBOX, STATIONS_CFG)
    cases = pd.DataFrame(
        {"night": np.array(["2023-10-16"], dtype="datetime64[D]"), "lon": [5.0], "lat": [50.0]}
    )
    out = assign_folds(cases, splits, SEASON[0], BBOX, STATIONS_CFG)
    cell_fold = splits[(splits["kind"] == "station_cell") & (splits["key"] == "c1_3")]["fold"].iloc[0]
    assert out["night_fold"].tolist() == [2]
    assert out["station_fold"].tolist() == [cell_fold]
    assert "night_fold" not in cases


def test_assign_folds_accepts_nights_read_back_from_a_csv():
    splits = make_splits(*SEASON, BBOX, STATIONS_CFG)
    cases = pd.DataFrame({"night": ["2023-10-16"], "lon": [5.0], "lat": [50.0]})
    assert assign_folds(cases, splits, SEASON[0], BBOX, STATIONS_CFG)["night_fold"].tolist() == [2]


def test_assign_folds_names_a_missing_cell():
    splits = pd.DataFrame({"kind": ["night_block"], "key": ["2023-10-01"], "fold": [1]})
    cases = pd.DataFrame({"night": [np.datetime64("2023-10-02")], "lon": [5.0], "lat": [50.0]})
    with pytest.raises(ValueError, match="c1_3"):
        assign_folds(cases, splits, np.datetime64("2023-10-01"), BBOX, STATIONS_CFG)


def test_assign_folds_names_a_missing_block():
    splits = pd.DataFrame({"kind": ["station_cell"], "key": ["c1_3"], "fold": [1]})
    cases = pd.DataFrame({"night": [np.datetime64("2023-10-16")], "lon": [5.0], "lat": [50.0]})
    with pytest.raises(ValueError, match="2023-10-15"):
        assign_folds(cases, splits, np.datetime64("2023-10-01"), BBOX, STATIONS_CFG)


def split_args(tmp_path, season=None, overwrite=False):
    cfg = load_config("configs/default.yaml")
    cfg["stations"]["data_dir"] = str(tmp_path / "stations")
    cfg["data"]["log_dir"] = str(tmp_path / "logs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, season=season, overwrite=overwrite)


def test_split_writes_the_file_and_keeps_it_without_overwrite(tmp_path, caplog):
    args = split_args(tmp_path)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_split(args)

    path = splits_path(tmp_path / "stations", "2023-24")
    assert path == tmp_path / "stations" / "splits_2023-24.csv"
    splits = pd.read_csv(path, dtype={"key": str})
    assert set(splits["kind"]) == {"night_block", "station_cell"}
    assert splits["fold"].between(1, 5).all()
    assert str(path) in caplog.text

    with pytest.raises(FileExistsError, match="--overwrite"):
        cmd_split(args)

    args.overwrite = True
    cmd_split(args)


def test_split_refuses_the_test_season(tmp_path):
    with pytest.raises(PermissionError):
        cmd_split(split_args(tmp_path, season="2024-25"))
    assert not (tmp_path / "stations").exists()
