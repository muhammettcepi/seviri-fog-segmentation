"""Tests for the diagnosis of the missed station fog."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
import zarr
from helpers import (
    GRID_LAT0,
    GRID_LON0,
    TEST_PARAMS,
    add_station_grid,
    label_scenes,
)

from fogseg.build import variable_length_strings
from fogseg.cli import cmd_diagnose, cmd_validate
from fogseg.config import load_config
from fogseg.diagnosis import (
    case_features,
    diagnosis_table,
    first_cause,
    read_selection_cases,
)
from fogseg.label_store import label_dataset, label_store_path
from fogseg.splits import make_splits, splits_path

PARAMS = TEST_PARAMS
SCENE_TIMES = pd.to_datetime(
    ["2023-11-20T21:00", "2023-11-21T03:00", "2023-11-21T21:00", "2023-11-22T03:00"]
).to_numpy()
CHANNELS = ["IR_039", "WV_062", "IR_108", "IR_120"]
SIZE = 4


def make_data(times: np.ndarray) -> xr.Dataset:
    """Scenes of 4 x 4 pixels: IR_108 270 K with one warm pixel, BTD 5 K, WV_062 250 K, deep night."""
    bt = np.empty((len(times), len(CHANNELS), SIZE, SIZE), dtype="float32")
    bt[:, CHANNELS.index("IR_039")] = 265.0
    bt[:, CHANNELS.index("WV_062")] = 250.0
    bt[:, CHANNELS.index("IR_108")] = 270.0
    bt[:, CHANNELS.index("IR_108"), 1, 1] = 273.0
    bt[:, CHANNELS.index("IR_120")] = 269.5
    sza = np.full((len(times), SIZE, SIZE), 120.0, dtype="float32")
    sza[:, 3, 3] = 100.0
    return xr.Dataset(
        {"bt": (("time", "channel", "y", "x"), bt), "sza": (("time", "y", "x"), sza)},
        coords={"time": times, "channel": CHANNELS},
    )


def make_labels(times: np.ndarray, fog_at: list[tuple[int, int, int]]) -> xr.Dataset:
    """Label scenes of no fog, with fog at the (scene, y, x) given."""
    label = np.zeros((len(times), SIZE, SIZE), dtype="uint8")
    for scene, y, x in fog_at:
        label[scene, y, x] = 1
    return xr.Dataset({"label": (("time", "y", "x"), label)}, coords={"time": times})


def make_era5(times: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"t2m": (("time", "y", "x"), np.full((len(times), SIZE, SIZE), 275.0))}, coords={"time": times}
    )


def pixel_cases(scene_times: np.ndarray, pixels: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_time": scene_times,
            "iy": [pixel[0] for pixel in pixels],
            "ix": [pixel[1] for pixel in pixels],
        }
    )


def test_first_cause_order():
    cases = pd.DataFrame(
        {
            "sza": [90, 100, 100, 100, 100, 100],
            "btd": [5.0, np.nan, 2.0, 4.2, 5.0, 5.0],
            "ir108_std3": [0.1, 0.1, 0.1, 0.1, 2.0, 0.1],
        }
    )
    assert first_cause(cases, PARAMS).tolist() == [
        "twilight",
        "no_data",
        "low_btd",
        "band",
        "texture",
        "small_blob_or_other",
    ]


def test_first_cause_band_needs_a_smooth_top():
    cases = pd.DataFrame({"sza": [100], "btd": [4.2], "ir108_std3": [2.0]})
    assert first_cause(cases, PARAMS).tolist() == ["texture"]


def test_features_read_the_channels_at_the_pixel_of_the_case():
    cases = pixel_cases(SCENE_TIMES[[1, 1]], [(1, 1), (3, 3)])

    features = case_features(
        make_data(SCENE_TIMES), make_labels(SCENE_TIMES, []), make_era5(SCENE_TIMES), cases
    )

    assert features["btd"].tolist() == [8.0, 5.0]
    assert features["ir108_minus_wv062"].tolist() == [23.0, 20.0]
    assert features["sza"].tolist() == [120.0, 100.0]
    assert features["t2m_minus_ir108"].tolist() == [2.0, 5.0]
    assert features["ir108_std3"][0] > 0.5
    assert features["ir108_std3"][1] == pytest.approx(0.0, abs=1e-3)


def test_features_without_era5_have_no_temperature_contrast():
    cases = pixel_cases(SCENE_TIMES[[0]], [(1, 1)])

    features = case_features(make_data(SCENE_TIMES), make_labels(SCENE_TIMES, []), None, cases)

    assert features["t2m_minus_ir108"].isna().all()
    assert features["btd"].notna().all()


def test_features_reject_a_case_at_a_scene_the_dataset_does_not_have():
    cases = pixel_cases(np.array([np.datetime64("2023-12-01T00:00")]), [(1, 1)])

    with pytest.raises(ValueError, match="dataset"):
        case_features(make_data(SCENE_TIMES), make_labels(SCENE_TIMES, []), None, cases)


def test_fog_other_slot_looks_only_inside_the_night():
    labels = make_labels(SCENE_TIMES, [(0, 1, 1), (2, 1, 1)])
    cases = pixel_cases(SCENE_TIMES, [(1, 1)] * 4)

    features = case_features(make_data(SCENE_TIMES), labels, None, cases)

    assert features["fog_other_slot"].tolist() == [False, True, False, True]


def test_fog_other_slot_needs_the_same_pixel():
    labels = make_labels(SCENE_TIMES, [(0, 2, 2)])
    cases = pixel_cases(SCENE_TIMES[[1]], [(1, 1)])

    features = case_features(make_data(SCENE_TIMES), labels, None, cases)

    assert features["fog_other_slot"].tolist() == [False]


def case_row(truth: int, label: int, **overrides) -> dict:
    return {
        "truth_fls": truth,
        "label": label,
        "hidden": False,
        "sza": 120.0,
        "btd": 2.0,
        "ir108_std3": 0.1,
        "ceiling_amount": "",
        "auto": pd.NA,
        "slot": "00:00",
        "fog_other_slot": False,
        "ir108_minus_wv062": 20.0,
        "t2m_minus_ir108": np.nan,
        "elevation_diff_m": 0.0,
    } | overrides


def cases_fixture() -> pd.DataFrame:
    rows = [
        case_row(1, 0, ceiling_amount="OVC", ir108_minus_wv062=10.0, fog_other_slot=True),
        case_row(1, 0, ceiling_amount="BKN", ir108_minus_wv062=20.0),
        case_row(0, 0, ir108_minus_wv062=30.0),
        case_row(0, 0, ir108_minus_wv062=40.0),
        case_row(1, 1, btd=5.0),
        case_row(1, 0, hidden=True),
        case_row(1, 2),
        case_row(1, 255, sza=90.0, btd=5.0),
        case_row(1, 255, btd=4.2),
    ]
    return pd.DataFrame(rows).astype({"auto": "boolean"})


def lookup(table: pd.DataFrame, section: str, variable: str, statistic: str) -> pd.Series:
    selected = table[
        (table["section"] == section) & (table["variable"] == variable) & (table["statistic"] == statistic)
    ]
    return selected.iloc[0]


def test_low_btd_section_compares_missed_fls_with_no_fls():
    table = diagnosis_table(cases_fixture(), PARAMS)

    row = lookup(table, "low_btd", "ceiling_amount", "share:OVC")

    assert (row["fls_missed"], row["no_fls"]) == (pytest.approx(0.5), pytest.approx(0.0))
    assert lookup(table, "low_btd", "ceiling_amount", "share:unknown")["no_fls"] == pytest.approx(1.0)
    assert lookup(table, "low_btd", "fog_other_slot", "share:True")["fls_missed"] == pytest.approx(0.5)


def test_low_btd_section_gives_quantiles_of_the_numeric_variables():
    table = diagnosis_table(cases_fixture(), PARAMS)

    row = lookup(table, "low_btd", "ir108_minus_wv062", "q50")

    assert (row["fls_missed"], row["no_fls"]) == (pytest.approx(15.0), pytest.approx(35.0))
    assert np.isnan(lookup(table, "low_btd", "t2m_minus_ir108", "q50")["fls_missed"])


def test_missing_flags_count_as_unknown_in_the_low_btd_section():
    table = diagnosis_table(cases_fixture(), PARAMS)

    assert lookup(table, "low_btd", "auto", "share:unknown")["fls_missed"] == pytest.approx(1.0)


def test_first_cause_section_counts_visible_misses_with_ignored_labels_among_them():
    table = diagnosis_table(cases_fixture(), PARAMS)

    first_cause_rows = table[table["section"] == "first_cause"].set_index(["variable", "statistic"])
    # the misses are the two low-BTD cases, the twilight case and the band case; hits, hidden and obscured are no misses
    assert first_cause_rows.loc[("low_btd", "count"), "fls_missed"] == 2
    assert first_cause_rows.loc[("twilight", "count"), "fls_missed"] == 1
    assert first_cause_rows.loc[("band", "share"), "fls_missed"] == pytest.approx(0.25)
    assert first_cause_rows.loc[("texture", "count"), "fls_missed"] == 0
    assert first_cause_rows["no_fls"].isna().all()


def test_low_btd_section_skips_twilight_cases():
    cases = pd.DataFrame([case_row(1, 255, sza=90.0, ceiling_amount="OVC")])

    table = diagnosis_table(cases, PARAMS)

    assert table[table["variable"] == "ceiling_amount"].empty


def test_diagnosis_without_misses_gives_empty_shares():
    table = diagnosis_table(pd.DataFrame([case_row(0, 0)]).astype({"auto": "boolean"}), PARAMS)

    assert lookup(table, "first_cause", "low_btd", "count")["fls_missed"] == 0
    assert np.isnan(lookup(table, "first_cause", "low_btd", "share")["fls_missed"])


def diagnose_config(tmp_path: Path, dataset_path: Path) -> Path:
    cfg = load_config("configs/default.yaml")
    cfg["data"]["processed_path"] = str(dataset_path)
    cfg["dem"]["out_path"] = str(tmp_path / "dem.zarr")
    cfg["era5"]["out_path"] = str(tmp_path / "era5.zarr")
    cfg["labels"]["output_dir"] = str(tmp_path / "labels")
    cfg["labels"]["mode"] = "fixed"
    cfg["stations"]["data_dir"] = str(tmp_path / "stations")
    cfg["stations"]["bootstrap_samples"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def diagnose_setup(tmp_path: Path) -> argparse.Namespace:
    """A dataset with WV_062, its fixed labels, three METAR stations and a split file that makes every case selection."""
    scenes = label_scenes(n_scenes=3, scene_sza=None, background_btd=1.8)
    wv062 = scenes["bt"].sel(channel=["IR_108"]).assign_coords(channel=["WV_062"])
    bt = xr.concat([scenes["bt"], wv062], dim="channel")
    dataset = add_station_grid(scenes.drop_vars(["bt", "channel"]).assign(bt=bt))
    dataset_path = tmp_path / "data.zarr"
    variable_length_strings(dataset).to_zarr(dataset_path, mode="w", zarr_format=3, consolidated=False)
    label_dataset(dataset_path, label_store_path(tmp_path / "labels", "fixed"), "fixed", TEST_PARAMS)

    stations = [
        ("HIT", GRID_LAT0 - 0.2, GRID_LON0 + 0.2, 500.0),
        ("MISS", GRID_LAT0 - 0.6, GRID_LON0 + 0.6, 500.0),
        ("CLEAR", GRID_LAT0 - 0.5, GRID_LON0 + 0.1, 9999.0),
    ]
    data_dir = tmp_path / "stations"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "source": "metar",
            "station": [station[0] for station in stations],
            "time": pd.Timestamp("2023-11-21T00:12"),
            "lat": [station[1] for station in stations],
            "lon": [station[2] for station in stations],
            "elevation_m": 100.0,
            "visibility_m": [station[3] for station in stations],
            "ceiling_m": np.inf,
        }
    ).to_csv(data_dir / "observations_metar.csv", index=False)

    config_path = diagnose_config(tmp_path, dataset_path)
    cfg = load_config(config_path)
    entry = cfg["time"]["seasons"][0]
    splits = make_splits(
        np.datetime64(entry["start"], "D"),
        np.datetime64(entry["end"], "D"),
        cfg["region"]["bbox"],
        cfg["stations"],
    )
    splits.assign(fold=1).to_csv(splits_path(data_dir, "2023-24"), index=False)
    return argparse.Namespace(config=config_path, season=None)


def validate_cases(args: argparse.Namespace) -> None:
    cmd_validate(argparse.Namespace(config=args.config, labels=None, season="2023-24", final=False))


def test_diagnose_writes_the_table_and_logs_the_first_causes(tmp_path, caplog):
    args = diagnose_setup(tmp_path)
    validate_cases(args)

    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_diagnose(args)

    path = tmp_path / "stations" / "diagnosis_2023-24.csv"
    table = pd.read_csv(path).set_index(["section", "variable", "statistic"])
    assert table.loc[("first_cause", "low_btd", "count"), "fls_missed"] == 1
    assert table.loc[("first_cause", "low_btd", "share"), "fls_missed"] == pytest.approx(1.0)
    assert table.loc[("low_btd", "ir108_minus_wv062", "q50"), "no_fls"] == pytest.approx(0.0)
    assert pd.isna(table.loc[("low_btd", "t2m_minus_ir108", "q50"), "fls_missed"])
    assert "low_btd 100.0%" in caplog.text
    assert "ERA5" in caplog.text
    assert str(path) in caplog.text


def test_diagnose_keeps_only_selection_cases(tmp_path):
    args = diagnose_setup(tmp_path)
    cfg = load_config(args.config)
    splits_file = splits_path(tmp_path / "stations", "2023-24")
    pd.read_csv(splits_file).assign(fold=cfg["stations"]["n_folds"]).to_csv(splits_file, index=False)
    validate_cases(args)

    cmd_diagnose(args)

    table = pd.read_csv(tmp_path / "stations" / "diagnosis_2023-24.csv").set_index(
        ["section", "variable", "statistic"]
    )
    assert table.loc[("first_cause", "low_btd", "count"), "fls_missed"] == 0


def test_read_selection_cases_drops_the_held_out_cases(tmp_path):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    cfg = load_config(args.config)
    stations_cfg = cfg["stations"]
    split_path = splits_path(tmp_path / "stations", "2023-24")
    kwargs = {
        "cases_path": tmp_path / "stations" / "cases_fixed_2023-24.csv",
        "split_path": split_path,
        "season_start": np.datetime64("2023-10-01"),
        "bbox": cfg["region"]["bbox"],
        "stations_cfg": stations_cfg,
    }

    assert len(read_selection_cases(**kwargs)) == 3

    splits = pd.read_csv(split_path)
    splits.loc[splits["kind"] == "station_cell", "fold"] = stations_cfg["n_folds"]
    splits.to_csv(split_path, index=False)
    assert read_selection_cases(**kwargs).empty


def test_read_selection_cases_counts_a_metar_and_synop_pair_at_one_pixel_and_scene_once(tmp_path):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    cfg = load_config(args.config)
    cases_path = tmp_path / "stations" / "cases_fixed_2023-24.csv"
    cases = pd.read_csv(cases_path, dtype={"station": str})
    synop_twin = cases[cases["station"] == "MISS"].assign(source="synop", station="10001")
    pd.concat([synop_twin, cases]).to_csv(cases_path, index=False)

    selection = read_selection_cases(
        cases_path,
        splits_path(tmp_path / "stations", "2023-24"),
        np.datetime64("2023-10-01"),
        cfg["region"]["bbox"],
        cfg["stations"],
    )

    assert sorted(selection["station"]) == ["CLEAR", "HIT", "MISS"]


def test_diagnose_refuses_dynamic_labels(tmp_path):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    cfg = load_config(args.config)
    cfg["labels"]["mode"] = "dynamic"
    args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match=r"fogseg diagnose.*fixed or teacher"):
        cmd_diagnose(args)
    assert not list((tmp_path / "stations").glob("diagnosis_*"))


def test_diagnose_in_teacher_mode_reads_the_fixed_store_and_cases(tmp_path):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    cfg = load_config(args.config)
    cfg["labels"]["mode"] = "teacher"
    args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd_diagnose(args)

    assert (tmp_path / "stations" / "diagnosis_2023-24.csv").exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [("texture", "blob"), ("fog_index", "emissivity"), ("split_window", "kma"), ("temporal_relax", 0.5)],
)
def test_diagnose_refuses_labels_of_another_rule(tmp_path, option, value):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    cfg = load_config(args.config)
    store = label_store_path(Path(cfg["labels"]["output_dir"]), "fixed")
    zarr.open_group(store, mode="r+").attrs[option] = value

    with pytest.raises(ValueError, match=rf"labels\.{option}.*default BTD rule"):
        cmd_diagnose(args)
    assert not list((tmp_path / "stations").glob("diagnosis_*"))


def test_diagnose_refuses_the_test_season(tmp_path):
    args = diagnose_setup(tmp_path)
    args.season = "2024-25"

    with pytest.raises(PermissionError, match="tuning season"):
        cmd_diagnose(args)
    assert not list((tmp_path / "stations").glob("diagnosis_*"))


def test_diagnose_without_a_split_file_names_the_command(tmp_path):
    args = diagnose_setup(tmp_path)
    validate_cases(args)
    splits_path(tmp_path / "stations", "2023-24").unlink()

    with pytest.raises(FileNotFoundError, match="fogseg split"):
        cmd_diagnose(args)


def test_diagnose_without_a_case_table_names_the_command(tmp_path):
    args = diagnose_setup(tmp_path)

    with pytest.raises(FileNotFoundError, match="fogseg validate"):
        cmd_diagnose(args)
