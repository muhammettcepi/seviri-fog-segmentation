from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from helpers import TEST_PARAMS, label_scenes

from fogseg.build import variable_length_strings
from fogseg.cli import cmd_experiment
from fogseg.config import load_config
from fogseg.experiment import (
    GRID_COLUMNS,
    PHASE_B_VARIANTS,
    KeepDecision,
    block_counts,
    check_rule_mode,
    combined_variants,
    config_changes,
    cross_validate,
    experiment_cases,
    held_out_comparison,
    log_winner,
    matched_value,
    run_variants,
    scored_cases,
    value_grid,
    value_scores,
    variant_grid,
    winner,
)
from fogseg.label_store import LABEL_CHANNELS
from fogseg.labels import FOG, IGNORE, label_night, label_scene
from fogseg.scores import paired_difference
from fogseg.splits import held_out_part, splits_path
from fogseg.validation import night_of, read_case_table

SEASON_START = np.datetime64("2023-10-01")
FIRST_SLOT = np.datetime64("2023-11-20T21:00", "ns")
SLOT = np.timedelta64(3, "h")
NEAR_MISS_BTD = 3.7
SCENE_OF_CASE_1 = 3
KMA_CLEAR_IR120 = 271.6  # IR_108 - IR_120 on the `kma` line at an IR_108 of 270 K
SLOT_OF_CASE_1 = 1


def small_cases(truth, nights=None, hidden=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truth_fls": truth,
            "night": pd.to_datetime(nights or ["2023-10-01"] * len(truth)),
            "hidden": hidden or [False] * len(truth),
        }
    )


def small_grid(value, labels) -> pd.DataFrame:
    return pd.DataFrame({"value": value, "case": np.arange(len(labels)), "label": labels})[GRID_COLUMNS]


def two_value_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four folds of 10 fog-free and 10 fog cases; value 3.0 has many false alarms in fold 1 only."""
    cases, rows = [], []
    for fold in range(1, 5):
        hits_3, false_alarms_3 = (2, 8) if fold == 1 else (10, 2)
        for truth, fog_at_3, fog_at_4 in ((0, false_alarms_3, 0), (1, hits_3, 5)):
            for i in range(10):
                cases.append({"truth_fls": truth, "hidden": False, "night_fold": fold, "station_fold": fold})
                rows.append((3.0, len(cases) - 1, int(i < fog_at_3)))
                rows.append((4.0, len(cases) - 1, int(i < fog_at_4)))
    return scored_cases(pd.DataFrame(cases)), pd.DataFrame(rows, columns=GRID_COLUMNS)


def test_scored_cases_drops_hidden_and_undefined_cases_and_numbers_the_rest():
    cases = small_cases([1, -1, 0, 1], hidden=[False, False, False, True])
    cases.index = [10, 11, 12, 13]

    scored = scored_cases(cases)

    assert scored["truth_fls"].tolist() == [1, 0]
    assert scored["case"].tolist() == [0, 1]
    assert scored.index.tolist() == [0, 1]


def test_value_scores_counts_an_ignored_label_as_no_and_follows_the_mask():
    cases = scored_cases(small_cases([1, 1, 0, 0]))
    grid = small_grid([1.0] * 4, [1, 255, 1, 0])

    everything = value_scores(grid, cases, pd.Series(True, index=cases.index)).iloc[0]
    first_three = value_scores(grid, cases, pd.Series([True, True, True, False])).iloc[0]

    assert everything[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [1, 1, 1, 1]
    assert everything["pod"] == 0.5
    assert everything["far"] == 0.5
    assert everything["n_cases"] == 4
    assert first_three[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [1, 1, 1, 0]


def test_value_scores_lists_every_value_of_the_grid_even_without_cases_in_the_mask():
    cases = scored_cases(small_cases([1, 0]))
    grid = pd.concat([small_grid([1.0, 1.0], [1, 0]), small_grid([2.0, 2.0], [0, 0])])

    scores = value_scores(grid, cases, pd.Series([False, False]))

    assert scores["value"].tolist() == [1.0, 2.0]
    assert scores["n_cases"].tolist() == [0, 0]
    assert scores["far"].isna().all()


def test_matched_value_is_the_highest_far_not_above_the_target():
    scores = pd.DataFrame({"value": [3.0, 4.0, 5.0], "far": [0.35, 0.27, 0.20], "pod": [0.7, 0.6, 0.5]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_skips_values_without_any_fog():
    scores = pd.DataFrame({"value": [4.0, 9.0], "far": [0.25, np.nan], "pod": [0.5, 0.0]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_falls_back_to_the_lowest_far():
    scores = pd.DataFrame({"value": [3.0, 4.0], "far": [0.5, 0.4], "pod": [0.8, 0.7]})
    assert matched_value(scores, 0.28) == (4.0, False)


def test_matched_value_breaks_a_tie_by_pod_and_then_by_the_lower_value():
    scores = pd.DataFrame(
        {"value": [3.0, 4.0, 5.0], "far": [0.25, 0.25, 0.25], "pod": [0.6, 0.7, 0.7]},
    )
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_refuses_scores_without_any_fog():
    scores = pd.DataFrame({"value": [4.0], "far": [np.nan], "pod": [0.0]})
    with pytest.raises(ValueError, match="any fog"):
        matched_value(scores, 0.28)


def test_each_fold_chooses_on_its_training_part():
    cases, grid = two_value_grid()  # value 3.0 reaches FAR <= 0.28 only on fold 1's training part
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    assert table.loc[1, "value"] == 3.0
    assert table.loc[2, "value"] == 4.0
    folds = table.drop(index="pooled")
    assert table.loc["pooled", "hits"] == folds["hits"].sum()
    assert folds["hits"].dtype.kind == "i"
    assert folds["n_cases"].dtype.kind == "i"


def test_the_pooled_row_scores_the_summed_test_counts():
    cases, grid = two_value_grid()
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    pooled = table.loc["pooled"]
    # fold 1 tests value 3.0 (2 hits, 8 false alarms); folds 2 to 4 test value 4.0 (5 hits, 0 false alarms)
    assert pooled[["hits", "misses", "false_alarms", "correct_negatives"]].tolist() == [17, 23, 8, 32]
    assert pooled["far"] == pytest.approx(8 / 25)
    assert pooled["n_cases"] == 80
    assert pooled["matched"]


@pytest.mark.parametrize("fold_column", ["night_fold", "station_fold"])
def test_held_out_cases_never_enter_the_cross_validation(fold_column):
    cases, grid = two_value_grid()
    fog_free_in_fold_4 = (cases["night_fold"] == 4) & (cases["truth_fls"] == 0)
    cases.loc[fog_free_in_fold_4, fold_column] = 5
    base = cross_validate(grid, cases, 5, 0.28)

    grid.loc[grid["case"].isin(cases.index[fog_free_in_fold_4]), "label"] = 1  # false alarms at both values

    assert cross_validate(grid, cases, 5, 0.28).equals(base)


def test_a_fold_that_falls_back_is_logged(caplog):
    cases, grid = two_value_grid()
    grid["label"] = 1  # every value has a FAR of 0.5
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    assert not table.loc[1, "matched"]
    assert not table.loc["pooled", "matched"]
    assert "Fold 1" in caplog.text
    assert "Fold 4" in caplog.text


def test_block_counts_has_one_row_per_night_block_in_order():
    cases = scored_cases(small_cases([1, 0, 1, 0], ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-03"]))
    grid = small_grid([1.0] * 4, [1, 1, 255, 0])
    everything = pd.Series(True, index=cases.index)

    counts = block_counts(grid, cases, 1.0, everything, SEASON_START, block_days=2)
    without_first = block_counts(
        grid, cases, 1.0, pd.Series([False, True, True, True]), SEASON_START, block_days=2
    )

    assert counts.tolist() == [[1, 0, 1, 0], [0, 1, 0, 1]]
    assert without_first.tolist() == [[0, 0, 1, 0], [0, 1, 0, 1]]


def tiny_season(tmp_path, ir120: float = 269.5, n_nights: int = 2) -> tuple[xr.Dataset, pd.DataFrame]:
    """`n_nights` nights of two scenes (21:00 and 00:00) and three cases, with IR_120 flat at `ir120` K.

    The fog patch of the last scene is a near miss at a threshold of 4 K (BTD `NEAR_MISS_BTD`), next to the
    full fog patch of the scene before it.
    """
    dataset = label_scenes(2 * n_nights, None, background_btd=1.8)
    dataset["bt"].loc[{"channel": "IR_120"}] = ir120
    night_starts = FIRST_SLOT + np.arange(n_nights) * np.timedelta64(1, "D")
    dataset["time"] = (night_starts[:, None] + np.arange(2) * SLOT).ravel()
    dataset["bt"][SCENE_OF_CASE_1, 0, :5, :5] = 270.0 - NEAR_MISS_BTD
    path = tmp_path / "season.zarr"
    variable_length_strings(dataset).to_zarr(path, mode="w", zarr_format=3, consolidated=False)

    scenes = dataset["time"].values[[0, SCENE_OF_CASE_1, 1]]
    cases = pd.DataFrame(
        {"scene_time": scenes, "night": night_of(scenes), "iy": [2, 2, 7], "ix": [2, 2, 9], "case": [0, 1, 2]}
    )
    return xr.open_zarr(path, consolidated=False), cases


def night_inputs(data: xr.Dataset, night: np.datetime64) -> tuple[np.ndarray, ...]:
    """IR_039, IR_108, IR_120 and sza of the stored scenes of one night."""
    scenes = data.isel(time=night_of(data["time"].values) == night)
    bt = scenes["bt"].sel(channel=LABEL_CHANNELS).values
    return bt[:, 0], bt[:, 1], bt[:, 2], scenes["sza"].values


def scene_inputs(data: xr.Dataset, time: np.datetime64) -> tuple[np.ndarray, ...]:
    """IR_039, IR_108, IR_120 and sza of one stored scene."""
    scene = data.sel(time=time)
    bt = scene["bt"].sel(channel=LABEL_CHANNELS).values
    return bt[0], bt[1], bt[2], scene["sza"].values


def test_value_grid_follows_the_fog_index():
    stations = {"btd_fixed_grid": [3.0, 4.0, 0.5], "emissivity_grid": [0.90, 1.00, 0.05]}
    emissivity = dataclasses.replace(TEST_PARAMS, fog_index="emissivity", emissivity_max=0.9)
    assert value_grid(TEST_PARAMS, stations).tolist() == [3.0, 3.5, 4.0]
    assert value_grid(emissivity, stations).tolist() == [0.9, 0.95, 1.0]


def test_variant_grid_without_relax_is_the_scene_label(tmp_path):
    data, cases = tiny_season(tmp_path)
    grid = variant_grid(data, cases, TEST_PARAMS, np.array([3.5, 4.5]))
    assert grid.columns.tolist() == GRID_COLUMNS
    assert len(grid) == 2 * len(cases)
    for value in (3.5, 4.5):
        scene_labels = label_scene(*scene_inputs(data, cases.loc[0, "scene_time"]), value, TEST_PARAMS)
        row = grid[(grid["case"] == 0) & (grid["value"] == value)]
        assert row["label"].item() == scene_labels[cases.loc[0, "iy"], cases.loc[0, "ix"]]


def test_variant_grid_relabels_whole_nights(tmp_path):
    data, cases = tiny_season(tmp_path)
    params = dataclasses.replace(TEST_PARAMS, temporal_relax=0.5)
    grid = variant_grid(data, cases, params, np.array([4.0]))
    iy, ix = cases.loc[1, "iy"], cases.loc[1, "ix"]
    night = label_night(*night_inputs(data, cases.loc[1, "night"]), np.full(2, 4.0), params)
    scene = label_scene(*scene_inputs(data, cases.loc[1, "scene_time"]), 4.0, params)
    assert scene[iy, ix] == IGNORE
    assert night[SLOT_OF_CASE_1, iy, ix] == FOG
    assert grid.loc[grid["case"] == 1, "label"].item() == FOG


def test_variant_grid_keeps_the_case_numbers_of_its_input(tmp_path):
    data, cases = tiny_season(tmp_path)
    cases["case"] = [7, 5, 6]
    grid = variant_grid(data, cases, TEST_PARAMS, np.array([4.0]))
    assert sorted(grid["case"]) == [5, 6, 7]


def test_case_of_a_missing_scene_is_an_error(tmp_path):
    data, cases = tiny_season(tmp_path)
    cases.loc[0, "scene_time"] = pd.Timestamp("2030-01-01T21:00")
    with pytest.raises(ValueError, match="not in the dataset"):
        variant_grid(data, cases, TEST_PARAMS, np.array([4.0]))


def test_read_case_table_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg validate"):
        read_case_table(tmp_path / "cases_fixed_2023-24.csv")


def test_variant_grid_of_no_cases_or_no_values_is_empty(tmp_path):
    data, cases = tiny_season(tmp_path)
    no_cases = variant_grid(data, cases.iloc[:0], TEST_PARAMS, np.array([4.0]))
    no_values = variant_grid(data, cases, TEST_PARAMS, np.array([]))
    assert no_cases.columns.tolist() == no_values.columns.tolist() == GRID_COLUMNS
    assert no_cases.empty
    assert no_values.empty


def pooled_table(pods: dict[str, float], unmatched: tuple[str, ...] = ()) -> pd.DataFrame:
    """A fold-1 row with a misleading POD and the pooled row of every variant; some are not matched."""
    rows = [
        {"variant": name, "fold": fold, "pod": pod, "matched": name not in unmatched}
        for name, pooled_pod in pods.items()
        for fold, pod in (("1", 1.0), ("pooled", pooled_pod))
    ]
    return pd.DataFrame(rows)


def test_combined_variants_take_the_best_texture_and_index():
    table = pooled_table(
        {
            "1 baseline": 0.40,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        }
    )
    combined = combined_variants(table, [0.5, 1.0])
    assert combined[0] == ("9 combined", {"texture": "ir108_mad5", "fog_index": "emissivity"})
    assert combined[2] == (
        "10 temporal relax 1.0",
        {"texture": "ir108_mad5", "fog_index": "emissivity", "temporal_relax": 1.0},
    )


def test_baseline_texture_can_win_variant_9():
    table = pooled_table(
        {
            "1 baseline": 0.46,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        }
    )
    assert combined_variants(table, [0.5])[0] == ("9 combined", {"texture": "ir108_std3", "fog_index": "btd"})


def test_the_baseline_of_variant_9_is_the_configured_one():
    table = pooled_table(
        {
            "1 baseline": 0.46,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        }
    )
    combined = combined_variants(table, [], baseline_texture="blob", baseline_index="emissivity")
    assert combined == [("9 combined", {"texture": "blob", "fog_index": "emissivity"})]


def test_the_winner_has_the_highest_pooled_pod_and_a_tie_goes_to_the_lower_number():
    table = pooled_table(
        {"1 baseline": 0.40, "7 emissivity": 0.45, "9 combined": 0.45, "10 temporal relax 0.5": 0.3}
    )
    assert winner(table) == "7 emissivity"


def test_an_unmatched_variant_cannot_win():
    table = pooled_table(
        {"1 baseline": 0.40, "7 emissivity": 0.45, "9 combined": 0.50}, unmatched=("9 combined",)
    )
    assert winner(table) == "7 emissivity"


def test_a_variant_without_a_pooled_pod_cannot_win():
    table = pooled_table({"1 baseline": 0.40, "7 emissivity": np.nan})
    assert winner(table) == "1 baseline"


def test_the_winner_needs_a_matched_variant():
    table = pooled_table({"1 baseline": 0.40, "7 emissivity": 0.45}, unmatched=("1 baseline", "7 emissivity"))
    with pytest.raises(ValueError, match=r"No variant.*matched"):
        winner(table)


def test_variant_9_ignores_unmatched_variants():
    table = pooled_table(
        {
            "1 baseline": 0.40,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        },
        unmatched=("3 texture ir108_mad5", "7 emissivity"),
    )
    combined = combined_variants(table, [])
    assert combined == [("9 combined", {"texture": "btd_std3", "fog_index": "btd"})]


def test_variant_9_needs_a_matched_variant_of_each_choice():
    pods = {
        "1 baseline": 0.40,
        "2 texture btd_std3": 0.42,
        "3 texture ir108_mad5": 0.45,
        "4 texture blob": 0.41,
        "5 texture none": 0.39,
        "7 emissivity": 0.44,
    }
    with pytest.raises(ValueError, match=r"No variant.*matched"):
        combined_variants(pooled_table(pods, unmatched=("1 baseline", "7 emissivity")), [])


def test_run_variants_marks_every_row_with_its_variant_and_options(tmp_path):
    cfg = experiment_config(tmp_path)
    data, cases = experiment_data(cfg)
    variants = [("1 baseline", {}), ("6 band 0.25", {"uncertainty_band": 0.25})]
    table = run_variants(data, cases, TEST_PARAMS, variants, cfg["stations"])
    assert table.columns[:2].tolist() == ["variant", "options"]
    assert table["variant"].unique().tolist() == ["1 baseline", "6 band 0.25"]
    assert table.loc[table["variant"] == "6 band 0.25", "options"].unique().tolist() == [
        '{"uncertainty_band": 0.25}'
    ]
    folds = table.loc[table["variant"] == "1 baseline", "fold"]
    assert folds.astype(str).tolist() == ["1", "2", "3", "4", "pooled"]


def test_a_variant_without_any_fog_is_recorded_as_unusable_and_the_run_goes_on(tmp_path, caplog):
    cfg = experiment_config(tmp_path)
    data, cases = experiment_data(cfg)
    variants = [("6 never fog", {"night_min_sza": 200.0}), ("1 baseline", {})]
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        table = run_variants(data, cases, TEST_PARAMS, variants, cfg["stations"])
    unusable = table[table["variant"] == "6 never fog"]
    assert unusable["fold"].astype(str).tolist() == ["1", "2", "3", "4", "pooled"]
    assert not unusable["matched"].any()
    assert unusable["pod"].isna().all()
    assert unusable["hits"].isna().all()
    assert table.loc[table["variant"] == "1 baseline", "pod"].notna().all()
    assert "variant 6 never fog" in caplog.text
    assert "Fold 1" in caplog.text
    assert "Fold 4" in caplog.text
    assert winner(table) == "1 baseline"


def test_a_fold_without_any_fog_in_its_training_part_is_unusable():
    cases, grid = two_value_grid()
    grid["label"] = 0
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28)
    assert table["pod"].isna().all()
    assert not table["matched"].any()


def test_a_matched_value_at_the_edge_of_the_grid_is_logged_with_its_variant(caplog):
    cases, grid = two_value_grid()  # fold 1 matches 3.0, the others 4.0: the first and the last value
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cross_validate(grid, cases, 5, 0.28, "6 band 0.25")
    edge_warnings = [message for message in caplog.messages if "edge" in message]
    assert len(edge_warnings) == 4
    assert "Fold 1 of variant 6 band 0.25" in edge_warnings[0]


def test_a_matched_value_inside_the_grid_is_not_an_edge(caplog):
    cases, grid = two_value_grid()
    grid = pd.concat([grid, grid[grid["value"] == 4.0].assign(value=5.0)], ignore_index=True)
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        cross_validate(grid, cases, 5, 0.28)
    edge_warnings = [message for message in caplog.messages if "edge" in message]
    assert len(edge_warnings) == 1
    assert edge_warnings[0].startswith("Fold 1")


def test_a_fold_fallback_names_its_variant(tmp_path, caplog):
    cfg = experiment_config(tmp_path)
    cfg["stations"]["matched_far"] = 0.0
    data, cases = experiment_data(cfg)
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        run_variants(data, cases, TEST_PARAMS, [("6 band 0.25", {"uncertainty_band": 0.25})], cfg["stations"])
    assert "variant 6 band 0.25" in caplog.text


CELL_A, CELL_B, CELL_C = (
    (5.0, 45.0),
    (9.0, 49.0),
    (7.0, 47.0),
)  # (lon, lat) in the station cells c1_1, c3_3, c2_2
PIXEL_CASES = [  # (iy, ix, truth, cell): the fog patch is the 5 x 5 corner of the scene
    (1, 1, 1, CELL_A),
    (3, 3, 1, CELL_B),
    (0, 4, 0, CELL_A),
    (4, 0, 0, CELL_B),
    (7, 9, 0, CELL_A),
    (6, 8, 0, CELL_B),
    (2, 2, 1, CELL_C),
    (4, 4, 1, CELL_C),
    (1, 3, 0, CELL_C),
    (7, 7, 0, CELL_C),
]
STATIONS_CFG = {
    "btd_fixed_grid": [3.0, 5.0, 0.5],
    "emissivity_grid": [0.8, 1.0, 0.1],
    "temporal_relax_grid": [0.5, 1.0],
    "night_block_days": 1,
    "n_folds": 5,
    "matched_far": 0.28,
    "bootstrap_samples": 200,
    "seed": 0,
}
START = np.datetime64("2023-11-20")


def experiment_config(tmp_path: Path) -> dict:
    """A tiny season on disk: dataset, case table and split file, the folds meeting in no selection case.

    Each night is a night block: nights 2023-11-20 and 2023-11-21 are in folds 1 and 2, night 2023-11-22 in
    the held-out fold 5. The station cells c1_1 and c3_3 are in folds 3 and 4, so every training part of the
    four folds has fog, false alarms and clear cases; the station cell c2_2 is in the held-out fold.
    """
    data, _ = tiny_season(tmp_path, ir120=KMA_CLEAR_IR120, n_nights=3)
    scenes = data["time"].values
    cases = pd.DataFrame(
        [
            {
                "scene_time": scene,
                "night": night_of(scene),
                "station": f"S{n}",
                "source": "metar",
                "lon": cell[0],
                "lat": cell[1],
                "iy": iy,
                "ix": ix,
                "truth_fls": truth,
                "hidden": False,
            }
            for scene in scenes
            for n, (iy, ix, truth, cell) in enumerate(PIXEL_CASES)
        ]
    )
    data_dir = tmp_path / "stations"
    data_dir.mkdir()
    cases.to_csv(data_dir / "cases_fixed_2023-24.csv", index=False)
    splits = pd.DataFrame(
        {
            "kind": ["night_block"] * 3 + ["station_cell"] * 3,
            "key": ["2023-11-20", "2023-11-21", "2023-11-22", "c1_1", "c3_3", "c2_2"],
            "fold": [1, 2, 5, 3, 4, 5],
        }
    )
    splits.to_csv(splits_path(data_dir, "2023-24"), index=False)

    cfg = load_config("configs/default.yaml")
    cfg["data"] |= {"processed_path": str(tmp_path / "season.zarr"), "log_dir": str(tmp_path / "logs")}
    cfg["labels"] |= dataclasses.asdict(TEST_PARAMS)
    cfg["time"]["seasons"][0] |= {"start": "2023-11-20", "end": "2023-11-22"}
    cfg["stations"] |= {**STATIONS_CFG, "data_dir": str(data_dir), "matched_far": 0.6}
    return cfg


def experiment_data(cfg: dict) -> tuple[xr.Dataset, pd.DataFrame]:
    """The dataset and the scored selection cases of `experiment_config`."""
    selection, _ = experiment_cases(cfg, "2023-24")
    return xr.open_zarr(cfg["data"]["processed_path"], consolidated=False), selection


def held_out_cases(cfg: dict) -> pd.DataFrame:
    """The scored cases of `experiment_config` that are in the held-out fold by night or by station."""
    return experiment_cases(cfg, "2023-24")[1]


def split_season(tmp_path: Path) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """The dataset of `experiment_config` with its scored selection cases and its scored held-out cases."""
    cfg = experiment_config(tmp_path)
    data, selection = experiment_data(cfg)
    return data, selection, held_out_cases(cfg)


def selection_grid(
    data: xr.Dataset, selection: pd.DataFrame, options: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The grid, cases and all-true mask of a variant on the selection cases."""
    params = dataclasses.replace(TEST_PARAMS, **options)
    grid = variant_grid(data, selection, params, value_grid(params, STATIONS_CFG))
    return grid, selection, pd.Series(True, index=selection.index)


def experiment_args(cfg: dict, season: str | None = None, held_out: bool = False) -> argparse.Namespace:
    config_path = Path(cfg["stations"]["data_dir"]).parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return argparse.Namespace(config=config_path, phase="B", season=season, held_out=held_out)


def test_experiment_writes_every_variant_and_fold(tmp_path, caplog):
    cfg = experiment_config(tmp_path)
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_experiment(experiment_args(cfg))
    table = pd.read_csv(tmp_path / "stations" / "experiment_B.csv")
    assert table["variant"].nunique() == 8 + 1 + len(cfg["stations"]["temporal_relax_grid"])
    assert set(table["fold"].astype(str)) == {"1", "2", "3", "4", "pooled"}
    assert "winner" in caplog.text


def test_experiment_refuses_the_test_season(tmp_path):
    with pytest.raises(PermissionError):
        cmd_experiment(experiment_args(experiment_config(tmp_path), season="2024-25"))


def test_experiment_needs_the_split_file(tmp_path):
    cfg = experiment_config(tmp_path)
    splits_path(tmp_path / "stations", "2023-24").unlink()
    with pytest.raises(FileNotFoundError, match="fogseg split"):
        cmd_experiment(experiment_args(cfg))


def test_the_variant_list_numbers_variants_one_to_eight_in_order():
    assert [name.split()[0] for name, _ in PHASE_B_VARIANTS] == [str(n) for n in range(1, 9)]


WINNER = ("6 band 0.25", {"uncertainty_band": 0.25})
RULE_AT_4_K = dataclasses.replace(TEST_PARAMS, btd_fixed=4.0)
HELD_OUT_PARTS = ["held_out", "nights", "stations", "both"]


def compare(tmp_path, params=RULE_AT_4_K, winner=WINNER) -> tuple[pd.DataFrame, KeepDecision]:
    data, selection, held_out = split_season(tmp_path)
    return held_out_comparison(data, selection, held_out, params, winner, STATIONS_CFG, START)


def test_keep_needs_a_pod_gain_above_zero_and_no_higher_far():
    assert KeepDecision("9 combined", 4.0, 0.05, 0.01, 0.09, 0.24, 0.25, -0.03, 0.01).keep
    assert not KeepDecision("9 combined", 4.0, 0.05, -0.01, 0.11, 0.24, 0.25, -0.03, 0.01).keep
    assert not KeepDecision("9 combined", 4.0, 0.05, 0.01, 0.09, 0.26, 0.25, -0.01, 0.03).keep


def test_a_fog_free_baseline_is_never_kept():
    assert not KeepDecision("9 combined", 4.0, 0.05, 0.01, 0.09, 0.24, np.nan, np.nan, np.nan).keep


def test_winner_threshold_is_chosen_on_selection_cases_only(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    winner = ("5 texture none", {"texture": "none"})
    _, decision = held_out_comparison(data, selection, held_out, TEST_PARAMS, winner, STATIONS_CFG, START)
    expected, _ = matched_value(value_scores(*selection_grid(data, selection, {"texture": "none"})), 0.28)
    assert decision.winner == "5 texture none"
    assert decision.value == expected


def test_held_out_cases_do_not_move_the_winner_threshold(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    args = (data, selection, held_out, RULE_AT_4_K, WINNER, STATIONS_CFG, START)
    _, decision = held_out_comparison(*args)
    held_out["truth_fls"] = 1 - held_out["truth_fls"]
    _, flipped = held_out_comparison(*args)
    assert flipped.value == decision.value


def test_the_baseline_is_the_configured_rule_at_its_threshold(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    table, _ = held_out_comparison(data, selection, held_out, RULE_AT_4_K, WINNER, STATIONS_CFG, START)
    baseline = table[(table["rule"] == "baseline") & (table["part"] == "held_out")].iloc[0]
    grid = variant_grid(data, held_out, RULE_AT_4_K, np.array([4.0]))
    expected = value_scores(grid, held_out, pd.Series(True, index=held_out.index)).iloc[0]
    assert baseline[expected.index].tolist() == expected.tolist()


def test_an_emissivity_baseline_is_scored_at_its_emissivity_threshold(tmp_path):
    emissivity = dataclasses.replace(TEST_PARAMS, fog_index="emissivity", emissivity_max=0.9)
    table, _ = compare(tmp_path, params=emissivity, winner=("7 emissivity", {}))
    assert table.loc[table["rule"] == "baseline", "value"].unique().tolist() == [0.9]


def test_an_emissivity_baseline_needs_its_threshold(tmp_path):
    emissivity = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    with pytest.raises(ValueError, match=r"labels\.emissivity_max"):
        compare(tmp_path, params=emissivity)


def test_the_table_scores_both_rules_on_every_part_of_the_held_out_set(tmp_path):
    table, _ = compare(tmp_path)
    assert table.columns[:2].tolist() == ["rule", "part"]
    assert table[["rule", "part"]].values.tolist() == [
        [rule, part] for rule in ("baseline", "winner") for part in HELD_OUT_PARTS
    ]
    for _, scores in table.groupby("rule"):
        n_cases = scores.set_index("part")["n_cases"]
        assert n_cases["held_out"] == n_cases[["nights", "stations", "both"]].sum() == 36
        assert (n_cases > 0).all()


def test_pod_gain_is_the_winner_minus_the_baseline_on_the_whole_held_out_set(tmp_path):
    table, decision = compare(tmp_path)
    whole = table[table["part"] == "held_out"].set_index("rule")
    assert decision.pod_gain == pytest.approx(whole.loc["winner", "pod"] - whole.loc["baseline", "pod"])
    assert decision.pod_gain > 0
    assert decision.pod_gain_low <= decision.pod_gain <= decision.pod_gain_high
    assert decision.far_winner == whole.loc["winner", "far"]
    assert decision.far_baseline == whole.loc["baseline", "far"]


def test_intervals_come_from_a_paired_bootstrap_over_the_held_out_night_blocks(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    _, decision = held_out_comparison(data, selection, held_out, RULE_AT_4_K, WINNER, STATIONS_CFG, START)
    everything = pd.Series(True, index=held_out.index)
    winner_params = dataclasses.replace(RULE_AT_4_K, uncertainty_band=0.25)
    winner_grid = variant_grid(data, held_out, winner_params, np.array([decision.value]))
    baseline_grid = variant_grid(data, held_out, RULE_AT_4_K, np.array([4.0]))
    winner = block_counts(winner_grid, held_out, decision.value, everything, START, 1)
    baseline = block_counts(baseline_grid, held_out, 4.0, everything, START, 1)
    assert len(winner) == 3
    samples, seed = STATIONS_CFG["bootstrap_samples"], STATIONS_CFG["seed"]
    assert (decision.pod_gain_low, decision.pod_gain_high) == paired_difference(
        winner, baseline, "pod", samples, seed
    )
    assert (decision.far_difference_low, decision.far_difference_high) == paired_difference(
        winner, baseline, "far", samples, seed
    )


def test_an_unreachable_far_on_the_selection_cases_is_logged(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        compare(tmp_path, winner=("5 texture none", {"texture": "none"}))
    assert "5 texture none" in caplog.text
    assert "selection" in caplog.text


def test_an_empty_held_out_set_is_an_error(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        held_out_comparison(data, selection, held_out.iloc[:0], RULE_AT_4_K, WINNER, STATIONS_CFG, START)


def test_the_config_changes_name_the_options_and_the_threshold_key():
    assert config_changes(TEST_PARAMS, {"texture": "none"}, 4.5) == {
        "labels.texture": "none",
        "labels.btd_fixed": 4.5,
    }
    assert config_changes(TEST_PARAMS, {"fog_index": "emissivity"}, 0.9) == {
        "labels.fog_index": "emissivity",
        "labels.emissivity_max": 0.9,
    }
    emissivity = dataclasses.replace(TEST_PARAMS, fog_index="emissivity")
    assert config_changes(emissivity, {}, 0.9) == {"labels.emissivity_max": 0.9}


def run_held_out(tmp_path) -> dict:
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    cmd_experiment(experiment_args(cfg, held_out=True))
    return cfg


def test_held_out_runs_once(tmp_path):
    cfg = run_held_out(tmp_path)
    with pytest.raises(FileExistsError, match="once"):
        cmd_experiment(experiment_args(cfg, held_out=True))


def test_held_out_needs_the_cross_validation_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiment --phase B"):
        cmd_experiment(experiment_args(experiment_config(tmp_path), held_out=True))


def test_held_out_refuses_the_test_season(tmp_path):
    with pytest.raises(PermissionError):
        cmd_experiment(experiment_args(experiment_config(tmp_path), season="2024-25", held_out=True))


def test_the_second_opening_is_refused_before_any_labelling(tmp_path):
    cfg = run_held_out(tmp_path)
    shutil.rmtree(cfg["data"]["processed_path"])
    with pytest.raises(FileExistsError, match="once"):
        cmd_experiment(experiment_args(cfg, held_out=True))


def test_held_out_writes_the_comparison_and_logs_the_decision(tmp_path, caplog):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_experiment(experiment_args(cfg, held_out=True))
    table = pd.read_csv(tmp_path / "stations" / "experiment_B_held_out.csv")
    assert table[["rule", "part"]].values.tolist() == [
        [rule, part] for rule in ("baseline", "winner") for part in HELD_OUT_PARTS
    ]
    assert table.loc[table["rule"] == "winner", "n_cases"].tolist() == [36, 12, 16, 8]
    assert "1 baseline" in caplog.text
    assert "POD gain" in caplog.text


def test_the_winner_and_its_options_come_from_the_cross_validation_table(tmp_path, monkeypatch):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    path = tmp_path / "stations" / "experiment_B.csv"
    table = pd.read_csv(path)
    table.loc[(table["variant"] == "5 texture none") & (table["fold"] == "pooled"), "pod"] = 2.0
    table.to_csv(path, index=False)
    chosen = []

    def comparison(*args):
        chosen.append(args[4])
        raise StopIteration

    monkeypatch.setattr("fogseg.experiment.held_out_comparison", comparison)
    with pytest.raises(StopIteration):
        cmd_experiment(experiment_args(cfg, held_out=True))
    assert chosen == [("5 texture none", {"texture": "none"})]


def test_a_kept_winner_logs_the_config_keys_to_set(tmp_path, caplog, monkeypatch):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    kept = KeepDecision("1 baseline", 4.5, 0.1, 0.05, 0.15, 0.2, 0.3, -0.2, 0.0)
    monkeypatch.setattr("fogseg.experiment.held_out_comparison", lambda *args: (pd.DataFrame(), kept))
    with caplog.at_level(logging.INFO, logger="fogseg"):
        cmd_experiment(experiment_args(cfg, held_out=True))
    assert "labels.btd_fixed" in caplog.text
    assert "4.5" in caplog.text


def test_a_failed_comparison_does_not_open_the_held_out_set(tmp_path, monkeypatch):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))

    def fail(*args):
        raise ValueError("no fog")

    with monkeypatch.context() as patched:
        patched.setattr("fogseg.experiment.held_out_comparison", fail)
        with pytest.raises(ValueError, match="no fog"):
            cmd_experiment(experiment_args(cfg, held_out=True))
    assert not (tmp_path / "stations" / "experiment_B_held_out.csv").exists()


def test_held_out_cases_do_not_change_the_cross_validation(tmp_path):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    path = tmp_path / "stations" / "experiment_B.csv"
    before = path.read_text(encoding="utf-8")

    case_path = tmp_path / "stations" / "cases_fixed_2023-24.csv"
    cases = pd.read_csv(case_path)
    held_out_rows = (cases["night"] == "2023-11-22") | (cases["lon"] == CELL_C[0])
    cases.loc[held_out_rows, "truth_fls"] = 1 - cases.loc[held_out_rows, "truth_fls"]
    cases.to_csv(case_path, index=False)
    cmd_experiment(experiment_args(cfg))

    assert path.read_text(encoding="utf-8") == before


def held_out_files(tmp_path) -> list[str]:
    return sorted(path.name for path in (tmp_path / "stations").glob("experiment_B_held_out*"))


def test_held_out_leaves_the_table_and_the_decision_record(tmp_path):
    run_held_out(tmp_path)
    stations = tmp_path / "stations"
    record = json.loads((stations / "experiment_B_held_out.json").read_text(encoding="utf-8"))
    table = pd.read_csv(stations / "experiment_B.csv")
    best = winner(table)
    assert record["winner"] == best
    assert record["options"] == json.loads(table.loc[table["variant"] == best, "options"].iloc[0])
    assert record["baseline_fog_index"] == "btd"
    assert record["baseline_threshold"] == TEST_PARAMS.btd_fixed
    assert set(record) >= {field.name for field in dataclasses.fields(KeepDecision)}
    assert isinstance(record["keep"], bool)
    assert "config_changes" in record
    assert held_out_files(tmp_path) == ["experiment_B_held_out.csv", "experiment_B_held_out.json"]


def test_a_crash_while_writing_leaves_no_held_out_file_and_the_decision_is_logged(
    tmp_path, monkeypatch, caplog
):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))

    def crash(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patched, caplog.at_level(logging.INFO, logger="fogseg"):
        patched.setattr("fogseg.cli.os.replace", crash)
        with pytest.raises(OSError, match="disk full"):
            cmd_experiment(experiment_args(cfg, held_out=True))
    assert "POD gain" in caplog.text
    assert "Held-out baseline" in caplog.text
    assert not [name for name in held_out_files(tmp_path) if not name.endswith(".part")]


@pytest.mark.parametrize("name", ["experiment_B_held_out.csv", "experiment_B_held_out.json"])
def test_either_held_out_file_closes_the_held_out_set(tmp_path, name):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    shutil.rmtree(cfg["data"]["processed_path"])
    (tmp_path / "stations" / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="once"):
        cmd_experiment(experiment_args(cfg, held_out=True))


@pytest.mark.parametrize("name", ["experiment_B_held_out.csv", "experiment_B_held_out.json"])
def test_the_cross_validation_does_not_overwrite_a_held_out_record(tmp_path, name):
    cfg = experiment_config(tmp_path)
    (tmp_path / "stations" / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="held-out"):
        cmd_experiment(experiment_args(cfg))
    assert not (tmp_path / "stations" / "experiment_B.csv").exists()


def test_the_cross_validation_records_its_configuration(tmp_path):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    record = json.loads((tmp_path / "stations" / "experiment_B_config.json").read_text(encoding="utf-8"))
    assert record["season"] == "2023-24"
    assert record["labels"] == dataclasses.asdict(TEST_PARAMS)
    assert set(record["stations"]) == {
        "matched_far",
        "n_folds",
        "seed",
        "bootstrap_samples",
        "night_block_days",
        "station_cell_deg",
        "btd_fixed_grid",
        "emissivity_grid",
        "temporal_relax_grid",
    }
    assert record["stations"]["matched_far"] == cfg["stations"]["matched_far"]


def test_held_out_refuses_a_configuration_that_differs_from_the_cross_validation(tmp_path):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    shutil.rmtree(cfg["data"]["processed_path"])
    cfg["stations"]["matched_far"] = 0.3
    cfg["labels"]["btd_fixed"] = 4.0
    with pytest.raises(ValueError, match=r"(?s)labels\.btd_fixed.*stations\.matched_far"):
        cmd_experiment(experiment_args(cfg, held_out=True))
    assert held_out_files(tmp_path) == []


def test_held_out_needs_the_configuration_of_the_cross_validation(tmp_path):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    (tmp_path / "stations" / "experiment_B_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="experiment --phase B"):
        cmd_experiment(experiment_args(cfg, held_out=True))


def test_a_held_out_match_at_the_edge_of_the_grid_is_logged_with_the_winner(tmp_path, caplog):
    stations_cfg = {**STATIONS_CFG, "matched_far": 0.6}
    data, selection, held_out = split_season(tmp_path)
    with caplog.at_level(logging.WARNING, logger="fogseg"):
        held_out_comparison(data, selection, held_out, RULE_AT_4_K, WINNER, stations_cfg, START)
    assert "Winner 6 band 0.25" in caplog.text
    assert "edge of the grid" in caplog.text


def log_table(far: float) -> pd.DataFrame:
    rows = [
        {"variant": name, "fold": "pooled", "pod": 0.5, "far": far, "matched": True}
        for name in ("1 baseline", "7 emissivity")
    ]
    return pd.DataFrame(rows)


def test_a_winner_above_the_matched_far_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="fogseg"):
        log_winner(log_table(far=0.31), 0.28)
    assert "winner 1 baseline" in caplog.text
    assert [record.levelname for record in caplog.records] == ["INFO", "WARNING"]
    assert "0.310" in caplog.records[1].getMessage()


def test_a_winner_within_the_matched_far_is_not_warned_about(caplog):
    with caplog.at_level(logging.INFO, logger="fogseg"):
        log_winner(log_table(far=0.25), 0.28)
    assert [record.levelname for record in caplog.records] == ["INFO"]


def test_the_held_out_and_selection_tables_cannot_be_swapped(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    args = (RULE_AT_4_K, WINNER, STATIONS_CFG, START)
    with pytest.raises(ValueError, match=r"held-out.*selection"):
        held_out_comparison(data, selection, selection, *args)
    with pytest.raises(ValueError, match=r"selection.*held-out"):
        held_out_comparison(data, held_out, held_out, *args)


def test_experiment_cases_split_the_scored_cases_by_the_frozen_folds(tmp_path):
    selection, held_out = experiment_cases(experiment_config(tmp_path), "2023-24")
    assert (len(selection), len(held_out)) == (24, 36)
    assert selection["case"].tolist() == list(range(24))
    assert held_out["case"].tolist() == list(range(36))
    assert (held_out_part(selection, 5) == "selection").all()
    assert (held_out_part(held_out, 5) != "selection").all()


def test_experiment_cases_read_the_rule_cases_in_teacher_mode(tmp_path):
    cfg = experiment_config(tmp_path)
    fixed_selection, fixed_held_out = experiment_cases(cfg, "2023-24")
    cfg["labels"]["mode"] = "teacher"

    selection, held_out = experiment_cases(cfg, "2023-24")

    pd.testing.assert_frame_equal(selection, fixed_selection)
    pd.testing.assert_frame_equal(held_out, fixed_held_out)


def test_check_rule_mode_refuses_dynamic():
    check_rule_mode("fixed", "fogseg teacher")
    check_rule_mode("teacher", "fogseg teacher")
    with pytest.raises(ValueError, match=r"fogseg teacher.*fixed or teacher"):
        check_rule_mode("dynamic", "fogseg teacher")
