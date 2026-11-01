"""Compare label variants at a matched false-alarm ratio: chosen on training folds, scored on test folds."""

from __future__ import annotations

import dataclasses
import json
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.label_store import LABEL_CHANNELS
from fogseg.labels import LabelParams, label_night
from fogseg.scores import COUNT_NAMES, contingency, counts_by_night, paired_difference, skill
from fogseg.splits import cv_folds, held_out_part, night_block_start
from fogseg.tuning import grid_values
from fogseg.validation import TRUTH_NO, TRUTH_YES, night_of

log = logging.getLogger(__name__)

GRID_COLUMNS = ["value", "case", "label"]
SCORE_COLUMNS = ["value", *COUNT_NAMES, "pod", "far", "csi", "n_cases"]
CV_COLUMNS = ["fold", "value", "matched", *SCORE_COLUMNS[1:]]

PHASE_B_VARIANTS: list[tuple[str, dict]] = [
    ("1 baseline", {}),
    ("2 texture btd_std3", {"texture": "btd_std3"}),
    ("3 texture ir108_mad5", {"texture": "ir108_mad5"}),
    ("4 texture blob", {"texture": "blob"}),
    ("5 texture none", {"texture": "none"}),
    ("6 band 0.25", {"uncertainty_band": 0.25}),
    ("7 emissivity", {"fog_index": "emissivity"}),
    ("8 split window kma", {"split_window": "kma"}),
]
TEXTURE_VARIANTS = [name for name, _ in PHASE_B_VARIANTS[:5]]
FOG_INDEX_VARIANTS = ["1 baseline", "7 emissivity"]


def scored_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """The cases that are scored: not hidden, truth defined; `case` numbers them 0..n-1 in a fresh index."""
    is_scored = ~cases["hidden"] & cases["truth_fls"].isin((TRUTH_NO, TRUTH_YES))
    scored = cases[is_scored].reset_index(drop=True)
    return scored.assign(case=np.arange(len(scored)))


def value_grid(params: LabelParams, stations_cfg: dict) -> np.ndarray:
    """The thresholds a variant is tried at: the BTD grid, or the emissivity grid for the emissivity index."""
    name = "emissivity_grid" if params.fog_index == "emissivity" else "btd_fixed_grid"
    return grid_values(stations_cfg[name])


def variant_grid(
    data: xr.Dataset,
    cases: pd.DataFrame,
    params: LabelParams,
    values: np.ndarray,
) -> pd.DataFrame:
    """The label of every case at every value, from labelling the whole night of the case with that value.

    Each night with a case is read once. All its stored scenes get the same threshold, so that the
    temporal relaxation sees the neighbouring scenes as the label store does.
    """
    times = data["time"].values
    scene = pd.Index(times).get_indexer(cases["scene_time"])
    if (scene < 0).any():
        missing = cases["scene_time"].to_numpy()[scene < 0]
        raise ValueError(f"{len(missing)} case scenes are not in the dataset, the first is {missing[0]}")

    case_night = night_of(times[scene])
    dataset_night = night_of(times)
    parts = []
    for night in np.unique(case_night):
        in_night = np.flatnonzero(dataset_night == night)
        scenes = data.isel(time=in_night)
        ir039, ir108, ir120 = np.moveaxis(scenes["bt"].sel(channel=LABEL_CHANNELS).values, 1, 0)
        sza = scenes["sza"].values

        of_night = cases[case_night == night]
        slot = np.searchsorted(in_night, scene[case_night == night])
        for value in values:
            labels = label_night(ir039, ir108, ir120, sza, np.full(len(in_night), value), params)
            picked = labels[slot, of_night["iy"].to_numpy(), of_night["ix"].to_numpy()]
            parts.append(pd.DataFrame({"value": value, "case": of_night["case"].to_numpy(), "label": picked}))
    if not parts:
        return pd.DataFrame(columns=GRID_COLUMNS)
    grid = pd.concat(parts, ignore_index=True)
    return grid.sort_values(["value", "case"], ignore_index=True)[GRID_COLUMNS]


def score_columns(counts: np.ndarray) -> dict:
    """The counts with their POD, FAR, CSI and case count: `SCORE_COLUMNS` without the value."""
    scores = skill(counts)
    return {
        **dict(zip(COUNT_NAMES, counts.tolist(), strict=True)),
        "pod": float(scores["pod"]),
        "far": float(scores["far"]),
        "csi": float(scores["csi"]),
        "n_cases": int(counts.sum()),
    }


def value_scores(grid: pd.DataFrame, cases: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Full-count counts and scores of every grid value over the cases where `mask` is true."""
    chosen = grid[mask.to_numpy()[grid["case"].to_numpy()]]
    label = chosen["label"].to_numpy()
    truth = cases["truth_fls"].to_numpy()[chosen["case"].to_numpy()]
    rows = []
    for value in np.sort(grid["value"].unique()):
        at_value = (chosen["value"] == value).to_numpy()
        counts = contingency(label[at_value], truth[at_value], full_count=True)
        rows.append({"value": value, **score_columns(counts)})
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def matched_value(scores: pd.DataFrame, far_target: float) -> tuple[float, bool]:
    """The value with the highest FAR not above the target, and whether there is one.

    Ties go to the higher POD, then to the lower value. If every FAR is above the target, the value with the
    lowest FAR is returned with False. A value that labels no fog has no FAR and is never chosen.
    """
    defined = scores.dropna(subset=["far"])
    if defined.empty:
        raise ValueError("No value labels any fog in the cases the value is chosen on")
    within_target = defined[defined["far"] <= far_target]
    qualifies = not within_target.empty
    candidates = within_target if qualifies else defined
    ranked = candidates.sort_values(["far", "pod", "value"], ascending=[not qualifies, False, True])
    return float(ranked["value"].iloc[0]), qualifies


def cross_validate(
    grid: pd.DataFrame, cases: pd.DataFrame, n_folds: int, far_target: float, variant: str = ""
) -> pd.DataFrame:
    """Per fold the value matched on the training part and its test scores, then the pooled test folds.

    Cases of the held-out fold are in no training or test part. `variant` names the variant in the warnings.
    """
    of_variant = f" of variant {variant}" if variant else ""
    rows, test_counts = [], []
    for fold, train, test in cv_folds(cases, n_folds):
        value, matched = matched_value(value_scores(grid, cases, train), far_target)
        if not matched:
            log.warning(
                "Fold %d%s: no value reaches a FAR of %.3f on its training part; using %g",
                fold,
                of_variant,
                far_target,
                value,
            )
        at_value = grid[grid["value"] == value]
        scores = value_scores(at_value, cases, test).to_dict("records")[0]
        test_counts.append([scores[name] for name in COUNT_NAMES])
        rows.append({"fold": fold, "matched": matched, **scores})
    pooled = {
        "fold": "pooled",
        "value": np.nan,
        "matched": all(row["matched"] for row in rows),
        **score_columns(np.sum(test_counts, axis=0)),
    }
    return pd.DataFrame([*rows, pooled], columns=CV_COLUMNS)


def block_counts(
    grid: pd.DataFrame,
    cases: pd.DataFrame,
    value: float,
    mask: pd.Series,
    season_start: np.datetime64,
    block_days: int,
) -> np.ndarray:
    """Full-count counts of one value per night block over the cases in `mask`, shape (n_blocks, 4)."""
    in_mask = mask.to_numpy()
    block = night_block_start(cases["night"].to_numpy(), season_start, block_days)
    at_value = grid[grid["value"] == value]
    selected = at_value[in_mask[at_value["case"].to_numpy()]]
    in_case = selected["case"].to_numpy()
    return counts_by_night(
        block[in_case],
        selected["label"].to_numpy(),
        cases["truth_fls"].to_numpy()[in_case],
        np.unique(block[in_mask]),
        full_count=True,
    )


def pooled_scores(table: pd.DataFrame) -> pd.DataFrame:
    """The pooled row of every variant, indexed by variant name."""
    pooled = table[table["fold"].astype(str) == "pooled"]
    return pooled.set_index("variant")


def best_variant(pooled: pd.DataFrame) -> str:
    """The variant with the highest pooled POD; ties go to the lower variant number."""
    number = pooled.index.str.split().str[0].astype(int)
    ranked = pooled.assign(number=number).sort_values(["pod", "number"], ascending=[False, True])
    return ranked.index[0]


def winner(table: pd.DataFrame) -> str:
    """The variant of the table with the highest pooled POD; ties go to the lower variant number."""
    return best_variant(pooled_scores(table))


def combined_variants(
    table: pd.DataFrame,
    relax_grid: list[float],
    baseline_texture: str = "ir108_std3",
    baseline_index: str = "btd",
) -> list[tuple[str, dict]]:
    """Variant 9, the best texture with the better fog index, then variant 9 with each temporal relaxation.

    The best texture is chosen among variants 1 to 5 and the better fog index between 1 and 7, by pooled POD.
    The baseline variant uses `baseline_texture` and `baseline_index`.
    """
    pooled = pooled_scores(table)
    options = dict(PHASE_B_VARIANTS)
    texture = options[best_variant(pooled.loc[TEXTURE_VARIANTS])].get("texture", baseline_texture)
    fog_index = options[best_variant(pooled.loc[FOG_INDEX_VARIANTS])].get("fog_index", baseline_index)
    combined = {"texture": texture, "fog_index": fog_index}
    relaxed = [(f"10 temporal relax {relax}", {**combined, "temporal_relax": relax}) for relax in relax_grid]
    return [("9 combined", combined), *relaxed]


def run_variants(
    data: xr.Dataset,
    cases: pd.DataFrame,
    params: LabelParams,
    variants: list[tuple[str, dict]],
    stations_cfg: dict,
) -> pd.DataFrame:
    """The cross-validation rows of every variant, with its name and its overrides of `params` in front."""
    tables = []
    for name, options in variants:
        started = time.perf_counter()
        variant_params = dataclasses.replace(params, **options)
        grid = variant_grid(data, cases, variant_params, value_grid(variant_params, stations_cfg))
        table = cross_validate(grid, cases, stations_cfg["n_folds"], stations_cfg["matched_far"], name)
        pooled = pooled_scores(table.assign(variant=name)).loc[name]
        log.info(
            "Variant %s: pooled POD %.3f, FAR %.3f, %.0f s",
            name,
            pooled["pod"],
            pooled["far"],
            time.perf_counter() - started,
        )
        tables.append(table.assign(variant=name, options=json.dumps(options)))
    return pd.concat(tables, ignore_index=True)[["variant", "options", *CV_COLUMNS]]


@dataclasses.dataclass(frozen=True)
class KeepDecision:
    """The held-out verdict on the winner: its POD gain over the baseline and both FARs, with 95 % intervals."""

    winner: str
    value: float
    pod_gain: float
    pod_gain_low: float
    pod_gain_high: float
    far_winner: float
    far_baseline: float
    far_difference_low: float
    far_difference_high: float

    @property
    def keep(self) -> bool:
        """The POD gain is above zero in the whole interval and the FAR is no higher than the baseline's."""
        return self.pod_gain_low > 0 and self.far_winner <= self.far_baseline


def threshold_name(fog_index: str) -> str:
    """The `labels:` key of the threshold of a fog index."""
    return "emissivity_max" if fog_index == "emissivity" else "btd_fixed"


def config_changes(params: LabelParams, options: dict, value: float) -> dict:
    """The `labels:` keys that make a variant the config's rule: its options and the threshold of its fog index."""
    threshold = threshold_name(options.get("fog_index", params.fog_index))
    return {**{f"labels.{key}": option for key, option in options.items()}, f"labels.{threshold}": value}


def held_out_comparison(
    data: xr.Dataset,
    selection: pd.DataFrame,
    held_out: pd.DataFrame,
    params: LabelParams,
    winner: tuple[str, dict],
    stations_cfg: dict,
    season_start: np.datetime64,
) -> tuple[pd.DataFrame, KeepDecision]:
    """Score the winner against the config's rule on the held-out cases, and decide whether to keep it.

    Both tables hold scored cases numbered from 0. The winner's threshold is matched to `matched_far` on all
    selection cases; the baseline keeps the threshold of `params`. The intervals are paired bootstraps over
    the night blocks of the held-out cases.
    """
    if held_out.empty:
        raise ValueError("The held-out set has no scored cases")
    baseline_value = getattr(params, threshold_name(params.fog_index))
    if baseline_value is None:
        raise ValueError(
            f"labels.{threshold_name(params.fog_index)} is not set, so the baseline has no threshold"
        )

    name, options = winner
    winner_params = dataclasses.replace(params, **options)
    winner_values = value_grid(winner_params, stations_cfg)
    selection_scores = value_scores(
        variant_grid(data, selection, winner_params, winner_values),
        selection,
        pd.Series(True, index=selection.index),
    )
    value, matched = matched_value(selection_scores, stations_cfg["matched_far"])
    if not matched:
        log.warning(
            "Winner %s: no value reaches a FAR of %.3f on the selection cases; using %g",
            name,
            stations_cfg["matched_far"],
            value,
        )

    rules = {"baseline": (params, baseline_value), "winner": (winner_params, value)}
    grids = {
        rule: variant_grid(data, held_out, rule_params, np.array([rule_value]))
        for rule, (rule_params, rule_value) in rules.items()
    }
    where = held_out_part(held_out, stations_cfg["n_folds"])
    masks = {"held_out": pd.Series(True, index=held_out.index)}
    masks |= {part: where == part for part in ("nights", "stations", "both")}
    rows = [
        {"rule": rule, "part": part, **value_scores(grids[rule], held_out, mask).to_dict("records")[0]}
        for rule in rules
        for part, mask in masks.items()
    ]
    table = pd.DataFrame(rows)

    counts = {
        rule: block_counts(
            grids[rule],
            held_out,
            rule_value,
            masks["held_out"],
            season_start,
            stations_cfg["night_block_days"],
        )
        for rule, (_, rule_value) in rules.items()
    }
    samples, seed = stations_cfg["bootstrap_samples"], stations_cfg["seed"]
    pod_low, pod_high = paired_difference(counts["winner"], counts["baseline"], "pod", samples, seed)
    far_low, far_high = paired_difference(counts["winner"], counts["baseline"], "far", samples, seed)
    whole = table[table["part"] == "held_out"].set_index("rule")
    decision = KeepDecision(
        winner=name,
        value=value,
        pod_gain=float(whole.loc["winner", "pod"] - whole.loc["baseline", "pod"]),
        pod_gain_low=pod_low,
        pod_gain_high=pod_high,
        far_winner=float(whole.loc["winner", "far"]),
        far_baseline=float(whole.loc["baseline", "far"]),
        far_difference_low=far_low,
        far_difference_high=far_high,
    )
    return table, decision
