"""Cross-validation of the U-Net settings over the night folds."""

from __future__ import annotations

import dataclasses
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from fogseg.experiment import CV_COLUMNS
from fogseg.teacher import setting_rows
from fogseg.tracking import MlflowTracker, NoTracker
from fogseg.tuning import grid_values
from fogseg.unet_train import UNetParams, predict_probabilities, train_model

log = logging.getLogger(__name__)

Tracker = NoTracker | MlflowTracker


def case_probabilities(probability: np.ndarray, times: np.ndarray, cases: pd.DataFrame) -> np.ndarray:
    """The probability at each case's scene and pixel; `times` are the scene times of `probability`."""
    scene = pd.Index(times).get_indexer(pd.to_datetime(cases["scene_time"]))
    if (scene < 0).any():
        missing = cases["scene_time"][scene < 0].iloc[0]
        raise ValueError(f"The case scene {missing} is not among the scenes of the probabilities")
    return probability[scene, cases["iy"].to_numpy(), cases["ix"].to_numpy()]


@dataclasses.dataclass(frozen=True)
class FitData:
    """The scenes of the tuning season: inputs, targets by name, usable pixels, night folds, station points.

    `usable` marks the trusted pixels that are not obscured, the only ones the product labels fog or no fog.
    """

    times: np.ndarray
    inputs: np.ndarray
    targets: dict[str, tuple[np.ndarray, np.ndarray]]
    usable: np.ndarray
    night_folds: np.ndarray
    points: pd.DataFrame

    def subset(self, scenes: np.ndarray) -> FitData:
        """The scenes at the positions `scenes`, with the points of other scenes dropped."""
        position = pd.Series(np.arange(len(scenes)), index=scenes)
        kept = self.points[self.points["scene"].isin(scenes)]
        points = kept.assign(scene=position.loc[kept["scene"]].to_numpy()).reset_index(drop=True)
        return FitData(
            self.times[scenes],
            self.inputs[scenes],
            {name: (target[scenes], weight[scenes]) for name, (target, weight) in self.targets.items()},
            self.usable[scenes],
            self.night_folds[scenes],
            points,
        )


def cross_validate_fold(
    fit: FitData,
    cases: pd.DataFrame,
    setting: tuple[str, dict],
    fold: int,
    params: UNetParams,
    stations_cfg: dict,
    device: torch.device,
    tracker: Tracker,
) -> pd.DataFrame:
    """The U-Net fitted on the other selection folds, as probabilities of the cases of `fold`."""
    name, options = setting
    selection = np.arange(1, stations_cfg["n_folds"])
    scenes = np.arange(len(fit.times))
    train = fit.subset(scenes[np.isin(fit.night_folds, selection[selection != fold])])
    test = fit.subset(scenes[fit.night_folds == fold])
    target, weight = train.targets[options["target"]]
    test_target, test_weight = test.targets[options["target"]]
    in_fold = cases[cases["night_fold"] == fold]
    with tracker.run(f"{name} fold {fold}", {"setting": name, "fold": fold, **options}, nested=True):
        model = train_model(
            train.inputs,
            target,
            weight,
            train.points,
            options,
            params,
            stations_cfg["seed"] + fold,
            device,
            monitor=(test.inputs, test_target, test_weight),
            on_epoch=lambda epoch, train_loss, test_loss: tracker.metrics(
                {"train_loss": train_loss, "test_loss": test_loss}, step=epoch
            ),
        )
        probability = predict_probabilities(model, test.inputs, device, params.batch_size)
    probability[~test.usable] = np.nan
    p = case_probabilities(probability, test.times, in_fold)
    return pd.DataFrame({"fold": fold, "case": in_fold["case"].to_numpy(), "p": p})


def cross_validate_unet(
    fit: FitData,
    cases: pd.DataFrame,
    settings: list[tuple[str, dict]],
    params: UNetParams,
    stations_cfg: dict,
    device: torch.device,
    tracker: Tracker,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The cross-validation table of every setting, and each setting's out-of-fold probabilities.

    Each night fold of the selection season in turn is the test part; the U-Net is fitted on the scenes of
    the other selection folds only, never on those of the held-out fold. The probabilities are those at the
    selection `cases` of the test fold, which are scored in the teacher's full-count way at the pooled p*.
    They are NaN where the pixel is not usable, which the full count scores as no fog, as it scores the
    product's obscured and ignored labels.
    """
    values = grid_values(params.probability_grid)
    folds = []
    for fold in range(1, stations_cfg["n_folds"]):
        if (fit.night_folds == fold).any():
            folds.append(fold)
        else:
            log.warning("Night fold %d has no scenes and is skipped", fold)
    tables, probabilities = [], {}
    for setting in settings:
        name, options = setting
        started = time.perf_counter()
        parts = [
            cross_validate_fold(fit, cases, setting, fold, params, stations_cfg, device, tracker)
            for fold in folds
        ]
        probabilities[name] = pd.concat(parts, ignore_index=True)
        table = setting_rows(probabilities[name], cases, values, 0.0, stations_cfg["matched_far"], name)
        pooled = table.iloc[-1]
        number = name.split()[0]
        tracker.metrics(
            {
                f"cv_pod_{number}": pooled.pod,
                f"cv_far_{number}": pooled.far,
                f"cv_csi_{number}": pooled.csi,
                f"cv_p_star_{number}": pooled.value,
            }
        )
        log.info(
            "Setting %s: p* %g, pooled POD %.3f, FAR %.3f, %.1f s",
            name,
            pooled.value,
            pooled.pod,
            pooled.far,
            time.perf_counter() - started,
        )
        tables.append(table.assign(variant=name, options=json.dumps(options)))
    return pd.concat(tables, ignore_index=True)[["variant", "options", *CV_COLUMNS]], probabilities
