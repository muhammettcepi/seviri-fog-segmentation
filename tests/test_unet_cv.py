import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from test_unet_train import tiny_params

from fogseg.experiment import CV_COLUMNS, winner
from fogseg.tracking import NoTracker
from fogseg.unet_cv import FitData, case_probabilities, cross_validate_unet
from fogseg.unet_data import INPUT_CHANNELS, TARGETS, point_targets
from fogseg.unet_train import unet_settings

CPU = torch.device("cpu")
STATIONS_CFG = {"matched_far": 0.3, "n_folds": 5, "seed": 0}
SCENE_FOLDS = [1, 1, 2, 2, 3, 3, 5, 5]
ID_CHANNEL = 1
HEIGHT, WIDTH = 8, 10


def synthetic(scene_folds=SCENE_FOLDS):
    """Scenes whose fog is where channel 0 is positive, and 8 cases in each, `fit` and the selection cases."""
    rng = np.random.default_rng(0)
    n = len(scene_folds)
    times = np.datetime64("2023-01-01T00:00") + np.arange(n) * np.timedelta64(15, "m")
    inputs = rng.normal(size=(n, len(INPUT_CHANNELS), HEIGHT, WIDTH)).astype("float32")
    inputs[:, ID_CHANNEL] = np.arange(n)[:, None, None]
    inputs[:, -1] = 1.0
    target = (inputs[:, 0] > 0).astype("float32")
    rows = []
    for scene in range(n):
        for _ in range(8):
            iy, ix = rng.integers(HEIGHT), rng.integers(WIDTH)
            rows.append(
                {
                    "scene_time": times[scene],
                    "iy": iy,
                    "ix": ix,
                    "night_fold": scene_folds[scene],
                    "station_fold": rng.integers(1, 5),
                    "truth_fls": int(target[scene, iy, ix]),
                }
            )
    cases = pd.DataFrame(rows)
    targets = {name: (target, np.ones_like(target)) for name in TARGETS}
    usable = np.ones(target.shape, dtype=bool)
    points = point_targets(cases, times, usable)
    fit = FitData(times, inputs, targets, usable, np.array(scene_folds), points)
    selection = cases[cases["night_fold"] != 5].reset_index(drop=True)
    return fit, selection.assign(case=selection.index)


def run(fit, cases, params=None, settings=None, tracker=None):
    settings = settings or unet_settings([1.0])[:1]
    params = params or tiny_params(epochs=1)
    return cross_validate_unet(fit, cases, settings, params, STATIONS_CFG, CPU, tracker or NoTracker())


def test_case_probabilities_read_their_pixel():
    times = np.datetime64("2023-01-01T00:00") + np.arange(3) * np.timedelta64(15, "m")
    probability = np.arange(3 * 4 * 5, dtype="float32").reshape(3, 4, 5)
    cases = pd.DataFrame({"scene_time": [times[2], times[0]], "iy": [3, 1], "ix": [4, 2]})
    expected = [probability[2, 3, 4], probability[0, 1, 2]]
    assert case_probabilities(probability, times, cases).tolist() == expected


def test_a_case_scene_outside_the_fit_is_an_error():
    times = np.datetime64("2023-01-01T00:00") + np.arange(2) * np.timedelta64(15, "m")
    cases = pd.DataFrame({"scene_time": [np.datetime64("2023-02-01T00:00")], "iy": [0], "ix": [0]})
    with pytest.raises(ValueError, match="scene"):
        case_probabilities(np.zeros((2, 4, 5), "float32"), times, cases)


def test_subset_reindexes_points():
    fit, _ = synthetic()
    chosen = np.array([4, 5, 1])
    part = fit.subset(chosen)
    assert (part.times == fit.times[chosen]).all()
    assert part.inputs.shape[0] == 3
    assert part.targets["soft"][0].shape[0] == 3
    assert part.usable.shape == (3, HEIGHT, WIDTH)
    assert (part.night_folds == fit.night_folds[chosen]).all()
    assert part.points["scene"].isin([0, 1, 2]).all()
    expected = fit.points[fit.points["scene"].isin(chosen)]
    assert len(part.points) == len(expected)
    kept = part.points.assign(time=part.times[part.points["scene"].to_numpy()])
    original = expected.assign(time=fit.times[expected["scene"].to_numpy()])
    columns = ["time", "iy", "ix", "truth"]
    assert (
        kept[columns]
        .sort_values(columns)
        .reset_index(drop=True)
        .equals(original[columns].sort_values(columns).reset_index(drop=True))
    )


def test_no_test_scene_and_no_held_out_scene_is_ever_fitted(monkeypatch):
    fit, cases = synthetic()
    seen = []

    def scene_ids(inputs):
        return {int(scene) for scene in inputs[:, ID_CHANNEL, 0, 0]}

    def record(inputs, target, weight, points, options, params, seed, device, monitor=None, on_epoch=None):
        seen.append((seed, scene_ids(inputs), scene_ids(monitor[0])))
        return torch.nn.Conv2d(len(INPUT_CHANNELS), 1, 1).eval()

    monkeypatch.setattr("fogseg.unet_cv.train_model", record)
    monkeypatch.setattr(
        "fogseg.unet_cv.predict_probabilities",
        lambda model, inputs, device, batch_size: np.full((len(inputs), HEIGHT, WIDTH), 0.5, "float32"),
    )
    run(fit, cases)
    scene_folds = np.array(SCENE_FOLDS)
    assert [seed for seed, *_ in seen] == [1, 2, 3]
    for fold, fitted, monitored in seen:
        assert {int(scene_folds[scene]) for scene in fitted} == {1, 2, 3} - {fold}
        assert {int(scene_folds[scene]) for scene in monitored} == {fold}


def test_cases_at_obscured_or_untrusted_pixels_are_scored_as_no(monkeypatch):
    fit, cases = synthetic()
    in_first_scene = cases["scene_time"] == fit.times[0]
    unusable, others = cases[in_first_scene], cases[~in_first_scene & (cases["night_fold"] == 1)]
    fit.usable[0, unusable["iy"], unusable["ix"]] = False
    monkeypatch.setattr(
        "fogseg.unet_cv.train_model", lambda *args, **kwargs: torch.nn.Conv2d(len(INPUT_CHANNELS), 1, 1)
    )
    monkeypatch.setattr(
        "fogseg.unet_cv.predict_probabilities",
        lambda model, inputs, device, batch_size: np.full((len(inputs), HEIGHT, WIDTH), 0.95, "float32"),
    )
    table, probabilities = run(fit, cases)
    p = probabilities["1 soft"].set_index("case")["p"]
    assert p[unusable["case"]].isna().all()
    assert (p[others["case"]] > 0.9).all()
    fold_1 = table[table["fold"].astype(str) == "1"].iloc[0]
    assert fold_1["hits"] == (others["truth_fls"] == 1).sum()
    assert fold_1["false_alarms"] == (others["truth_fls"] == 0).sum()
    assert fold_1["misses"] == (unusable["truth_fls"] == 1).sum()
    assert fold_1["correct_negatives"] == (unusable["truth_fls"] == 0).sum()


def test_cases_are_scored_only_in_their_night_fold():
    fit, cases = synthetic()
    _, probabilities = run(fit, cases)
    probs = probabilities["1 soft"]
    assert list(probs.columns) == ["fold", "case", "p"]
    assert sorted(probs["case"]) == list(cases["case"])
    assert (cases["night_fold"].to_numpy()[probs["case"].to_numpy()] == probs["fold"].to_numpy()).all()


def test_the_table_has_a_row_per_usable_fold_and_a_pooled_row_per_setting():
    fit, cases = synthetic()
    table, probabilities = run(fit, cases, settings=unet_settings([1.0])[:2])
    assert list(table.columns) == ["variant", "options", *CV_COLUMNS]
    assert list(probabilities) == ["1 soft", "2 hard"]
    for name in probabilities:
        rows = table[table["variant"] == name]
        assert rows["fold"].astype(str).tolist() == ["1", "2", "3", "pooled"]
    assert winner(table) in ("1 soft", "2 hard")


def test_a_fold_without_scenes_is_skipped(caplog):
    fit, cases = synthetic()
    with caplog.at_level(logging.WARNING):
        _, probabilities = run(fit, cases)
    assert 4 not in set(probabilities["1 soft"]["fold"])
    assert "fold 4" in caplog.text


def test_each_fold_is_a_nested_run_with_its_metrics():
    fit, cases = synthetic()
    events = []

    class Recorder:
        @contextmanager
        def run(self, name, params, nested=False):
            events.append(("run", name, params, nested))
            yield ""

        def metrics(self, values, step=None):
            events.append(("metrics", sorted(values), step))

        def artifact(self, path):
            pass

    run(fit, cases, params=tiny_params(epochs=2), tracker=Recorder())
    runs = [event for event in events if event[0] == "run"]
    assert [event[1] for event in runs] == ["1 soft fold 1", "1 soft fold 2", "1 soft fold 3"]
    assert all(event[3] for event in runs)
    assert runs[0][2]["setting"] == "1 soft"
    assert runs[0][2]["fold"] == 1
    epochs = [event for event in events if event[1] == ["test_loss", "train_loss"]]
    assert [event[2] for event in epochs] == [1, 2] * 3
    assert events[-1] == ("metrics", ["cv_csi_1", "cv_far_1", "cv_p_star_1", "cv_pod_1"], None)
