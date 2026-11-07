import dataclasses

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fogseg.labels import FOG, IGNORE, NO_FOG, OBSCURED
from fogseg.unet_data import INPUT_CHANNELS, Normalization
from fogseg.unet_train import (
    UNetModel,
    UNetParams,
    batch_points,
    load_unet,
    masked_bce,
    network,
    point_bce,
    predict_probabilities,
    save_unet,
    train_model,
    unet_labels,
    unet_settings,
)

N_CHANNELS = len(INPUT_CHANNELS)
CPU = torch.device("cpu")
SOFT = {"target": "soft", "point_weight": 0.0}
NO_POINTS = pd.DataFrame(
    {
        "scene": np.zeros(0, dtype="int64"),
        "iy": np.zeros(0, dtype="int64"),
        "ix": np.zeros(0, dtype="int64"),
        "truth": np.zeros(0, dtype="float32"),
    }
)


def tiny_params(**changes) -> UNetParams:
    values = {
        "model_path": "unet.pt",
        "predictions_path": "predictions.zarr",
        "tracking_uri": "sqlite:///mlflow.db",
        "artifact_dir": "artifacts",
        "experiment": "test",
        "base_channels": 4,
        "depth": 2,
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "point_weights": [0.25, 1.0],
        "probability_grid": [0.1, 0.9, 0.1],
        "device": "cpu",
    }
    return UNetParams(**{**values, **changes})


def scenes(n=4, height=8, width=10, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, N_CHANNELS, height, width)).astype("float32")
    inputs[:, -1] = 1.0
    target = (inputs[:, 0] > 0).astype("float32")
    return inputs, target, np.ones_like(target)


def train(inputs, target, weight, params=None, points=NO_POINTS, options=SOFT, seed=0, **kwargs):
    return train_model(inputs, target, weight, points, options, params or tiny_params(), seed, CPU, **kwargs)


def test_settings_are_numbered_from_the_point_weights():
    settings = unet_settings([0.25, 1.0])
    assert len(settings) == 4
    assert settings == [
        ("1 soft", {"target": "soft", "point_weight": 0.0}),
        ("2 hard", {"target": "hard", "point_weight": 0.0}),
        ("3 soft points 0.25", {"target": "soft", "point_weight": 0.25}),
        ("4 soft points 1", {"target": "soft", "point_weight": 1.0}),
    ]


def test_params_come_from_the_config_section():
    cfg = dataclasses.asdict(tiny_params())
    assert UNetParams.from_config({**cfg, "unused": 1}) == tiny_params()


def test_masked_bce_skips_weightless_pixels():
    logits = torch.randn(2, 4, 5)
    target = torch.rand(2, 4, 5)
    weight = (torch.rand(2, 4, 5) > 0.5).float()
    target[weight == 0] = 3.0
    other = logits.clone()
    other[weight == 0] += 50.0
    assert torch.isfinite(masked_bce(logits, target, weight))
    assert torch.isclose(masked_bce(logits, target, weight), masked_bce(other, target, weight))


def test_masked_bce_without_weight_is_zero_and_differentiable():
    logits = torch.randn(2, 4, 5, requires_grad=True)
    loss = masked_bce(logits, torch.full((2, 4, 5), 4.0), torch.zeros(2, 4, 5))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_point_bce_reads_its_own_pixel():
    logits = torch.zeros(2, 5, 6)
    logits[1, 2, 3] = 20.0
    batch, iy, ix = torch.tensor([1]), torch.tensor([2]), torch.tensor([3])
    assert point_bce(logits, batch, iy, ix, torch.tensor([1.0])) < 0.01
    assert point_bce(logits, batch, iy, ix, torch.tensor([0.0])) > 5


def test_batch_points_map_each_scene_to_its_position_in_the_shuffled_batch():
    points = pd.DataFrame(
        {
            "scene": [0, 2, 5, 5, 3],
            "iy": [1, 2, 3, 4, 5],
            "ix": [6, 7, 8, 9, 0],
            "truth": np.array([1, 0, 1, 0, 1], dtype="float32"),
        }
    )
    batch, iy, ix, truth = batch_points(points, np.array([5, 0, 3]), CPU)
    assert batch.tolist() == [1, 0, 0, 2]
    assert iy.tolist() == [1, 3, 4, 5]
    assert ix.tolist() == [6, 8, 9, 0]
    assert truth.tolist() == [1.0, 1.0, 0.0, 1.0]
    assert batch.dtype == torch.long and truth.dtype == torch.float32


def test_point_bce_without_points_is_zero():
    empty = torch.zeros(0, dtype=torch.long)
    assert point_bce(torch.zeros(2, 5, 6), empty, empty, empty, torch.zeros(0)).item() == 0.0


def test_training_learns_a_simple_map():
    inputs, target, weight = scenes(8)
    losses = []
    train(inputs, target, weight, tiny_params(epochs=30), on_epoch=lambda epoch, loss, _: losses.append(loss))
    assert losses[-1] < losses[0] / 2


def test_monitor_loss_is_reported_each_epoch():
    inputs, target, weight = scenes()
    reports = []
    train(
        inputs,
        target,
        weight,
        monitor=(inputs, target, weight),
        on_epoch=lambda epoch, loss, monitor: reports.append((epoch, monitor)),
    )
    assert [epoch for epoch, _ in reports] == [1, 2]
    assert all(isinstance(monitor, float) for _, monitor in reports)


def test_training_without_a_monitor_reports_none():
    inputs, target, weight = scenes()
    reports = []
    train(inputs, target, weight, on_epoch=lambda epoch, loss, monitor: reports.append(monitor))
    assert reports == [None, None]


def test_point_loss_changes_the_weights():
    inputs, target, weight = scenes()
    points = pd.DataFrame(
        {"scene": [0, 3], "iy": [1, 2], "ix": [3, 4], "truth": np.array([1.0, 0.0], dtype="float32")}
    )
    plain = train(inputs, target, weight, points=points)
    pointed = train(inputs, target, weight, points=points, options={"target": "soft", "point_weight": 1.0})
    assert not torch.allclose(
        next(iter(plain.state_dict().values())), next(iter(pointed.state_dict().values()))
    )


def test_same_seed_same_weights():
    inputs, target, weight = scenes()
    first = train(inputs, target, weight, seed=3)
    second = train(inputs, target, weight, seed=3)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_probabilities_are_nan_where_untrusted():
    inputs, target, weight = scenes()
    model = train(inputs, target, weight)
    inputs[1, -1, 2:4, 3:5] = 0.0
    p = predict_probabilities(model, inputs, CPU, 3)
    assert p.shape == (4, 8, 10) and p.dtype == np.float32
    assert np.isnan(p[1, 2:4, 3:5]).all()
    assert np.isfinite(p).sum() == p.size - 4
    assert ((p[np.isfinite(p)] >= 0) & (p[np.isfinite(p)] <= 1)).all()


def test_labels_follow_p_star_nan_and_obscured():
    p = np.array([0.2, 0.41, 0.9, np.nan, 0.9])
    obscured = np.array([False, False, False, False, True])
    labels = unet_labels(p, obscured, 0.41)
    assert labels.dtype == np.uint8
    assert labels.tolist() == [NO_FOG, FOG, FOG, IGNORE, OBSCURED]


def saved_model(tmp_path, channels=None):
    inputs, target, weight = scenes()
    net = train(inputs, target, weight)
    model = UNetModel(
        setting="1 soft",
        options=SOFT,
        state_dict=net.state_dict(),
        normalization=Normalization(np.arange(3, dtype="float64"), np.ones(3)),
        p_star=0.4,
        input_channels=channels or list(INPUT_CHANNELS),
        base_channels=4,
        depth=2,
        label_attrs={"mode": "teacher"},
        torch_version=torch.__version__,
        fogseg_version="0.1.0",
    )
    return inputs, net, model, tmp_path / "unet.pt"


def test_saved_model_predicts_like_the_trained_one(tmp_path):
    inputs, net, model, path = saved_model(tmp_path)
    save_unet(model, path)
    loaded = load_unet(path)
    assert loaded.setting == "1 soft" and loaded.p_star == 0.4 and loaded.label_attrs == {"mode": "teacher"}
    assert np.array_equal(loaded.normalization.mean, model.normalization.mean)
    expected = predict_probabilities(net, inputs, CPU, 2)
    assert np.allclose(predict_probabilities(network(loaded), inputs, CPU, 2), expected, atol=1e-6)


def test_loading_refuses_other_input_channels(tmp_path):
    _, _, model, path = saved_model(tmp_path, channels=INPUT_CHANNELS[:-1])
    save_unet(model, path)
    with pytest.raises(ValueError, match="input channels"):
        load_unet(path)


def test_loading_a_missing_model_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg train"):
        load_unet(tmp_path / "missing.pt")


def test_saving_leaves_no_partial_file(tmp_path):
    _, _, model, path = saved_model(tmp_path)
    save_unet(model, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unet.pt"]
