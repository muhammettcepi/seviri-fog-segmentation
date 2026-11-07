import pytest

mlflow = pytest.importorskip("mlflow")

from fogseg.tracking import MlflowTracker, flatten


def test_flatten_joins_nested_keys():
    assert flatten({"unet": {"epochs": 2, "grid": [0.1, 0.9]}, "mode": "teacher"}) == {
        "unet.epochs": 2,
        "unet.grid": [0.1, 0.9],
        "mode": "teacher",
    }


def test_tracker_writes_runs_metrics_and_artifacts(tmp_path, mlflow_uri):
    uri = mlflow_uri
    tracker = MlflowTracker(uri, tmp_path / "a", "t")
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    with (
        tracker.run("parent", {"unet": {"epochs": 2}, "long": "x" * 10000}) as parent,
        tracker.run("child", {"fold": 1}, nested=True) as child,
    ):
        tracker.metrics({"loss": 1.0}, step=0)
        tracker.metrics({"loss": 0.5}, step=1)
        tracker.artifact(note)
    with tracker.resume(child):
        tracker.metrics({"late": 3.0})

    client = mlflow.MlflowClient(uri)
    runs = client.search_runs([client.get_experiment_by_name("t").experiment_id])
    assert len(runs) == 2
    nested = client.get_run(child)
    assert nested.data.tags["mlflow.parentRunId"] == parent
    assert nested.data.metrics["late"] == 3.0
    assert [metric.value for metric in client.get_metric_history(child, "loss")] == [1.0, 0.5]
    assert [item.path for item in client.list_artifacts(child)] == ["note.txt"]
    params = client.get_run(parent).data.params
    assert params["unet.epochs"] == "2"
    assert 0 < len(params["long"]) < 10000
    assert (tmp_path / "a").exists()
