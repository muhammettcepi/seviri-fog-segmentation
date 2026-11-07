"""Experiment tracking: the tracker that records nothing and the MLflow one."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def flatten(values: dict, prefix: str = "") -> dict:
    """Nested dicts as one level of dotted keys, e.g. `unet.epochs`."""
    flat = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


class NoTracker:
    """A tracker with the methods of the MLflow one that does nothing."""

    @contextmanager
    def run(self, name: str, params: dict, nested: bool = False) -> Iterator[str]:
        yield ""

    @contextmanager
    def resume(self, run_id: str) -> Iterator[str]:
        yield run_id

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        pass

    def artifact(self, path: Path) -> None:
        pass


class MlflowTracker:
    """Runs, metrics and artifacts in an MLflow store, in one experiment."""

    def __init__(self, tracking_uri: str, artifact_dir: Path, experiment: str):
        import mlflow

        for name in ("mlflow", "alembic"):
            logging.getLogger(name).setLevel(logging.WARNING)
        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_dir.resolve().as_uri())
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, params: dict, nested: bool = False) -> Iterator[str]:
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

        with self.mlflow.start_run(run_name=name, nested=nested) as active:
            self.mlflow.log_params(
                {key: str(value)[:MAX_PARAM_VAL_LENGTH] for key, value in flatten(params).items()}
            )
            yield active.info.run_id

    @contextmanager
    def resume(self, run_id: str) -> Iterator[str]:
        with self.mlflow.start_run(run_id=run_id) as active:
            yield active.info.run_id

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(values, step=step)

    def artifact(self, path: Path) -> None:
        self.mlflow.log_artifact(str(path))
