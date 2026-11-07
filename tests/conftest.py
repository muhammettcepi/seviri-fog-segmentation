"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from helpers import TIMES, make_scene

from fogseg.config import scene_filename
from fogseg.preprocess import check_grid, save_scene


@pytest.fixture
def interim_dir(tmp_path: Path) -> Path:
    """A folder with grid.nc and three fake scenes (written out of time order on purpose)."""
    folder = tmp_path / "interim"
    for i, time in enumerate(TIMES):
        ds = make_scene(time, value=270.0 + i)
        check_grid(ds, folder)
        save_scene(ds, folder / scene_filename(time))
    return folder


@pytest.fixture(scope="session")
def migrated_mlflow_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty MLflow SQLite store with its migrations applied, which take over a minute."""
    mlflow = pytest.importorskip("mlflow")
    path = tmp_path_factory.mktemp("mlflow") / "template.db"
    mlflow.MlflowClient(f"sqlite:///{path}").search_experiments()
    return path


@pytest.fixture
def mlflow_uri(tmp_path: Path, migrated_mlflow_db: Path) -> str:
    """The URI of a private copy of the migrated MLflow store."""
    database = tmp_path / "mlflow.db"
    shutil.copy(migrated_mlflow_db, database)
    return f"sqlite:///{database}"
