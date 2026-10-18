"""Shared test fixtures."""

from __future__ import annotations

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
