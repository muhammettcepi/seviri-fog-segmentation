"""Experiment tracking: the tracker that records nothing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class NoTracker:
    """A tracker with the methods of the MLflow one that does nothing."""

    @contextmanager
    def run(self, name: str, params: dict, nested: bool = False) -> Iterator[str]:
        yield ""

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        pass

    def artifact(self, path: Path) -> None:
        pass
