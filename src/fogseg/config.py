"""Load the YAML config and derive the list of scene times from it."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scene_times(cfg: dict) -> list[dt.datetime]:
    """All nominal slot times (UTC) for the configured seasons and night slots.

    A "night" starts on a date d: evening slots (>= 12:00) are on d, morning slots on d + 1.
    """
    times = []
    slots = [dt.time.fromisoformat(s) for s in cfg["time"]["slots"]]
    step = dt.timedelta(days=cfg["time"].get("every_nth_night", 1))
    for season in cfg["time"]["seasons"]:
        night = dt.date.fromisoformat(season["start"])
        end = dt.date.fromisoformat(season["end"])
        while night <= end:
            for slot in slots:
                day = night if slot.hour >= 12 else night + dt.timedelta(days=1)
                times.append(dt.datetime.combine(day, slot))
            night += step
    return sorted(times)
