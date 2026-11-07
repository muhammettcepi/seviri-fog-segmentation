"""Dense teacher labels: the teacher's judgement of every trusted pixel of every stored scene.

The store has the layout of the rule's label stores, plus `fog_probability`: the teacher's calibrated
probability of fog as an integer from 0 to 250, 255 where it is undefined. A pixel the rule cannot see
(untrusted) or a scene without ERA5 gets no probability and is ignored; obscured pixels stay obscured.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fogseg.build import CHUNK_TIME
from fogseg.features import NightFeatures, StaticFields, night_features, open_era5, read_static
from fogseg.label_store import (
    TIME_ENCODING,
    check_dataset,
    label_fractions,
    night_runs,
    store_attrs,
    write_metadata,
    write_store,
)
from fogseg.labels import IGNORE, OBSCURED, LabelParams
from fogseg.teacher import (
    TeacherModel,
    TeacherParams,
    corrected_probability,
    correction_feeders,
    load_teacher,
    predict_probability,
    teacher_labels,
)
from fogseg.validation import season_bounds

log = logging.getLogger(__name__)

UNDEFINED = 255
PROBABILITY_SCALE = 250


@dataclasses.dataclass(frozen=True)
class Correction:
    """The station correction of the tuning season: the feeders of each scene and the season's bounds."""

    length_km: float
    feeders: dict[pd.Timestamp, pd.DataFrame]
    start: np.datetime64
    end: np.datetime64

    def applies(self, time: np.datetime64) -> bool:
        return bool(self.start <= time < self.end) and pd.Timestamp(time) in self.feeders


def read_correction(cfg: dict, teacher: TeacherModel) -> Correction | None:
    """The correction if the teacher's held-out record keeps it, else None."""
    stations_cfg = cfg["stations"]
    data_dir = Path(stations_cfg["data_dir"])
    record_path = data_dir / "teacher_held_out.json"
    if not record_path.exists():
        log.info("%s not found; the labels get no station correction", record_path)
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not (record.get("correction") or {}).get("keep"):
        log.info("The held-out record does not keep the station correction; the labels get none")
        return None
    if teacher.correction_km is None:
        raise ValueError(f"{record_path} keeps the correction, but the teacher has no correction length")

    oof = pd.read_csv(data_dir / "teacher_oof.csv", dtype={"station": str}, parse_dates=["scene_time"])
    feeders = dict(tuple(correction_feeders(oof).groupby("scene_time")))
    start, end = season_bounds(cfg, stations_cfg["tune_season"])
    return Correction(teacher.correction_km, feeders, start, end)


def label_scene(
    teacher: TeacherModel,
    night: NightFeatures,
    scene: int,
    lonlat: tuple[np.ndarray, np.ndarray],
    time: np.datetime64,
    correction: Correction | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The labels and the probabilities (0 to 250, 255 undefined) of one scene of a night."""
    shape = night.trusted.shape[1:]
    labels = np.full(shape, IGNORE, dtype=np.uint8)
    probability = np.full(shape, UNDEFINED, dtype=np.uint8)

    usable = night.trusted[scene]
    if night.has_era5[scene] and usable.any():
        p = predict_probability(teacher, night.stack[scene][:, usable].T)
        if correction is not None and correction.applies(time):
            lon, lat = lonlat
            targets = pd.DataFrame(
                {"scene_time": time, "lon": lon[usable], "lat": lat[usable], "station": ""}
            )
            p = corrected_probability(
                targets, p, correction.feeders[pd.Timestamp(time)], correction.length_km
            )
        labels[usable] = teacher_labels(p, teacher.p_star, teacher.ignore_halfwidth)
        probability[usable] = np.rint(p * PROBABILITY_SCALE)

    labels[night.rule_labels[scene] == OBSCURED] = OBSCURED
    return labels, probability


def teacher_batch(
    data: xr.Dataset,
    start: int,
    stop: int,
    teacher: TeacherModel,
    params: LabelParams,
    static: StaticFields,
    era5_store: xr.Dataset | None,
    correction: Correction | None,
) -> xr.Dataset:
    """Label scenes `start` to `stop` of the dataset, which must hold whole nights."""
    times = data["time"].values[start:stop]
    lonlat = data["lon"].values, data["lat"].values
    labels = np.empty((len(times), data.sizes["y"], data.sizes["x"]), dtype=np.uint8)
    probability = np.empty_like(labels)

    for first, last in night_runs(times):
        scenes = np.arange(start + first, start + last)
        night = night_features(data, scenes, static, era5_store, params, teacher.feature_groups)
        for slot in range(last - first):
            labels[first + slot], probability[first + slot] = label_scene(
                teacher, night, slot, lonlat, times[first + slot], correction
            )

    fractions = np.array([label_fractions(scene) for scene in labels], dtype=np.float32)
    # object dtype is stored as a variable-length string; Zarr v3 has no spec for fixed-width text
    sources = np.full(len(times), "teacher", dtype=object)
    return xr.Dataset(
        {
            "label": (
                ("time", "y", "x"),
                labels,
                {"long_name": "fog label: 0 no fog, 1 fog, 2 obscured, 255 ignore"},
            ),
            "fog_probability": (
                ("time", "y", "x"),
                probability,
                {"long_name": f"teacher probability of fog times {PROBABILITY_SCALE}, 255 undefined"},
            ),
            "threshold_source": (("time",), sources, {"long_name": "always teacher"}),
            "fog_fraction": (("time",), fractions[:, 0], {"long_name": "fog share of the decided pixels"}),
            "ignore_fraction": (("time",), fractions[:, 1], {"long_name": "ignored share of all pixels"}),
            "obscured_fraction": (("time",), fractions[:, 2], {"long_name": "obscured share of all pixels"}),
        },
        coords={"time": times, "y": data["y"].values, "x": data["x"].values},
    )


def teacher_attrs(
    teacher: TeacherModel, model_path: Path, correction: Correction | None, rule_attrs: dict
) -> dict:
    """The attributes of the rule's stores, plus the teacher that made the labels."""
    return {
        **rule_attrs,
        "mode": "teacher",
        "teacher_setting": teacher.setting,
        "teacher_p_star": teacher.p_star,
        "teacher_ignore_halfwidth": teacher.ignore_halfwidth,
        "teacher_feature_groups": list(teacher.feature_groups),
        "teacher_correction_km": None if correction is None else correction.length_km,
        "teacher_model_path": str(model_path),
    }


def label_teacher_dataset(
    dataset_path: Path, out_path: Path, cfg: dict, overwrite: bool = False, batch: int = 32
) -> int:
    """Label every scene of the dataset with the teacher into the Zarr store `out_path`.

    Returns the number of scenes. Written like the rule's stores: under a temporary name, renamed at the
    end, and the temporary store is deleted if anything fails. The teacher is loaded and checked first.
    """
    params = LabelParams.from_config(cfg["labels"])
    teacher_params = TeacherParams.from_config(cfg["teacher"])
    model_path = Path(teacher_params.model_path)
    teacher = load_teacher(model_path, teacher_params.feature_groups)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; use overwrite=True (--overwrite) to replace it")

    groups = teacher_params.feature_groups
    with ExitStack() as stack:
        data = stack.enter_context(xr.open_zarr(dataset_path, consolidated=False))
        check_dataset(data, dataset_path)
        correction = read_correction(cfg, teacher)
        static = read_static(data, Path(cfg["dem"]["out_path"]), groups)
        era5_store = open_era5(Path(cfg["era5"]["out_path"]), groups)
        if era5_store is not None:
            stack.enter_context(era5_store)

        def make_batch(start: int, stop: int) -> xr.Dataset:
            return teacher_batch(data, start, stop, teacher, params, static, era5_store, correction)

        shape = (CHUNK_TIME, data.sizes["y"], data.sizes["x"])
        encoding = {
            "label": {"chunks": shape},
            "fog_probability": {"chunks": shape},
            "threshold_source": {"chunks": (CHUNK_TIME,)},
            "fog_fraction": {"chunks": (CHUNK_TIME,)},
            "ignore_fraction": {"chunks": (CHUNK_TIME,)},
            "obscured_fraction": {"chunks": (CHUNK_TIME,)},
            "time": TIME_ENCODING,
        }
        tmp_path = out_path.with_name(out_path.name + ".part")
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_store(data, tmp_path, make_batch, encoding, batch)
            rule_attrs = store_attrs(data, dataset_path, "teacher", params, None)
            write_metadata(tmp_path, teacher_attrs(teacher, model_path, correction, rule_attrs))
        except BaseException:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise
        n_scenes = data.sizes["time"]

    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    return n_scenes
