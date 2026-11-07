"""Command line entry point: `fogseg fetch --config configs/default.yaml`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from fogseg.config import load_config, scene_filename, scene_times

if TYPE_CHECKING:
    from fogseg.experiment import KeepDecision
    from fogseg.tuning import Decision

log = logging.getLogger("fogseg")

RAW_MB = 271  # size of one full-disk .nat file, measured on a real download
MAX_TRIES = 3
BLOCKED_WAITS_MIN = [5, 10, 20, 30, 60, 60]  # pauses while EUMETSAT refuses access: ~3 h in total
_stop = threading.Event()  # set when access stays refused, so all workers stop


def fetch_one(slot: dt.datetime, collection, cfg: dict) -> str:
    """Download → crop → save → delete raw, for one slot. Returns a status string."""
    from fogseg.download import download_nat, find_product
    from fogseg.preprocess import check_grid, crop_scene, save_scene

    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["interim_dir"])
    out_path = out_dir / scene_filename(slot)

    product = find_product(collection, slot)
    if product is None:
        return "missing"  # data gaps happen (e.g. satellite manoeuvres); not an error

    nat_path = None
    try:
        nat_path = download_nat(product, raw_dir)
        ds = crop_scene(nat_path, cfg["channels"], cfg["region"]["bbox"])
        check_grid(ds, out_dir)
        save_scene(ds, out_path)
    finally:
        if nat_path is not None:
            nat_path.unlink(missing_ok=True)  # never keep raw files: the disk must not fill up
    return "ok"


def server_message(err: Exception) -> tuple[int | None, str]:
    """HTTP status and the server's own explanation, which eumdac errors carry in extra_info."""
    info = getattr(err, "extra_info", None) or {}
    texts = [e.get("exceptionText", "") for e in info.get("exceptions", [])]
    return info.get("status"), "; ".join(texts) or str(err)


def fetch_with_retry(slot: dt.datetime, collection, cfg: dict) -> str:
    """Retry short glitches a few times; wait out "access refused" (403) periods for up to ~3 h."""
    attempt = 0
    blocked = 0
    while not _stop.is_set():
        try:
            return fetch_one(slot, collection, cfg)
        except Exception as err:  # noqa: BLE001 - any failure of one scene must not stop the whole run
            status, message = server_message(err)
            if status == 403:
                # EUMETSAT sometimes refuses all downloads for a while ("GeneralLicense required"),
                # then allows them again. Retrying quickly is pointless, so pause and try again later.
                if blocked == len(BLOCKED_WAITS_MIN):
                    log.error(
                        "EUMETSAT still refuses downloads after ~3 h (%s). Stopping; run "
                        "`fogseg fetch` again later, it continues where it stopped.",
                        message,
                    )
                    _stop.set()
                    return "blocked"
                wait = BLOCKED_WAITS_MIN[blocked]
                blocked += 1
                log.warning("%s: access refused (403: %s). Pausing %d min.", slot, message, wait)
                _stop.wait(wait * 60)  # returns early if another worker has given up
                continue
            attempt += 1
            log.warning("%s attempt %d/%d failed: %s", slot, attempt, MAX_TRIES, message)
            if attempt == MAX_TRIES:
                return "failed"
            time.sleep(30 * attempt)
    return "skipped"


def cmd_build(args: argparse.Namespace) -> None:
    from fogseg.build import build_zarr

    cfg = load_config(args.config)
    out_path = Path(cfg["data"]["processed_path"])
    started = time.time()
    n = build_zarr(Path(cfg["data"]["interim_dir"]), out_path, cfg["channels"], overwrite=args.overwrite)
    log.info("Wrote %d scenes to %s in %.0f s", n, out_path, time.time() - started)


def cmd_label(args: argparse.Namespace) -> None:
    from fogseg.dynamic_threshold import DynamicParams
    from fogseg.label_store import label_dataset, label_store_path, summarize_store
    from fogseg.labels import LabelParams
    from fogseg.teacher_labels import label_teacher_dataset

    cfg = load_config(args.config)
    labels_cfg = cfg["labels"]
    mode = args.mode or labels_cfg["mode"]
    params = LabelParams.from_config(labels_cfg)
    dynamic = DynamicParams.from_config(labels_cfg["dynamic"]) if mode == "dynamic" else None

    dataset_path = Path(cfg["data"]["processed_path"])
    out_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    started = time.time()
    if mode == "teacher":
        n = label_teacher_dataset(dataset_path, out_path, cfg, overwrite=args.overwrite)
    else:
        n = label_dataset(dataset_path, out_path, mode, params, dynamic, overwrite=args.overwrite)
    log.info("Wrote %d scenes to %s in %.0f s", n, out_path, time.time() - started)

    source_counts, mean_fog_fraction = summarize_store(out_path)
    log.info("Threshold sources: %s", source_counts)
    log.info("Mean fog fraction over scenes with decided pixels: %.4f", mean_fog_fraction)


def cmd_dem(args: argparse.Namespace) -> None:
    import xarray as xr

    from fogseg.dem import build_dem_store, http_fetch

    cfg = load_config(args.config)
    out_path = Path(cfg["dem"]["out_path"])
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; use --overwrite to rebuild it")

    started = time.time()
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        build_dem_store(data, Path(cfg["dem"]["raw_dir"]), out_path, http_fetch)
    log.info("Wrote the DEM to %s in %.0f s", out_path, time.time() - started)


def cmd_era5(args: argparse.Namespace) -> None:
    import xarray as xr

    from fogseg.era5 import (
        build_era5_store,
        download_months,
        era5_area,
        has_cds_credentials,
        month_requests,
    )

    cfg = load_config(args.config)
    era5_cfg = cfg["era5"]
    out_path = Path(era5_cfg["out_path"])
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; use --overwrite to rebuild it")
    if not has_cds_credentials():
        raise SystemExit(
            "No CDS key found. Create an account and put your key in ~/.cdsapirc as described at "
            "https://cds.climate.copernicus.eu/how-to-api"
        )

    import cdsapi

    client = cdsapi.Client()

    def retrieve(request: dict, path: Path) -> None:
        client.retrieve("reanalysis-era5-single-levels", request, str(path))

    started = time.time()
    raw_dir = Path(era5_cfg["raw_dir"])
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        times = data["time"].values
        lon, lat, y, x = (data[name].values for name in ("lon", "lat", "y", "x"))

    requests = month_requests(times, era5_area(lon, lat), era5_cfg["variables"])
    failed = download_months(requests, raw_dir, retrieve)
    if failed:
        log.error(
            "%d of %d months failed: %s. No store written; run `fogseg era5` again (cached months are kept)",
            len(failed),
            len(requests),
            failed,
        )
        return
    build_era5_store(times, lon, lat, y, x, raw_dir, out_path)
    log.info("Wrote ERA5 to %s in %.0f s", out_path, time.time() - started)


def cmd_stations(args: argparse.Namespace) -> None:
    import xarray as xr

    from fogseg.observations import collect_observations

    cfg = load_config(args.config)
    data_dir = Path(cfg["stations"]["data_dir"])
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        lon, lat = data["lon"].values, data["lat"].values

    for source in [args.source] if args.source else ["metar", "synop"]:
        observations = collect_observations(source, cfg, lon, lat)
        out_path = data_dir / f"observations_{source}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        observations.to_csv(out_path, index=False)
        log.info(
            "Wrote %d %s observations of %d stations to %s",
            len(observations),
            source,
            observations["station"].nunique(),
            out_path,
        )


def cmd_validate(args: argparse.Namespace) -> None:
    import pandas as pd
    import xarray as xr

    from fogseg.dem import read_dem
    from fogseg.dynamic_threshold import DynamicParams
    from fogseg.label_store import label_store_path, stored_params
    from fogseg.splits import assign_folds, splits_path
    from fogseg.validation import (
        check_season_allowed,
        observation_paths,
        read_observations,
        score_table,
        season_cases,
        season_start,
    )

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    mode = args.labels or labels_cfg["mode"]
    check_season_allowed(cfg, args.season, args.final)

    data_dir = Path(stations_cfg["data_dir"])
    observations = read_observations(data_dir)
    dem = read_dem(Path(cfg["dem"]["out_path"]))

    store_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    with (
        xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        # The hidden test must use the thresholds the labels were made with, even after a config edit.
        params, _ = stored_params(labels.attrs, DynamicParams.from_config(labels_cfg["dynamic"]))
        cases = season_cases(data, labels, observations, cfg, args.season, params, dem)

    if cases.empty:
        log.warning(
            "No station observation matches a scene of season %s; check %s",
            args.season,
            ", ".join(str(path) for path in observation_paths(data_dir)),
        )

    split_path = splits_path(data_dir, args.season)
    if args.season == stations_cfg["tune_season"]:
        if split_path.exists():
            splits = pd.read_csv(split_path)
            start = season_start(cfg, args.season)
            cases = assign_folds(cases, splits, start, cfg["region"]["bbox"], stations_cfg)
        else:
            log.info("%s not found, so the cases get no folds; run `fogseg split` first", split_path)

    scores = score_table(cases, stations_cfg, stations_cfg["bootstrap_samples"], stations_cfg["seed"])
    cases_path = data_dir / f"cases_{mode}_{args.season}.csv"
    scores_path = data_dir / f"scores_{mode}_{args.season}.csv"
    cases.to_csv(cases_path, index=False)
    scores.to_csv(scores_path, index=False)

    main = scores[scores["view"] == "main"].iloc[0]
    log.info(
        "Season %s, %s labels, %d cases: POD %.3f, FAR %.3f, CSI %.3f, bias %.3f",
        args.season,
        mode,
        main["n_cases"],
        main["pod"],
        main["far"],
        main["csi"],
        main["bias"],
    )
    log.info("Full count, all cases: POD %.3f, FAR %.3f", main["pod_full"], main["far_full"])
    held_out = scores[(scores["view"] == "split") & (scores["group"] == "held_out")]
    if not held_out.empty:
        log.info(
            "Full count, held-out cases: POD %.3f, FAR %.3f",
            held_out["pod_full"].iloc[0],
            held_out["far_full"].iloc[0],
        )
    log.info("Wrote %s and %s", cases_path, scores_path)


def require_tune_season(stations_cfg: dict, season: str, verb: str) -> None:
    """Refuse every season but the tuning season, so the test season stays unseen."""
    if season != stations_cfg["tune_season"]:
        raise PermissionError(
            f"Season {season} is not the tuning season {stations_cfg['tune_season']}; only that season is {verb}"
        )


def cmd_split(args: argparse.Namespace) -> None:
    from fogseg.splits import make_splits, splits_path
    from fogseg.validation import season_days

    cfg = load_config(args.config)
    stations_cfg = cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "split")

    out_path = splits_path(Path(stations_cfg["data_dir"]), season)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists and fixes the folds; use --overwrite to redraw them")

    first_day, last_day = season_days(cfg, season)
    splits = make_splits(first_day, last_day, cfg["region"]["bbox"], stations_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)
    log.info("Wrote %d night blocks and station cells to %s", len(splits), out_path)


def cmd_diagnose(args: argparse.Namespace) -> None:
    from contextlib import ExitStack

    import xarray as xr

    from fogseg.diagnosis import case_features, check_fixed_mode, diagnosis_table, read_selection_cases
    from fogseg.label_store import check_labels_match, label_store_path
    from fogseg.labels import LabelParams, check_default_rule
    from fogseg.splits import splits_path
    from fogseg.validation import season_start

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "diagnosed")

    mode = labels_cfg["mode"]
    check_fixed_mode(mode)
    data_dir = Path(stations_cfg["data_dir"])
    cases = read_selection_cases(
        data_dir / f"cases_{mode}_{season}.csv",
        splits_path(data_dir, season),
        season_start(cfg, season),
        cfg["region"]["bbox"],
        stations_cfg,
    )

    era5_path = Path(cfg["era5"]["out_path"])
    if not era5_path.exists():
        log.warning("%s is missing; the ERA5 columns stay empty", era5_path)
    with ExitStack() as stack:
        data = stack.enter_context(xr.open_zarr(cfg["data"]["processed_path"], consolidated=False))
        labels = stack.enter_context(
            xr.open_zarr(label_store_path(Path(labels_cfg["output_dir"]), mode), consolidated=False)
        )
        era5 = (
            stack.enter_context(xr.open_zarr(era5_path, consolidated=False)) if era5_path.exists() else None
        )
        check_labels_match(data, labels)
        params = LabelParams.from_config(labels.attrs)
        check_default_rule(params, "fogseg diagnose")
        cases = case_features(data, labels, era5, cases)

    table = diagnosis_table(cases, params)
    out_path = data_dir / f"diagnosis_{season}.csv"
    table.to_csv(out_path, index=False)

    shares = table[(table["section"] == "first_cause") & (table["statistic"] == "share")]
    log.info(
        "Season %s, %d selection cases; first cause of the visible FLS misses: %s",
        season,
        len(cases),
        ", ".join(f"{row.variable} {row.fls_missed:.1%}" for row in shares.itertuples()),
    )
    log.info("Wrote %s", out_path)


def teacher_held_out_paths(data_dir: Path) -> list[Path]:
    """The table and the decision record of the teacher's held-out run: either one closes the held-out set."""
    return [data_dir / "teacher_held_out.csv", data_dir / "teacher_held_out.json"]


def held_out_paths(data_dir: Path) -> list[Path]:
    """The table and the decision record of the held-out run: either one closes the held-out set."""
    return [data_dir / "experiment_B_held_out.csv", data_dir / "experiment_B_held_out.json"]


def write_atomically(path: Path, text: str) -> None:
    """Write `text` to a `.part` file and move it into place, so a crash never leaves a partial `path`."""
    part = path.with_name(f"{path.name}.part")
    part.write_text(text, encoding="utf-8", newline="")
    os.replace(part, path)


def cmd_experiment(args: argparse.Namespace) -> None:
    if args.held_out:
        cmd_held_out(args)
        return

    import json

    import pandas as pd
    import xarray as xr

    from fogseg.experiment import (
        PHASE_B_VARIANTS,
        combined_variants,
        config_record,
        experiment_cases,
        log_winner,
        run_variants,
    )
    from fogseg.labels import LabelParams

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "experimented on")

    data_dir = Path(stations_cfg["data_dir"])
    for path in held_out_paths(data_dir):
        if path.exists():
            raise FileExistsError(f"{path} exists; a rerun would overwrite the record of the held-out result")
    cases, _ = experiment_cases(cfg, season)

    params = LabelParams.from_config(labels_cfg)
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        table = run_variants(data, cases, params, PHASE_B_VARIANTS, stations_cfg)
        combined = combined_variants(
            table, stations_cfg["temporal_relax_grid"], params.texture, params.fog_index
        )
        table = pd.concat(
            [table, run_variants(data, cases, params, combined, stations_cfg)], ignore_index=True
        )

    out_path = data_dir / "experiment_B.csv"
    table.to_csv(out_path, index=False)
    record = config_record(season, labels_cfg["mode"], params, stations_cfg)
    write_atomically(data_dir / "experiment_B_config.json", json.dumps(record, indent=2))

    log_winner(table, stations_cfg["matched_far"])
    log.info("Wrote %s", out_path)


def cmd_held_out(args: argparse.Namespace) -> None:
    import json

    import pandas as pd
    import xarray as xr

    from fogseg.experiment import (
        check_same_config,
        config_record,
        decision_record,
        experiment_cases,
        held_out_comparison,
        winner,
    )
    from fogseg.labels import LabelParams
    from fogseg.validation import season_start

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "experimented on")

    data_dir = Path(stations_cfg["data_dir"])
    table_path, record_path = held_out_paths(data_dir)
    for path in (table_path, record_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; the held-out set is opened once per phase")
    cv_path, config_path = data_dir / "experiment_B.csv", data_dir / "experiment_B_config.json"
    for path in (cv_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `fogseg experiment --phase B` first")

    params = LabelParams.from_config(labels_cfg)
    recorded = json.loads(config_path.read_text(encoding="utf-8"))
    check_same_config(recorded, config_record(season, labels_cfg["mode"], params, stations_cfg))
    cv_table = pd.read_csv(cv_path)
    best = winner(cv_table)
    options = json.loads(cv_table.loc[cv_table["variant"] == best, "options"].iloc[0])

    selection, held_out = experiment_cases(cfg, season)
    with xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data:
        table, decision = held_out_comparison(
            data, selection, held_out, params, (best, options), stations_cfg, season_start(cfg, season)
        )
    record = decision_record(decision, params, options)

    log.info("Held-out baseline: %s at %g", params.fog_index, record["baseline_threshold"])
    log.info(
        "Held-out, winner %s at %g: POD gain %+.3f (95%% interval %+.3f to %+.3f); "
        "FAR %.3f against %.3f (difference %+.3f to %+.3f)",
        decision.winner,
        decision.value,
        decision.pod_gain,
        decision.pod_gain_low,
        decision.pod_gain_high,
        decision.far_winner,
        decision.far_baseline,
        decision.far_difference_low,
        decision.far_difference_high,
    )
    if record["keep"]:
        changes = ", ".join(f"{key}: {value}" for key, value in record["config_changes"].items())
        log.info("Keep %s: set %s", decision.winner, changes)
    else:
        log.info("Do not keep %s: the labels stay as configured", decision.winner)

    write_atomically(table_path, table.to_csv(index=False))
    write_atomically(record_path, json.dumps(record, indent=2))
    log.info("Wrote %s and %s", table_path, record_path)


def cmd_teacher(args: argparse.Namespace) -> None:
    if args.held_out:
        cmd_teacher_held_out(args)
        return

    import dataclasses
    import json
    from contextlib import ExitStack

    import xarray as xr

    from fogseg.experiment import experiment_cases, pooled_scores, winner
    from fogseg.features import case_features, open_era5, read_static
    from fogseg.labels import LabelParams
    from fogseg.teacher import (
        MODEL_SETTINGS,
        TeacherParams,
        chosen_length,
        cross_validate_correction,
        cross_validate_settings,
        fit_teacher,
        oof_table,
        save_teacher,
        teacher_record,
    )

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "used to fit the teacher")
    if labels_cfg["mode"] != "fixed":
        raise ValueError(f"The teacher needs labels.mode fixed, not {labels_cfg['mode']!r}")

    data_dir = Path(stations_cfg["data_dir"])
    for path in teacher_held_out_paths(data_dir):
        if path.exists():
            raise FileExistsError(f"{path} exists; a rerun would overwrite the record of the held-out result")
    selection, _ = experiment_cases(cfg, season)

    params, teacher_params = LabelParams.from_config(labels_cfg), TeacherParams.from_config(cfg["teacher"])
    groups = teacher_params.feature_groups
    with ExitStack() as stack:
        data = stack.enter_context(xr.open_zarr(cfg["data"]["processed_path"], consolidated=False))
        static = read_static(data, Path(cfg["dem"]["out_path"]), groups)
        era5_store = open_era5(Path(cfg["era5"]["out_path"]), groups)
        if era5_store is not None:
            stack.enter_context(era5_store)
        features = case_features(data, selection, static, era5_store, params, groups)

    table, probabilities = cross_validate_settings(features, selection, teacher_params, stations_cfg)
    best = winner(table)
    pooled = pooled_scores(table).loc[best]
    log.info(
        "Teacher winner %s: p* %g, pooled POD %.3f, FAR %.3f",
        best,
        pooled["value"],
        pooled["pod"],
        pooled["far"],
    )
    teacher = fit_teacher(
        features,
        selection,
        (best, dict(MODEL_SETTINGS)[best]),
        probabilities[best],
        float(pooled["value"]),
        teacher_params,
        stations_cfg["seed"],
    )

    oof = oof_table(probabilities[best], selection)
    correction = cross_validate_correction(
        probabilities[best],
        selection,
        oof,
        teacher.p_star,
        teacher_params.ignore_halfwidth,
        teacher_params.idw_length_km,
        stations_cfg["n_folds"],
    )
    length = chosen_length(correction)
    teacher = dataclasses.replace(teacher, correction_km=length)
    pooled_correction = correction[correction["fold"].astype(str) == "pooled"].set_index("length_km")
    none, kept = (
        pooled_correction.loc["none"],
        pooled_correction.loc["none" if length is None else str(length)],
    )
    log.info(
        "Station correction: %s, pooled POD %.3f, FAR %.3f against %.3f, %.3f without",
        "no correction" if length is None else f"{length:g} km",
        kept["pod"],
        kept["far"],
        none["pod"],
        none["far"],
    )

    cv_path, oof_path = data_dir / "teacher_cv.csv", data_dir / "teacher_oof.csv"
    correction_path = data_dir / "teacher_correction_cv.csv"
    table.to_csv(cv_path, index=False)
    oof.to_csv(oof_path, index=False)
    correction.to_csv(correction_path, index=False)
    save_teacher(teacher, Path(teacher_params.model_path))
    record = teacher_record(
        season, labels_cfg["mode"], params, stations_cfg, dataclasses.asdict(teacher_params)
    )
    write_atomically(data_dir / "teacher_config.json", json.dumps(record, indent=2))
    log.info("Wrote %s, %s, %s and %s", cv_path, oof_path, correction_path, teacher_params.model_path)


def cmd_teacher_held_out(args: argparse.Namespace) -> None:
    import dataclasses
    import json
    from contextlib import ExitStack

    import pandas as pd
    import xarray as xr

    from fogseg.experiment import check_same_config, experiment_cases
    from fogseg.features import open_era5, read_static
    from fogseg.labels import LabelParams
    from fogseg.teacher import TeacherParams, load_teacher, teacher_held_out, teacher_record
    from fogseg.validation import season_start

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = args.season or stations_cfg["tune_season"]
    require_tune_season(stations_cfg, season, "used to fit the teacher")

    data_dir = Path(stations_cfg["data_dir"])
    table_path, record_path = teacher_held_out_paths(data_dir)
    for path in (table_path, record_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; the held-out set is opened once per phase")
    teacher_cfg = cfg["teacher"]
    model_path, config_path = Path(teacher_cfg["model_path"]), data_dir / "teacher_config.json"
    oof_path = data_dir / "teacher_oof.csv"
    for path in (model_path, config_path, oof_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `fogseg teacher` first")

    params, teacher_params = LabelParams.from_config(labels_cfg), TeacherParams.from_config(teacher_cfg)
    recorded = json.loads(config_path.read_text(encoding="utf-8"))
    current = teacher_record(
        season, labels_cfg["mode"], params, stations_cfg, dataclasses.asdict(teacher_params)
    )
    check_same_config(recorded, current)
    groups = teacher_params.feature_groups
    teacher = load_teacher(model_path, groups)
    oof = pd.read_csv(oof_path, dtype={"station": str}, parse_dates=["scene_time"])

    selection, held_out = experiment_cases(cfg, season)
    with ExitStack() as stack:
        data = stack.enter_context(xr.open_zarr(cfg["data"]["processed_path"], consolidated=False))
        static = read_static(data, Path(cfg["dem"]["out_path"]), groups)
        era5_store = open_era5(Path(cfg["era5"]["out_path"]), groups)
        if era5_store is not None:
            stack.enter_context(era5_store)
        table, teacher_decision, correction_decision = teacher_held_out(
            data,
            selection,
            held_out,
            teacher,
            oof,
            params,
            stations_cfg,
            season_start(cfg, season),
            static,
            era5_store,
        )

    decisions = {"teacher": teacher_decision, "correction": correction_decision}
    record = {
        **{
            name: None if decision is None else {**dataclasses.asdict(decision), "keep": bool(decision.keep)}
            for name, decision in decisions.items()
        },
        "p_star": teacher.p_star,
        "setting": teacher.setting,
        "correction_km": teacher.correction_km,
        "config_changes": {"labels.mode": "teacher"} if teacher_decision.keep else {},
    }
    for name, decision in decisions.items():
        if decision is not None:
            log_keep_decision(name, decision)
    if teacher_decision.keep:
        log.info("Keep the teacher: set labels.mode: teacher")
    else:
        log.info("Do not keep the teacher: the labels stay as configured")

    write_atomically(table_path, table.to_csv(index=False))
    write_atomically(record_path, json.dumps(record, indent=2))
    log.info("Wrote %s and %s", table_path, record_path)


def log_keep_decision(name: str, decision: KeepDecision) -> None:
    log.info(
        "Held-out, %s against its reference at p* %g: POD gain %+.3f (95%% interval %+.3f to %+.3f); "
        "FAR %.3f against %.3f (difference %+.3f to %+.3f)",
        name,
        decision.value,
        decision.pod_gain,
        decision.pod_gain_low,
        decision.pod_gain_high,
        decision.far_winner,
        decision.far_baseline,
        decision.far_difference_low,
        decision.far_difference_high,
    )


def log_decision(name: str, decision: Decision, report_only: bool = False) -> None:
    """Log the best value of a tuned parameter; a `report_only` grid does not decide the config value."""
    if report_only:
        verdict = "report only"
    elif decision.changed:
        verdict = "changed"
    else:
        verdict = "kept"
    log.info(
        "Tune %s: current %g, best %g, CSI %.3f -> %.3f, difference %+.3f to %+.3f, ignored %d -> %d, %s",
        name,
        decision.current,
        decision.best,
        decision.csi_current,
        decision.csi_best,
        decision.difference_low,
        decision.difference_high,
        decision.ignored_current,
        decision.ignored_best,
        verdict,
    )


def cmd_tune(args: argparse.Namespace) -> None:
    import dataclasses

    import xarray as xr

    from fogseg.dynamic_threshold import DynamicParams
    from fogseg.label_store import label_store_path
    from fogseg.labels import LabelParams, check_default_rule
    from fogseg.splits import read_splits, selection_cases, splits_path
    from fogseg.tuning import tune_parameter
    from fogseg.validation import (
        TRUTH_NO,
        TRUTH_YES,
        deduplicate,
        observation_paths,
        read_observations,
        season_cases,
        season_start,
    )

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = stations_cfg["tune_season"]
    params = LabelParams.from_config(labels_cfg)
    check_default_rule(params, "fogseg tune")
    dynamic = DynamicParams.from_config(labels_cfg["dynamic"])

    data_dir = Path(stations_cfg["data_dir"])
    split_path = splits_path(data_dir, season)
    splits = read_splits(split_path)
    observations = read_observations(data_dir)

    store_path = label_store_path(Path(labels_cfg["output_dir"]), "dynamic")
    with (
        xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        # Only the selection cases choose: the held-out fold never reaches the tuning
        cases = selection_cases(
            deduplicate(season_cases(data, labels, observations, cfg, season, params)),
            splits,
            season_start(cfg, season),
            cfg["region"]["bbox"],
            stations_cfg,
        )
        cases = cases[cases["truth_fls"].isin([TRUTH_NO, TRUTH_YES]) & ~cases["hidden"]]
        if cases.empty:
            log.warning(
                "No selection case with a known truth in season %s; check %s and %s",
                season,
                ", ".join(str(path) for path in observation_paths(data_dir)),
                split_path,
            )
            return

        table, offset = tune_parameter(
            data, cases, params, dynamic, "offset", "dynamic", dynamic.offset, stations_cfg
        )
        table.to_csv(data_dir / "tuning_offset.csv", index=False)
        log_decision("offset", offset)

        # The fallback is tuned with the offset that was just decided
        chosen = dataclasses.replace(dynamic, offset=offset.best if offset.changed else dynamic.offset)
        table, fallback = tune_parameter(
            data, cases, params, chosen, "btd_fixed", "dynamic", params.btd_fixed, stations_cfg
        )
        table.to_csv(data_dir / "tuning_btd_fixed.csv", index=False)
        log_decision("btd_fixed", fallback)

        # Fixed mode is scored for comparison only: the fallback above decides the config value
        table, fixed = tune_parameter(
            data, cases, params, chosen, "btd_fixed", "fixed", params.btd_fixed, stations_cfg
        )
        table.to_csv(data_dir / "tuning_btd_fixed_fixed_mode.csv", index=False)
        log_decision("btd_fixed (fixed mode)", fixed, report_only=True)


def cmd_inspect(args: argparse.Namespace) -> None:
    import pandas as pd
    import xarray as xr

    from fogseg.dynamic_threshold import DynamicParams
    from fogseg.inspect import inspect_figure
    from fogseg.label_store import label_store_path, stored_params

    cfg = load_config(args.config)
    labels_cfg = cfg["labels"]
    mode = args.labels or labels_cfg["mode"]
    config_dynamic = DynamicParams.from_config(labels_cfg["dynamic"])

    dataset_path = Path(cfg["data"]["processed_path"])
    store_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        # The figure must show the pixels the labels were made from, even after a config edit.
        params, dynamic = stored_params(labels.attrs, config_dynamic)
        wanted = pd.Timestamp(args.time)
        if wanted.tzinfo is not None:
            wanted = wanted.tz_convert("UTC").tz_localize(None)
        png = inspect_figure(data, labels, wanted, params, dynamic, Path(labels_cfg["figures_dir"]))
    log.info("Wrote %s", png)


def cmd_select_scenes(args: argparse.Namespace) -> None:
    import xarray as xr

    from fogseg.check_scenes import group_counts, scene_statistics, select_check_scenes, write_check_scenes
    from fogseg.label_store import label_store_path

    cfg = load_config(args.config)
    labels_cfg = cfg["labels"]
    mode = args.labels or labels_cfg["mode"]
    # The committed check scenes are the ones judged in docs/notes/label_check.md; replace them only on purpose.
    out_path = Path(args.config).parent / "check_scenes.yaml"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; use --overwrite to replace it")

    dataset_path = Path(cfg["data"]["processed_path"])
    store_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    with (
        xr.open_zarr(dataset_path, consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        stats = scene_statistics(data, labels)

    selected = select_check_scenes(stats)
    write_check_scenes(selected, mode, out_path)
    log.info("Wrote %d check scenes to %s", len(selected), out_path)
    log.info("Scenes per group: %s", group_counts(selected))


def cmd_check(args: argparse.Namespace) -> None:
    """Test EUMETSAT access in seconds: token, search, and the first 1 KB of one download."""
    from fogseg.download import connect, find_product

    cfg = load_config(args.config)
    collection = connect(cfg["data"]["collection"])
    print("Token valid until:", collection.datastore.token.expiration)

    slot = scene_times(cfg)[0]
    product = find_product(collection, slot)
    print("Search OK:", product)
    entry = next(e for e in product.entries if e.endswith(".nat"))

    # Three different downloads: if all fail, the account is blocked; if only some fail,
    # the problem is specific to one collection or one download method.
    cloud_mask = collection.datastore.get_collection("EO:EUM:DAT:MSG:CLM")
    tests = [
        ("SEVIRI L1.5, only the .nat file", product, entry),
        ("SEVIRI L1.5, whole product zip", product, None),
        ("SEVIRI cloud mask (other collection)", find_product(cloud_mask, slot), None),
    ]
    for name, prod, ent in tests:
        if prod is None:
            print(f"[SKIP]   {name}: no product found for {slot}")
            continue
        try:
            with prod.open(entry=ent, chunk=(0, 1024)) as stream:
                print(f"[OK]     {name}: received {len(stream.read())} bytes")
        except Exception as err:  # noqa: BLE001 - report every failure kind, then test the next one
            status, message = server_message(err)
            print(f"[FAILED] {name}: status {status} {message}")


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["interim_dir"])

    slots = scene_times(cfg)
    if args.start:
        slots = [s for s in slots if s >= dt.datetime.fromisoformat(args.start)]
    if args.end:
        slots = [s for s in slots if s <= dt.datetime.fromisoformat(args.end)]
    todo = [s for s in slots if not (out_dir / scene_filename(s)).exists()]
    n_done = len(slots) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    log.info("%d slots in config range, %d already done, %d to fetch now", len(slots), n_done, len(todo))
    log.info(
        "Raw download volume: ~%.1f GB (deleted after cropping, never more than %d raw files on disk)",
        len(todo) * RAW_MB / 1000,
        args.workers,
    )
    if args.dry_run or not todo:
        return

    from fogseg.download import connect

    # A crash can leave half-written outputs and undeleted raw files behind; they are never reused
    raw_dir = Path(cfg["data"]["raw_dir"])
    leftovers = list(out_dir.glob("*.part")) + list(raw_dir.glob("*.nat")) + list(raw_dir.glob("*.part"))
    for leftover in leftovers:
        log.info("Removing leftover %s", leftover.name)
        leftover.unlink()

    collection = connect(cfg["data"]["collection"])
    counts = {"ok": 0, "missing": 0, "failed": 0, "blocked": 0, "skipped": 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_with_retry, s, collection, cfg): s for s in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            slot = futures[future]
            status = future.result()
            counts[status] += 1
            if status == "skipped":  # run was stopped; don't print hundreds of identical lines
                continue
            seconds_per_scene = (time.time() - started) / done
            minutes_left = seconds_per_scene * (len(todo) - done) / 60
            log.info("[%d/%d] %s %s | ~%.1f min left", done, len(todo), slot, status, minutes_left)
    log.info("Done: %s", counts)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fogseg")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download, crop and save SEVIRI scenes (resumable)")
    fetch.add_argument("--config", default="configs/default.yaml")
    fetch.add_argument("--start", help="only slots at/after this time, e.g. 2023-11-20T21:00")
    fetch.add_argument("--end", help="only slots at/before this time")
    fetch.add_argument("--limit", type=int, help="fetch at most N scenes (for testing)")
    fetch.add_argument("--workers", type=int, default=2, help="parallel downloads")
    fetch.add_argument("--dry-run", action="store_true", help="only show what would be fetched")
    fetch.set_defaults(func=cmd_fetch)

    check = sub.add_parser("check", help="quickly test EUMETSAT login, search and download access")
    check.add_argument("--config", default="configs/default.yaml")
    check.set_defaults(func=cmd_check)

    build = sub.add_parser("build", help="stack all cropped scenes into one ML-ready Zarr store")
    build.add_argument("--config", default="configs/default.yaml")
    build.add_argument("--overwrite", action="store_true", help="rebuild if the store already exists")
    build.set_defaults(func=cmd_build)

    label = sub.add_parser("label", help="write fog / low-stratus labels for every scene")
    label.add_argument("--config", default="configs/default.yaml")
    label.add_argument(
        "--mode", choices=["fixed", "dynamic", "teacher"], help="default: labels.mode of the config"
    )
    label.add_argument("--overwrite", action="store_true", help="replace the label store if it exists")
    label.set_defaults(func=cmd_label)

    dem_cmd = sub.add_parser("dem", help="average the Copernicus DEM onto the SEVIRI grid (resumable)")
    dem_cmd.add_argument("--config", default="configs/default.yaml")
    dem_cmd.add_argument("--overwrite", action="store_true", help="replace the DEM store if it exists")
    dem_cmd.set_defaults(func=cmd_dem)

    era5_cmd = sub.add_parser(
        "era5",
        help="download ERA5 surface fields and put them on the SEVIRI grid (resumable, needs a CDS key)",
    )
    era5_cmd.add_argument("--config", default="configs/default.yaml")
    era5_cmd.add_argument("--overwrite", action="store_true", help="replace the ERA5 store if it exists")
    era5_cmd.set_defaults(func=cmd_era5)

    stations = sub.add_parser(
        "stations", help="download METAR and SYNOP observations of the dataset nights (resumable)"
    )
    stations.add_argument("--config", default="configs/default.yaml")
    stations.add_argument("--source", choices=["metar", "synop"], help="default: both")
    stations.set_defaults(func=cmd_stations)

    validate = sub.add_parser("validate", help="score the label store against station observations")
    validate.add_argument("--config", default="configs/default.yaml")
    validate.add_argument("--labels", choices=["fixed", "dynamic"], help="default: labels.mode of the config")
    validate.add_argument("--season", required=True, help="season name from the config, e.g. 2023-24")
    validate.add_argument(
        "--final", action="store_true", help="allow the test season, which is scored once at the end"
    )
    validate.set_defaults(func=cmd_validate)

    split = sub.add_parser(
        "split", help="draw the night blocks and station cells of the tuning season into the folds"
    )
    split.add_argument("--config", default="configs/default.yaml")
    split.add_argument("--season", help="default: stations.tune_season of the config")
    split.add_argument("--overwrite", action="store_true", help="redraw the folds if the file exists")
    split.set_defaults(func=cmd_split)

    diagnose = sub.add_parser(
        "diagnose", help="find why visible station fog is missed, on the selection cases of the tuning season"
    )
    diagnose.add_argument("--config", default="configs/default.yaml")
    diagnose.add_argument("--season", help="default and only allowed: stations.tune_season of the config")
    diagnose.set_defaults(func=cmd_diagnose)

    experiment = sub.add_parser(
        "experiment", help="compare label variants at a matched FAR, cross-validated on the tuning season"
    )
    experiment.add_argument("--config", default="configs/default.yaml")
    experiment.add_argument("--phase", required=True, choices=["B"], help="the set of variants to compare")
    experiment.add_argument("--season", help="default and only allowed: stations.tune_season of the config")
    experiment.add_argument(
        "--held-out",
        action="store_true",
        help="score the winner of experiment_B.csv against the configured rule on the held-out cases, once",
    )
    experiment.set_defaults(func=cmd_experiment)

    teacher = sub.add_parser(
        "teacher",
        help="cross-validate the teacher settings on the tuning season, then fit and save the winner",
    )
    teacher.add_argument("--config", default="configs/default.yaml")
    teacher.add_argument("--season", help="default and only allowed: stations.tune_season of the config")
    teacher.add_argument(
        "--held-out",
        action="store_true",
        help="score the saved teacher against the configured rule on the held-out cases, once",
    )
    teacher.set_defaults(func=cmd_teacher)

    tune = sub.add_parser(
        "tune", help="score grids of the BTD offset and the fallback threshold on the tuning season"
    )
    tune.add_argument("--config", default="configs/default.yaml")
    tune.set_defaults(func=cmd_tune)

    inspect_cmd = sub.add_parser(
        "inspect", help="write a check figure of one scene (RGB, labels, histogram, sza)"
    )
    inspect_cmd.add_argument("--config", default="configs/default.yaml")
    inspect_cmd.add_argument("--time", required=True, help="scene time in UTC, e.g. 2023-11-21T00:00")
    inspect_cmd.add_argument(
        "--labels", choices=["fixed", "dynamic"], help="default: labels.mode of the config"
    )
    inspect_cmd.set_defaults(func=cmd_inspect)

    select_scenes = sub.add_parser(
        "select-scenes", help="choose the training-season scenes to check, into check_scenes.yaml"
    )
    select_scenes.add_argument("--config", default="configs/default.yaml")
    select_scenes.add_argument(
        "--labels", choices=["fixed", "dynamic"], help="default: labels.mode of the config"
    )
    select_scenes.add_argument(
        "--overwrite", action="store_true", help="replace check_scenes.yaml if it exists"
    )
    select_scenes.set_defaults(func=cmd_select_scenes)

    args = parser.parse_args()
    cfg = load_config(args.config)
    log_dir = Path(cfg["data"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / f"{args.command}.log")],
    )
    args.func(args)


if __name__ == "__main__":
    main()
