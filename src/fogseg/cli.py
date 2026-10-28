"""Command line entry point: `fogseg fetch --config configs/default.yaml`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from fogseg.config import load_config, scene_filename, scene_times

if TYPE_CHECKING:
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

    cfg = load_config(args.config)
    labels_cfg = cfg["labels"]
    mode = args.mode or labels_cfg["mode"]
    params = LabelParams.from_config(labels_cfg)
    dynamic = DynamicParams.from_config(labels_cfg["dynamic"]) if mode == "dynamic" else None

    dataset_path = Path(cfg["data"]["processed_path"])
    out_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    started = time.time()
    n = label_dataset(dataset_path, out_path, mode, params, dynamic, overwrite=args.overwrite)
    log.info("Wrote %d scenes to %s in %.0f s", n, out_path, time.time() - started)

    source_counts, mean_fog_fraction = summarize_store(out_path)
    log.info("Threshold sources: %s", source_counts)
    log.info("Mean fog fraction over scenes with decided pixels: %.4f", mean_fog_fraction)


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
    import xarray as xr

    from fogseg.dynamic_threshold import DynamicParams
    from fogseg.label_store import label_store_path, stored_params
    from fogseg.validation import (
        check_season_allowed,
        observation_paths,
        read_observations,
        score_table,
        season_cases,
    )

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    mode = args.labels or labels_cfg["mode"]
    check_season_allowed(cfg, args.season, args.final)

    data_dir = Path(stations_cfg["data_dir"])
    observations = read_observations(data_dir)

    store_path = label_store_path(Path(labels_cfg["output_dir"]), mode)
    with (
        xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        # The hidden test must use the thresholds the labels were made with, even after a config edit.
        params, _ = stored_params(labels.attrs, DynamicParams.from_config(labels_cfg["dynamic"]))
        cases = season_cases(data, labels, observations, cfg, args.season, params)

    if cases.empty:
        log.warning(
            "No station observation matches a scene of season %s; check %s",
            args.season,
            ", ".join(str(path) for path in observation_paths(data_dir)),
        )

    scores = score_table(
        cases, stations_cfg["lat_bands"], stations_cfg["bootstrap_samples"], stations_cfg["seed"]
    )
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
    log.info("Wrote %s and %s", cases_path, scores_path)


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
    from fogseg.labels import LabelParams
    from fogseg.tuning import tune_parameter
    from fogseg.validation import (
        TRUTH_NO,
        TRUTH_YES,
        deduplicate,
        observation_paths,
        read_observations,
        season_cases,
    )

    cfg = load_config(args.config)
    labels_cfg, stations_cfg = cfg["labels"], cfg["stations"]
    season = stations_cfg["tune_season"]
    params = LabelParams.from_config(labels_cfg)
    dynamic = DynamicParams.from_config(labels_cfg["dynamic"])

    data_dir = Path(stations_cfg["data_dir"])
    observations = read_observations(data_dir)

    store_path = label_store_path(Path(labels_cfg["output_dir"]), "dynamic")
    with (
        xr.open_zarr(cfg["data"]["processed_path"], consolidated=False) as data,
        xr.open_zarr(store_path, consolidated=False) as labels,
    ):
        cases = deduplicate(season_cases(data, labels, observations, cfg, season, params))
        cases = cases[cases["truth_fls"].isin([TRUTH_NO, TRUTH_YES]) & ~cases["hidden"]]
        if cases.empty:
            log.warning(
                "No station observation with a known truth matches a scene of season %s; check %s",
                season,
                ", ".join(str(path) for path in observation_paths(data_dir)),
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
    label.add_argument("--mode", choices=["fixed", "dynamic"], help="default: labels.mode of the config")
    label.add_argument("--overwrite", action="store_true", help="replace the label store if it exists")
    label.set_defaults(func=cmd_label)

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
