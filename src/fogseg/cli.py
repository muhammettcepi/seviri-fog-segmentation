"""Command line entry point: `fogseg fetch --config configs/default.yaml`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fogseg.config import load_config, scene_filename, scene_times

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
    log.info("Mean fog fraction over scenes with night pixels: %.4f", mean_fog_fraction)


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
