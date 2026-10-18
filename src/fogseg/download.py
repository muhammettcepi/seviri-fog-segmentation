"""Search and download SEVIRI Level 1.5 scenes from the EUMETSAT Data Store with eumdac."""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path

import eumdac


def connect(collection_id: str) -> eumdac.collection.Collection:
    """Log in with the API consumer key + secret and open a Data Store collection.

    Credentials come from the environment variables EUMETSAT_KEY / EUMETSAT_SECRET,
    so they never appear in code or config files.
    """
    key = os.environ.get("EUMETSAT_KEY", "").strip()
    secret = os.environ.get("EUMETSAT_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("Set EUMETSAT_KEY and EUMETSAT_SECRET environment variables first")
    token = eumdac.AccessToken((key, secret))  # short-lived; eumdac renews it when it expires
    return eumdac.DataStore(token).get_collection(collection_id)


def find_product(collection, slot: dt.datetime):
    """Return the product whose scan starts at the nominal slot time, or None.

    A full-disk scan takes ~12 minutes, so searching [slot+1min, slot+11min] only overlaps
    the scan that started at `slot`, not its neighbours.
    """
    search_start = slot + dt.timedelta(minutes=1)
    search_end = slot + dt.timedelta(minutes=11)
    products = list(collection.search(dtstart=search_start, dtend=search_end))
    if not products:
        return None

    def offset_from_slot(product) -> dt.timedelta:
        scan_start = product.sensing_start.replace(tzinfo=None)
        return abs(scan_start - slot)

    return min(products, key=offset_from_slot)


def download_nat(product, out_dir: Path) -> Path:
    """Download only the .nat image file of a product (not its metadata files)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = next(e for e in product.entries if e.endswith(".nat"))
    out_path = out_dir / Path(entry).name
    tmp_path = out_path.with_suffix(".nat.part")  # a crash never leaves a half file that looks complete
    with product.open(entry=entry) as fsrc, open(tmp_path, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)
    tmp_path.replace(out_path)
    return out_path
