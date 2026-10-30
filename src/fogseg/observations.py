"""Download METAR and SYNOP observations of the dataset nights and decode them into one table per source."""

from __future__ import annotations

import datetime as dt
import http.client
import io
import json
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fogseg.config import scene_times
from fogseg.metar import ceiling_m, prevailing_visibility_m
from fogseg.synop import decode_synop
from fogseg.validation import STATION_COLUMNS, night_of, select_stations

log = logging.getLogger("fogseg")

IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
IEM_REQUEST_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OGIMET_URL = "https://www.ogimet.com/cgi-bin/getsynop"
OBSERVATION_COLUMNS = ["source", "station", "time", "visibility_m", "ceiling_m"]
NIGHT_START = dt.timedelta(hours=20)
NIGHT_LENGTH = dt.timedelta(hours=10)
INVALID_ANSWER_PAUSE_FACTOR = 6
LOG_EVERY = 25
ISD_HISTORY_FILE = "isd-history.csv"
WMO_WBAN = "99999"
SCENE_MARGIN = np.timedelta64(1, "h")
MISSING_METAR = {"", "M"}


def fetch_text(url: str, timeout: float = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def iem_stations(network_geojson: str) -> pd.DataFrame:
    """The stations of one IEM network, with coordinates and elevation from its GeoJSON file."""
    features = json.loads(network_geojson)["features"]
    rows = [
        (
            feature["id"],
            "metar",
            feature["geometry"]["coordinates"][1],
            feature["geometry"]["coordinates"][0],
            feature["properties"].get("elevation"),
        )
        for feature in features
    ]
    return pd.DataFrame(rows, columns=STATION_COLUMNS).astype({"elevation_m": float})


def wmo_stations(isd_history_csv: str) -> pd.DataFrame:
    """The WMO stations of NOAA's `isd-history.csv`, one row per station with its latest record.

    The WMO id is the USAF id without its trailing digit.
    """
    isd = pd.read_csv(io.StringIO(isd_history_csv), dtype=str, keep_default_na=False)
    wmo = isd[isd["WBAN"] == WMO_WBAN].sort_values("END", kind="stable")
    stations = pd.DataFrame(
        {
            "station": wmo["USAF"].str[:5],
            "source": "synop",
            "lat": pd.to_numeric(wmo["LAT"], errors="coerce"),
            "lon": pd.to_numeric(wmo["LON"], errors="coerce"),
            "elevation_m": pd.to_numeric(wmo["ELEV(M)"], errors="coerce"),
        }
    )
    latest = stations.drop_duplicates("station", keep="last")
    return latest.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def metar_url(station: str, start: dt.date, end: dt.date) -> str:
    """IEM request for the raw METARs of one station from `start` to `end` (UTC days)."""
    query = [
        ("station", station),
        ("data", "metar"),
        ("year1", start.year),
        ("month1", start.month),
        ("day1", start.day),
        ("year2", end.year),
        ("month2", end.month),
        ("day2", end.day),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("report_type", 3),
        ("report_type", 4),
    ]
    return f"{IEM_REQUEST_URL}?{urllib.parse.urlencode(query, safe='/')}"


def synop_url(block: str, start: dt.datetime, end: dt.datetime) -> str:
    """Ogimet request for the SYNOP messages of all stations of one WMO block."""
    return f"{OGIMET_URL}?begin={start:%Y%m%d%H%M}&end={end:%Y%m%d%H%M}&block={block}"


def download(url: str, path: Path, pause_s: float, max_tries: int, is_valid: Callable[[str], bool]) -> bool:
    """Save the answer to `url` as `path` unless it exists; only an answer accepted by `is_valid` is saved.

    The servers throttle by answering with an empty body, so there is a pause before every try and a
    longer one after an invalid answer. Returns whether `path` exists afterwards.
    """
    if path.exists():
        return True
    for attempt in range(1, max_tries + 1):
        time.sleep(pause_s)
        text = _answer(url)
        if is_valid(text):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            partial.write_text(text, encoding="utf-8")
            partial.replace(path)
            return True
        log.warning("Invalid answer to %s (try %d/%d)", url, attempt, max_tries)
        if attempt < max_tries:
            time.sleep(pause_s * INVALID_ANSWER_PAUSE_FACTOR)
    return False


def _answer(url: str) -> str:
    """The answer to `url`, empty after a network error or a connection that broke while reading."""
    try:
        return fetch_text(url)
    except (OSError, http.client.HTTPException) as err:
        log.warning("Request %s failed: %s", url, err)
        return ""


def observation_table(rows: list[tuple], skipped_lines: int) -> pd.DataFrame:
    """The observation table of `rows`; `attrs["skipped_lines"]` holds the number of malformed lines."""
    table = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    table = table.astype({"time": "datetime64[ns]", "visibility_m": float, "ceiling_m": float})
    table.attrs["skipped_lines"] = skipped_lines
    return table


def parse_metar_csv(text: str) -> pd.DataFrame:
    """Decode the raw METARs of an IEM answer with the header `station,valid,metar`.

    Lines are split on the first two commas only, because the METAR text may contain commas. Malformed
    lines are skipped and counted in `attrs["skipped_lines"]`.
    """
    rows = []
    skipped_lines = 0
    for line in text.splitlines()[1:]:
        parts = line.split(",", 2)
        if len(parts) < 3:
            skipped_lines += 1
            continue
        station, valid, metar = parts
        if metar in MISSING_METAR:
            continue
        try:
            time_utc = dt.datetime.strptime(valid, "%Y-%m-%d %H:%M")
        except ValueError:
            skipped_lines += 1
            continue
        rows.append(("metar", station, time_utc, prevailing_visibility_m(metar), ceiling_m(metar)))
    return observation_table(rows, skipped_lines)


def parse_synop_text(text: str) -> pd.DataFrame:
    """Decode the Ogimet lines `station,year,month,day,hour,minute,message`; NIL reports are dropped.

    Lines without a valid date and time are skipped and counted in `attrs["skipped_lines"]`.
    """
    rows = []
    skipped_lines = 0
    for line in text.splitlines():
        parts = line.split(",", 6)
        if len(parts) < 7:
            if line.strip():
                skipped_lines += 1
            continue
        station, year, month, day, hour, minute, message = parts
        try:
            time_utc = dt.datetime(int(year), int(month), int(day), int(hour), int(minute))
        except (ValueError, OverflowError):
            skipped_lines += 1
            continue
        synop = decode_synop(message)
        if synop is not None:
            rows.append(("synop", station, time_utc, synop.visibility_m, synop.ceiling_m))
    return observation_table(rows, skipped_lines)


def night_windows(cfg: dict) -> list[tuple[dt.datetime, dt.datetime]]:
    """20:00 UTC on every dataset night to 06:00 UTC on the next day."""
    nights = np.unique(night_of(scene_times(cfg)))
    starts = [pd.Timestamp(night).to_pydatetime() + NIGHT_START for night in nights]
    return [(start, start + NIGHT_LENGTH) for start in starts]


def is_metar_answer(text: str) -> bool:
    return text.startswith("station,")


def is_synop_answer(text: str) -> bool:
    return "AAXX" in text


def is_isd_answer(text: str) -> bool:
    return text.startswith('"USAF"')


def is_geojson_answer(text: str) -> bool:
    return text.startswith("{")


ANSWER_CHECKS = {"metar": is_metar_answer, "synop": is_synop_answer}
PAUSE_KEYS = {"metar": "iem_pause_s", "synop": "ogimet_pause_s"}
PARSERS = {"metar": parse_metar_csv, "synop": parse_synop_text}


def candidate_stations(source: str, stations_cfg: dict, data_dir: Path) -> pd.DataFrame:
    """All stations of the source that might be used, before they are matched to the grid.

    The station files are downloaded like the observations, so a re-run uses the same lists.
    """
    stations_dir = data_dir / "raw" / "stations"
    if source == "synop":
        isd_path = stations_dir / ISD_HISTORY_FILE
        if not download_station_file(stations_cfg["isd_history_url"], isd_path, stations_cfg, is_isd_answer):
            raise RuntimeError(f"Could not download {stations_cfg['isd_history_url']}")
        stations = wmo_stations(isd_path.read_text(encoding="utf-8"))
        return stations[stations["station"].str[:2].isin(stations_cfg["wmo_blocks"])].reset_index(drop=True)

    networks = []
    for network in stations_cfg["iem_networks"]:
        path = stations_dir / f"{network}.geojson"
        url = IEM_NETWORK_URL.format(network=network)
        if download_station_file(url, path, stations_cfg, is_geojson_answer):
            networks.append(iem_stations(path.read_text(encoding="utf-8")))
        else:
            log.error("Skipping network %s: could not download %s", network, url)
    if not networks:
        raise RuntimeError("Could not download any IEM network file")
    return pd.concat(networks, ignore_index=True).drop_duplicates(["station", "source"], ignore_index=True)


def download_station_file(url: str, path: Path, stations_cfg: dict, is_valid: Callable[[str], bool]) -> bool:
    return download(url, path, stations_cfg["iem_pause_s"], stations_cfg["max_tries"], is_valid)


def download_requests(
    source: str, stations: pd.DataFrame, cfg: dict, raw_dir: Path
) -> list[tuple[str, Path]]:
    """(url, raw file) per request: METAR per station and season, SYNOP per block and night."""
    if source == "metar":
        seasons = cfg["time"]["seasons"]
        return [
            (
                metar_url(
                    station,
                    dt.date.fromisoformat(season["start"]) - dt.timedelta(days=1),
                    dt.date.fromisoformat(season["end"]) + dt.timedelta(days=2),
                ),
                raw_dir / f"{station}_{season['name']}.csv",
            )
            for station in stations["station"]
            for season in seasons
        ]

    blocks = sorted(set(stations["station"].str[:2]))
    return [
        (synop_url(block, start, end), raw_dir / f"{block}_{start:%Y%m%d}.txt")
        for block in blocks
        for start, end in night_windows(cfg)
    ]


def download_all(
    requests: list[tuple[str, Path]], pause_s: float, max_tries: int, is_valid: Callable[[str], bool]
) -> int:
    """Download every request that has no raw file yet, logging progress; returns the number that failed."""
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    for done, (url, path) in enumerate(requests, start=1):
        if path.exists():
            counts["cached"] += 1
        elif download(url, path, pause_s, max_tries, is_valid):
            counts["downloaded"] += 1
        else:
            counts["failed"] += 1
        if done % LOG_EVERY == 0 or done == len(requests):
            log.info(
                "%d/%d requests: %d downloaded, %d cached, %d failed",
                done,
                len(requests),
                counts["downloaded"],
                counts["cached"],
                counts["failed"],
            )
    return counts["failed"]


def is_near_scene(times: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    """True for the times within one hour of a (sorted) scene time."""
    after = np.searchsorted(scenes, times).clip(max=len(scenes) - 1)
    before = (after - 1).clip(min=0)
    gap = np.minimum(np.abs(times - scenes[before]), np.abs(times - scenes[after]))
    return gap <= SCENE_MARGIN


def collect_observations(source: str, cfg: dict, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """Download (resumable), decode and filter the observations of one source for the stations on the grid.

    Only observations within one hour of a scene time are kept. The result is the observation table of
    `validation.station_cases`: the decoded columns plus `lat`, `lon` and `elevation_m` of the station.
    """
    stations_cfg = cfg["stations"]
    data_dir = Path(stations_cfg["data_dir"])
    candidates = candidate_stations(source, stations_cfg, data_dir)
    selected = select_stations(
        candidates, lon, lat, stations_cfg["max_distance_km"], stations_cfg["max_elevation_m"]
    )
    log.info("%s: %d of %d stations lie on the grid", source, len(selected), len(candidates))

    requests = download_requests(source, selected, cfg, data_dir / "raw" / source)
    pause_s = stations_cfg[PAUSE_KEYS[source]]
    failed = download_all(requests, pause_s, stations_cfg["max_tries"], ANSWER_CHECKS[source])
    if failed:
        log.warning("%s: %d of %d downloads failed; run the command again", source, failed, len(requests))

    parse = PARSERS[source]
    tables = [parse(path.read_text(encoding="utf-8")) for _, path in requests if path.exists()]
    if not tables:
        raise RuntimeError(f"No {source} observations: all {len(requests)} downloads failed")

    skipped_lines = sum(table.attrs["skipped_lines"] for table in tables)
    if skipped_lines:
        log.warning("%s: skipped %d malformed lines in the raw files", source, skipped_lines)
    located = pd.concat(tables, ignore_index=True).merge(selected[STATION_COLUMNS], on=["station", "source"])
    scenes = np.array(scene_times(cfg), dtype="datetime64[ns]")
    kept = located[is_near_scene(located["time"].values, scenes)]
    return kept.sort_values(["station", "time"]).reset_index(drop=True)
