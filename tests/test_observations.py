"""Tests for the observation downloads and parsers; the network is replaced by `fetch_text` stubs."""

import datetime as dt
import http.client
import json
import math

import numpy as np
import pandas as pd
import pytest

from fogseg import observations
from fogseg.observations import (
    collect_observations,
    download,
    is_near_scene,
    night_windows,
    parse_metar_csv,
    parse_synop_text,
    wmo_stations,
)

VALID_ANSWER = "station,valid,metar\n"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def stub_fetch(monkeypatch, answers: list[str]) -> list[str]:
    """Answer successive requests with `answers` (the last one repeats) and return the requested URLs."""
    urls = []

    def fetch_text(url: str, timeout: float = 120) -> str:
        urls.append(url)
        return answers[min(len(urls), len(answers)) - 1]

    monkeypatch.setattr(observations, "fetch_text", fetch_text)
    return urls


def one_night_config(tmp_path) -> dict:
    return {
        "time": {
            "seasons": [{"name": "test", "start": "2024-01-01", "end": "2024-01-02"}],
            "slots": ["21:00", "00:00"],
            "every_nth_night": 2,
        },
        "stations": {
            "data_dir": str(tmp_path / "stations"),
            "iem_networks": ["DE__ASOS"],
            "wmo_blocks": ["10"],
            "isd_history_url": "https://example.test/isd-history.csv",
            "iem_pause_s": 1,
            "ogimet_pause_s": 2,
            "max_tries": 2,
            "max_distance_km": 5.0,
            "max_elevation_m": 1000.0,
        },
    }


def small_grid() -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.linspace(11.0, 12.5, 16), np.linspace(47.5, 49.0, 16))


def test_parse_metar_csv():
    text = (
        "station,valid,metar\n"
        "EDDM,2024-01-28 00:20,EDDM 280020Z 24005KT 0800 FG VV002 02/02 Q1020 NOSIG\n"
        "EDDM,2024-01-28 00:50,EDDM 280050Z 24005KT CAVOK 02/02 Q1020\n"
    )

    table = parse_metar_csv(text)

    assert list(table.columns) == ["source", "station", "time", "visibility_m", "ceiling_m"]
    assert (table["source"] == "metar").all()
    assert np.issubdtype(table["time"].dtype, np.datetime64)
    assert table["time"].tolist() == [pd.Timestamp("2024-01-28 00:20"), pd.Timestamp("2024-01-28 00:50")]
    assert table["visibility_m"].tolist() == [800.0, 10000.0]
    assert table["ceiling_m"].iloc[0] == pytest.approx(60.96)
    assert math.isinf(table["ceiling_m"].iloc[1])


def test_parse_metar_csv_without_rows():
    table = parse_metar_csv(VALID_ANSWER)

    assert table.empty
    assert list(table.columns) == ["source", "station", "time", "visibility_m", "ceiling_m"]


def test_parse_synop_text():
    text = (
        "10381,2024,01,28,00,00,AAXX 28001 10381 41002 82105 10062 20046 333 55300=\n"
        "10005,2024,01,28,00,00,AAXX 28001 10005 NIL==\n"
    )

    table = parse_synop_text(text)

    assert list(table.columns) == ["source", "station", "time", "visibility_m", "ceiling_m"]
    assert len(table) == 1
    row = table.iloc[0]
    assert (row["source"], row["station"]) == ("synop", "10381")
    assert row["time"] == pd.Timestamp("2024-01-28 00:00")
    assert row["visibility_m"] == 200.0
    assert row["ceiling_m"] == 0.0


def test_wmo_stations_takes_latest_record():
    isd = (
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
        '"108700","99999","MUENCHEN OLD","GM","","","+48.100","+011.500","+0500.0","19500101","20100101"\n'
        '"108700","99999","MUENCHEN","GM","","EDDM","+48.348","+011.813","+0448.0","20100102","20240101"\n'
        '"108700","12345","MUENCHEN US","GM","","","+49.000","+012.000","+0100.0","20000101","20250101"\n'
    )

    table = wmo_stations(isd)

    assert list(table.columns) == ["station", "source", "lat", "lon", "elevation_m"]
    assert len(table) == 1
    row = table.iloc[0]
    assert (row["station"], row["source"]) == ("10870", "synop")
    assert (row["lat"], row["lon"], row["elevation_m"]) == (48.348, 11.813, 448.0)


def test_download_skips_existing_file(monkeypatch, tmp_path):
    urls = stub_fetch(monkeypatch, [VALID_ANSWER])
    path = tmp_path / "answer.csv"
    path.write_text("already here")

    assert download("https://example.test", path, 0, 3, lambda text: True)

    assert urls == []
    assert path.read_text() == "already here"


def test_download_does_not_save_empty_answer(monkeypatch, tmp_path):
    urls = stub_fetch(monkeypatch, [""])
    path = tmp_path / "answer.csv"

    assert not download("https://example.test", path, 0, 3, lambda text: bool(text))

    assert len(urls) == 3
    assert list(tmp_path.iterdir()) == []


def test_download_retries_then_saves(monkeypatch, tmp_path):
    urls = stub_fetch(monkeypatch, ["", VALID_ANSWER])
    path = tmp_path / "raw" / "answer.csv"

    assert download("https://example.test", path, 0, 3, lambda text: bool(text))

    assert len(urls) == 2
    assert path.read_text() == VALID_ANSWER


def test_download_pauses_longer_after_an_invalid_answer(monkeypatch, tmp_path):
    stub_fetch(monkeypatch, ["", VALID_ANSWER])
    pauses = []
    monkeypatch.setattr("time.sleep", pauses.append)

    download("https://example.test", tmp_path / "answer.csv", 10, 3, lambda text: bool(text))

    assert pauses == [10, 60, 10]


def test_download_treats_network_errors_as_invalid_answers(monkeypatch, tmp_path):
    def fetch_text(url: str, timeout: float = 120) -> str:
        raise TimeoutError

    monkeypatch.setattr(observations, "fetch_text", fetch_text)

    assert not download("https://example.test", tmp_path / "answer.csv", 0, 2, lambda text: bool(text))


def test_night_windows(tmp_path):
    windows = night_windows(one_night_config(tmp_path))

    assert windows == [(dt.datetime(2024, 1, 1, 20), dt.datetime(2024, 1, 2, 6))]


def test_collect_metar_observations(monkeypatch, tmp_path):
    network = {
        "features": [
            {"id": "EDDM", "geometry": {"coordinates": [11.8, 48.4]}, "properties": {"elevation": 448}},
            {"id": "LOWW", "geometry": {"coordinates": [16.5, 48.1]}, "properties": {"elevation": 190}},
        ]
    }
    csv_text = (
        "station,valid,metar\n"
        "EDDM,2024-01-01 21:00,EDDM 012100Z 24005KT 0800 FG VV002 02/02 Q1020\n"
        "EDDM,2024-01-01 15:00,EDDM 011500Z 24005KT CAVOK 02/02 Q1020\n"
    )
    urls = []

    def fetch_text(url: str, timeout: float = 120) -> str:
        urls.append(url)
        return json.dumps(network) if "geojson" in url else csv_text

    monkeypatch.setattr(observations, "fetch_text", fetch_text)
    lon, lat = small_grid()

    table = collect_observations("metar", one_night_config(tmp_path), lon, lat)

    assert sum("asos.py" in url for url in urls) == 1
    assert (tmp_path / "stations" / "raw" / "metar" / "EDDM_test.csv").exists()
    assert table["station"].tolist() == ["EDDM"]
    row = table.iloc[0]
    assert row["time"] == pd.Timestamp("2024-01-01 21:00")
    assert (row["lat"], row["lon"], row["elevation_m"]) == (48.4, 11.8, 448.0)
    assert row["visibility_m"] == 800.0


def test_collect_synop_observations(monkeypatch, tmp_path):
    isd = (
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
        '"103810","99999","TEST","GM","","","+48.400","+011.800","+0448.0","20100102","20260101"\n'
        '"103820","99999","FAR AWAY","GM","","","+52.400","+013.800","+0048.0","20100102","20260101"\n'
    )
    synop = "10381,2024,01,02,00,00,AAXX 02001 10381 41002 82105 10062 20046=\n"
    urls = []

    def fetch_text(url: str, timeout: float = 120) -> str:
        urls.append(url)
        return isd if "isd-history" in url else synop

    monkeypatch.setattr(observations, "fetch_text", fetch_text)
    lon, lat = small_grid()

    table = collect_observations("synop", one_night_config(tmp_path), lon, lat)

    assert [url for url in urls if "getsynop" in url] == [
        "https://www.ogimet.com/cgi-bin/getsynop?begin=202401012000&end=202401020600&block=10"
    ]
    assert (tmp_path / "stations" / "raw" / "synop" / "10_20240101.txt").exists()
    assert table["station"].tolist() == ["10381"]
    assert table.iloc[0]["time"] == pd.Timestamp("2024-01-02 00:00")
    assert table.iloc[0]["lat"] == 48.4


@pytest.mark.parametrize(
    "error",
    [http.client.IncompleteRead(b"partial"), http.client.RemoteDisconnected("closed"), TimeoutError()],
    ids=["incomplete-read", "remote-disconnected", "timeout"],
)
def test_download_survives_broken_connections(monkeypatch, tmp_path, error):
    calls = []

    def fetch_text(url: str, timeout: float = 120) -> str:
        calls.append(url)
        raise error

    monkeypatch.setattr(observations, "fetch_text", fetch_text)
    path = tmp_path / "answer.csv"

    assert not download("https://example.test", path, 0, 3, lambda text: bool(text))

    assert len(calls) == 3
    assert list(tmp_path.iterdir()) == []


def test_parse_metar_csv_keeps_commas_of_the_metar():
    text = (
        "station,valid,metar\n"
        "EDDM,2024-01-28 00:20,EDDM 280020Z 24005KT, 0800 FG VV002 02/02 Q1020\n"
        "EDDM,not a time,EDDM 280050Z 24005KT CAVOK 02/02 Q1020\n"
        "EDDM\n"
        "EDDM,2024-01-28 01:20,M\n"
    )

    table = parse_metar_csv(text)

    assert table["time"].tolist() == [pd.Timestamp("2024-01-28 00:20")]
    assert table["visibility_m"].tolist() == [800.0]
    assert table.attrs["skipped_lines"] == 2


def test_parse_synop_text_skips_lines_with_a_bad_date():
    text = (
        "10381,2024,13,45,00,00,AAXX 28001 10381 41002 82105 10062=\n"
        "10381,2024,01,28,00,00,AAXX 28001 10381 41002 82105 10062=\n"
        "<html>busy</html>\n"
        "\n"
    )

    table = parse_synop_text(text)

    assert table["time"].tolist() == [pd.Timestamp("2024-01-28 00:00")]
    assert table.attrs["skipped_lines"] == 2


def test_parse_synop_text_skips_a_line_whose_date_is_too_large_to_convert():
    text = (
        "10381,2024,01,99999999999999999999,00,00,AAXX 28001 10381 41002 82105 10062=\n"
        "10381,2024,01,28,00,00,AAXX 28001 10381 41002 82105 10062=\n"
    )

    table = parse_synop_text(text)

    assert table["time"].tolist() == [pd.Timestamp("2024-01-28 00:00")]
    assert table.attrs["skipped_lines"] == 1


def test_is_near_scene_boundaries():
    scenes = np.array(["2024-01-01T21:00", "2024-01-02T00:00"], dtype="datetime64[ns]")
    times = np.array(
        [
            "2024-01-01T20:00",  # exactly one hour before the first scene
            "2024-01-01T19:59",
            "2024-01-01T22:30",  # between the scenes, within reach of none
            "2024-01-02T01:00",  # exactly one hour after the last scene
            "2024-01-02T01:01",
        ],
        dtype="datetime64[ns]",
    )

    assert is_near_scene(times, scenes).tolist() == [True, False, False, True, False]


def geojson_stub(monkeypatch, networks: dict[str, str], csv_text: str) -> list[str]:
    """Serve the GeoJSON of each network (an empty answer for others) and `csv_text` for METAR requests."""
    urls = []

    def fetch_text(url: str, timeout: float = 120) -> str:
        urls.append(url)
        if "asos.py" in url:
            return csv_text
        network = url.rsplit("/", 1)[-1].removesuffix(".geojson")
        return networks.get(network, "")

    monkeypatch.setattr(observations, "fetch_text", fetch_text)
    return urls


EDDM_GEOJSON = json.dumps(
    {
        "features": [
            {"id": "EDDM", "geometry": {"coordinates": [11.8, 48.4]}, "properties": {"elevation": 448}}
        ]
    }
)
EDDM_CSV = "station,valid,metar\nEDDM,2024-01-01 21:00,EDDM 012100Z 24005KT 0800 FG VV002 02/02 Q1020\n"


def test_collect_uses_a_station_listed_in_two_networks_once(monkeypatch, tmp_path):
    cfg = one_night_config(tmp_path)
    cfg["stations"]["iem_networks"] = ["DE__ASOS", "AT__ASOS"]
    urls = geojson_stub(monkeypatch, {"DE__ASOS": EDDM_GEOJSON, "AT__ASOS": EDDM_GEOJSON}, EDDM_CSV)
    lon, lat = small_grid()

    table = collect_observations("metar", cfg, lon, lat)

    assert len(table) == 1
    assert sum("asos.py" in url for url in urls) == 1


def test_collect_skips_a_network_that_cannot_be_downloaded(monkeypatch, tmp_path):
    cfg = one_night_config(tmp_path)
    cfg["stations"]["iem_networks"] = ["DE__ASOS", "XX__ASOS"]
    geojson_stub(monkeypatch, {"DE__ASOS": EDDM_GEOJSON}, EDDM_CSV)
    lon, lat = small_grid()

    table = collect_observations("metar", cfg, lon, lat)

    assert table["station"].tolist() == ["EDDM"]
    assert not (tmp_path / "stations" / "raw" / "stations" / "XX__ASOS.geojson").exists()


def test_second_collect_run_makes_no_requests(monkeypatch, tmp_path):
    cfg = one_night_config(tmp_path)
    urls = geojson_stub(monkeypatch, {"DE__ASOS": EDDM_GEOJSON}, EDDM_CSV)
    lon, lat = small_grid()
    first = collect_observations("metar", cfg, lon, lat)
    urls.clear()

    second = collect_observations("metar", cfg, lon, lat)

    assert urls == []
    pd.testing.assert_frame_equal(first, second)


def test_each_source_waits_its_own_pause(monkeypatch, tmp_path):
    isd = (
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
        '"103810","99999","TEST","GM","","","+48.400","+011.800","+0448.0","20100102","20260101"\n'
    )
    synop = "10381,2024,01,02,00,00,AAXX 02001 10381 41002 82105 10062 20046=\n"
    pauses = []
    monkeypatch.setattr("time.sleep", pauses.append)
    cfg = one_night_config(tmp_path)
    lon, lat = small_grid()

    geojson_stub(monkeypatch, {"DE__ASOS": EDDM_GEOJSON}, EDDM_CSV)
    collect_observations("metar", cfg, lon, lat)
    assert pauses == [1, 1]  # the network file, then the station's season

    pauses.clear()
    monkeypatch.setattr(
        observations, "fetch_text", lambda url, timeout=120: isd if "isd-history" in url else synop
    )
    collect_observations("synop", cfg, lon, lat)
    assert pauses == [1, 2]  # the NOAA station file, then the Ogimet night
