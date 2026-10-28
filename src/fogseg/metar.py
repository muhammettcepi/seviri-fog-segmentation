"""Decode prevailing visibility and cloud ceiling from raw METAR messages."""

from __future__ import annotations

import math
import re

FEET_TO_METRES = 0.3048
UNLIMITED_VISIBILITY_M = 10000.0

TIME_GROUP = re.compile(r"\d{6}Z")
TREND_OR_REMARK_TOKENS = {"TEMPO", "BECMG", "NOSIG", "RMK"}
NO_CLOUD_TOKENS = {"NSC", "NCD", "SKC", "CLR", "CAVOK"}

VISIBILITY_GROUP = re.compile(r"\d{4}(NDV)?")
CLOUD_LAYER = re.compile(r"(?P<amount>FEW|SCT|BKN|OVC)(?P<height>\d{3}|///)(CB|TCU|///)?")
VERTICAL_VISIBILITY = re.compile(r"(?P<amount>VV)(?P<height>\d{3}|///)")
UNKNOWN_LAYER = re.compile(r"/{6}(CB|TCU|///)?")
CEILING_AMOUNTS = {"BKN", "OVC", "VV"}


def main_body(metar: str) -> list[str]:
    """Tokens between the time group and the first trend or remark token; empty without a time group."""
    tokens = metar.split()
    time_group_index = next((i for i, token in enumerate(tokens) if TIME_GROUP.fullmatch(token)), None)
    if time_group_index is None:
        return []

    body = []
    for token in tokens[time_group_index + 1 :]:
        if token in TREND_OR_REMARK_TOKENS:
            break
        body.append(token)
    return body


def prevailing_visibility_m(metar: str) -> float:
    """Visibility in m from the first visibility group of the main body; NaN when it is missing or `////`."""
    for token in main_body(metar):
        if token == "CAVOK":
            return UNLIMITED_VISIBILITY_M
        if token == "////":
            return math.nan
        if VISIBILITY_GROUP.fullmatch(token):
            visibility = float(token[:4])
            return UNLIMITED_VISIBILITY_M if visibility == 9999 else visibility
    return math.nan


def ceiling_m(metar: str) -> float:
    """Base in m of the first (lowest) BKN, OVC or VV layer of the main body.

    Infinite when the body reports cloud but no such layer. NaN when the base is `///`, when a layer of
    unknown amount (`//////`) is reported and no known BKN, OVC or VV layer comes first, or when the body
    has no cloud information.
    """
    has_cloud_information = False
    has_unknown_layer = False
    for token in main_body(metar):
        if token in NO_CLOUD_TOKENS:
            has_cloud_information = True
            continue
        if UNKNOWN_LAYER.fullmatch(token):
            has_unknown_layer = True
            continue
        layer = CLOUD_LAYER.fullmatch(token) or VERTICAL_VISIBILITY.fullmatch(token)
        if layer is None:
            continue
        has_cloud_information = True
        if layer["amount"] in CEILING_AMOUNTS:
            return _base_m(layer["height"])
    if has_unknown_layer:
        return math.nan
    return math.inf if has_cloud_information else math.nan


def _base_m(height_hundreds_of_feet: str) -> float:
    if height_hundreds_of_feet == "///":
        return math.nan
    return int(height_hundreds_of_feet) * 100 * FEET_TO_METRES
