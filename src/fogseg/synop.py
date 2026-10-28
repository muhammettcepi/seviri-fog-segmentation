"""Decode horizontal visibility and cloud ceiling from raw SYNOP (FM-12) messages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import takewhile

CLOUD_BASE_LOWER_M = (0, 50, 100, 200, 300, 600, 1000, 1500, 2000, 2500)

SECTION_ONE_END_GROUPS = {"333", "555"}
SPECIAL_VISIBILITY_M = (0, 50, 200, 500, 1000, 2000, 4000, 10000, 20000, 50000)
MAX_CLEAR_OR_SCATTERED_OKTAS = 5
SKY_OBSCURED = "9"


@dataclass(frozen=True)
class Synop:
    station: str
    visibility_m: float
    ceiling_m: float


def visibility_m(vv: str) -> float:
    """Visibility in m from the two-character VV code (WMO code table 4377); NaN for `//` and 51-55."""
    if vv == "//":
        return math.nan
    code = int(vv)
    if code <= 50:
        return code * 100.0
    if code <= 55:
        return math.nan
    if code <= 80:
        return (code - 50) * 1000.0
    if code <= 88:
        return ((code - 80) * 5 + 30) * 1000.0
    if code == 89:
        return 70000.0
    return float(SPECIAL_VISIBILITY_M[code - 90])


def decode_synop(message: str) -> Synop | None:
    """Station, visibility and ceiling of a SYNOP message; None for NIL reports and non-AAXX messages."""
    tokens = message.replace("=", " ").split()
    if len(tokens) < 5 or tokens[0] != "AAXX":
        return None

    station, visibility_group, cloud_group = tokens[2], tokens[3], tokens[4]
    optional_groups = list(takewhile(lambda group: group not in SECTION_ONE_END_GROUPS, tokens[5:]))
    return Synop(
        station=station,
        visibility_m=visibility_m(visibility_group[3:5]),
        ceiling_m=_ceiling_m(cloud_group, optional_groups, base_class=visibility_group[2]),
    )


def _ceiling_m(cloud_group: str, optional_groups: list[str], base_class: str) -> float:
    """Lower edge of the `h` base class when more than 5 oktas are reported; infinite when 5 or fewer.

    The cover is `Nh` of the first `8NhCLCMCH` group, else `N` of the `Nddff` group. Skipping the optional
    `00fff` group after a wind of 99 kt or more needs no special case because the search is by leading `8`.
    """
    total_cover = cloud_group[0]
    if total_cover == SKY_OBSCURED:
        return 0.0

    low_cloud_group = next((group for group in optional_groups if group.startswith("8")), None)
    cover = low_cloud_group[1] if low_cloud_group else total_cover
    if cover == "/":
        return math.nan
    if int(cover) <= MAX_CLEAR_OR_SCATTERED_OKTAS:
        return math.inf
    if base_class == "/":
        return math.nan
    return float(CLOUD_BASE_LOWER_M[int(base_class)])
