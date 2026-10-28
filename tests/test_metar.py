import math

import pytest

from fogseg.metar import ceiling_m, prevailing_visibility_m


def test_visibility_from_main_body():
    metar = "EDDM 280450Z AUTO 05002KT 7000 1300 R08L/P2000N BR NCD M04/M04 Q1034 NOSIG"
    assert prevailing_visibility_m(metar) == 7000


def test_visibility_ignores_rvr_and_trend():
    metar = "LOWW 010050Z 00000KT 0400 R29/1200N FG VV002 01/01 Q1030 TEMPO 2000 BR"
    assert prevailing_visibility_m(metar) == 400


def test_minimum_visibility_with_direction_does_not_replace_prevailing():
    metar = "EDDM 280450Z 05002KT 4000 2800NE BR NCD M04/M04 Q1034"
    assert prevailing_visibility_m(metar) == 4000


def test_cavok_and_9999_mean_10_km():
    cavok = "EDDF 010050Z 22005KT CAVOK 05/02 Q1020"
    assert prevailing_visibility_m(cavok) == 10000
    assert ceiling_m(cavok) == math.inf

    unlimited = "EDDF 010050Z 22005KT 9999 FEW030 05/02 Q1020"
    assert prevailing_visibility_m(unlimited) == 10000


def test_missing_visibility_is_nan():
    metar = "EDDM 010050Z AUTO 22005KT //// NCD 05/02 Q1020"
    assert math.isnan(prevailing_visibility_m(metar))


def test_ceiling_is_lowest_broken_layer():
    metar = "EDDF 010050Z 22005KT 9999 FEW005 SCT008 BKN012 OVC030 05/02 Q1020"
    assert ceiling_m(metar) == pytest.approx(12 * 100 * 0.3048)


def test_vertical_visibility_is_a_ceiling():
    metar = "LOWW 010050Z 00000KT 0100 FG VV001 01/01 Q1030"
    assert ceiling_m(metar) == pytest.approx(30.48)


def test_few_and_scattered_are_no_ceiling():
    metar = "EDDF 010050Z 22005KT 9999 FEW005 SCT008 05/02 Q1020"
    assert ceiling_m(metar) == math.inf


@pytest.mark.parametrize("no_cloud_code", ["NSC", "NCD", "SKC", "CLR"])
def test_no_cloud_codes_are_no_ceiling(no_cloud_code):
    metar = f"EDDF 010050Z 22005KT 9999 {no_cloud_code} 05/02 Q1020"
    assert ceiling_m(metar) == math.inf


def test_unknown_layer_height_is_nan():
    broken = "EDDF 010050Z 22005KT 9999 BKN/// 05/02 Q1020"
    assert math.isnan(ceiling_m(broken))

    unknown_cumulonimbus = "EDDF 010050Z 22005KT 9999 //////CB 05/02 Q1020"
    assert math.isnan(ceiling_m(unknown_cumulonimbus))


def test_missing_cloud_information_is_nan():
    metar = "EDDM 010050Z AUTO 22005KT 9999 05/02 Q1020"
    assert math.isnan(ceiling_m(metar))


def test_trend_layers_are_ignored():
    metar = "EDDF 010050Z 22005KT 9999 SCT040 05/02 Q1020 BECMG BKN005"
    assert ceiling_m(metar) == math.inf


def test_remark_layers_are_ignored():
    metar = "KJFK 010051Z 22005KT 9999 FEW030 05/02 Q1020 RMK BKN005"
    assert ceiling_m(metar) == math.inf


def test_correction_prefix():
    metar = "COR EDDM 280650Z AUTO VRB01KT 4400 BR NCD M04/M05 Q1035"
    assert prevailing_visibility_m(metar) == 4400
