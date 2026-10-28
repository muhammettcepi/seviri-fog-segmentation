import math

import pytest

from fogseg.synop import decode_synop, visibility_m


@pytest.mark.parametrize(
    ("code", "expected_m"),
    [
        ("00", 0),
        ("09", 900),
        ("10", 1000),
        ("50", 5000),
        ("56", 6000),
        ("80", 30000),
        ("81", 35000),
        ("89", 70000),
        ("90", 0),
        ("94", 1000),
        ("97", 10000),
        ("52", math.nan),
        ("//", math.nan),
    ],
)
def test_visibility_code_table(code, expected_m):
    assert visibility_m(code) == pytest.approx(expected_m, nan_ok=True)


def test_missing_cover_gives_unknown_ceiling():
    synop = decode_synop("AAXX 28001 10004 46/60 /2209 10062 20046 30317 40317 333 55300==")
    assert synop.station == "10004"
    assert synop.visibility_m == 10000
    assert math.isnan(synop.ceiling_m)


def test_low_overcast_from_8_group():
    synop = decode_synop("AAXX 28031 10870 11408 82504 10012 20008 39950 40330 57010 70500 87500 333 55300")
    assert synop.visibility_m == 800
    assert synop.ceiling_m == 300


def test_cover_from_n_without_8_group():
    synop = decode_synop("AAXX 28031 10870 11560 32504 10012 20008==")
    assert synop.ceiling_m == math.inf


def test_sky_obscured():
    synop = decode_synop("AAXX 28031 10870 11/02 92502 10012 20008 70545==")
    assert synop.visibility_m == 200
    assert synop.ceiling_m == 0


def test_unknown_base_with_overcast_is_nan():
    synop = decode_synop("AAXX 28031 10870 11/60 82504 10012 87500==")
    assert math.isnan(synop.ceiling_m)


def test_high_wind_group_is_skipped():
    synop = decode_synop("AAXX 28031 10870 11408 82599 00105 10012 20008 87500==")
    assert synop.ceiling_m == 300


def test_groups_after_section_one_are_ignored():
    synop = decode_synop("AAXX 28031 10870 11560 32504 10012 20008 333 87500==")
    assert synop.ceiling_m == math.inf


def test_nil_message():
    assert decode_synop("AAXX 28031 10870 NIL==") is None


def test_message_that_is_not_synop():
    assert decode_synop("BBXX 28031 10870 11560 32504==") is None
