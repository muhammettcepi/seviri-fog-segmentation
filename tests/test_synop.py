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


def test_cloud_group_found_after_high_wind_group():
    synop = decode_synop("AAXX 28031 10870 11408 82599 00105 10012 20008 87500==")
    assert synop.ceiling_m == 300


def test_groups_after_section_one_are_ignored():
    synop = decode_synop("AAXX 28031 10870 11560 32504 10012 20008 333 87500==")
    assert synop.ceiling_m == math.inf


def test_overcast_n_with_low_cover_in_8_group_is_no_ceiling():
    synop = decode_synop("AAXX 28031 10870 11408 82504 10012 83500==")
    assert synop.ceiling_m == math.inf


def test_unknown_n_uses_cover_of_8_group():
    synop = decode_synop("AAXX 28031 10870 11408 /2504 10012 87500==")
    assert synop.ceiling_m == 300


def test_unknown_base_with_low_cover_is_no_ceiling():
    synop = decode_synop("AAXX 28031 10870 11/60 32504==")
    assert synop.ceiling_m == math.inf


def test_obscured_sky_in_8_group():
    synop = decode_synop("AAXX 28031 10870 11408 82504 10012 89500==")
    assert synop.ceiling_m == 0


@pytest.mark.parametrize("broken_group", ["8", "83", "835000", "8350/x"])
def test_broken_8_group_is_ignored(broken_group):
    synop = decode_synop(f"AAXX 28031 10870 11408 82504 10012 {broken_group}==")
    assert synop.ceiling_m == 300


def test_non_numeric_visibility_code_is_nan():
    synop = decode_synop("AAXX 28031 10870 1141/ 82504==")
    assert math.isnan(synop.visibility_m)
    assert synop.ceiling_m == 300


@pytest.mark.parametrize(
    "message",
    [
        "",
        "AAXX 28031 10870",
        "AAXX 28031 10870 11408",
        "AAXX 28031 10870 114 82504",
        "AAXX 28031 10870 114080 82504",
        "AAXX 28031 10870 11x08 82504",
        "AAXX 28031 10870 11408 8",
        "AAXX 28031 10870 11408 8x504",
        "AAXX 28031 10870 NIL 333 55300",
    ],
)
def test_malformed_message_is_undecodable(message):
    assert decode_synop(message) is None


def test_nil_message():
    assert decode_synop("AAXX 28031 10870 NIL==") is None


def test_message_that_is_not_synop():
    assert decode_synop("BBXX 28031 10870 11560 32504==") is None
