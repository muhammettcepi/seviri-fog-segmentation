import numpy as np
import pytest

from fogseg.parallax import apparent_position


def test_shift_points_away_from_the_subsatellite_point():
    lon, lat = apparent_position(np.array([0.0]), np.array([50.0]), np.array([1000.0]))
    assert lat[0] > 50.0 and lon[0] == pytest.approx(0.0, abs=1e-9)


def test_shift_size_at_50_north_for_one_kilometre():
    _, lat = apparent_position(np.array([0.0]), np.array([50.0]), np.array([1000.0]))
    shift_km = (lat[0] - 50.0) * np.pi / 180 * 6371
    assert shift_km == pytest.approx(1.6, abs=0.15)


def test_zero_height_is_no_shift():
    lon, lat = apparent_position(np.array([10.0]), np.array([45.0]), np.array([0.0]))
    assert (lon[0], lat[0]) == pytest.approx((10.0, 45.0))
