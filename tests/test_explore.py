"""Tests for the helpers of the BTD exploration notebook."""

import numpy as np
import pytest

from fogseg.explore import satellite_zenith_angle


def test_satellite_zenith_is_zero_below_the_satellite():
    assert satellite_zenith_angle(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_satellite_zenith_at_50_north_on_the_meridian():
    # geostationary at 0°E, Earth radius 6378.169 km, orbit radius 42164.0 km
    assert satellite_zenith_angle(np.array([0.0]), np.array([50.0]))[0] == pytest.approx(57.3, abs=0.2)
