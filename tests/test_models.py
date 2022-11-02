"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected_mean",
    [
        ([[0, 0], [0, 0], [0, 0]], [0,0]),
        ([[1, 2], [3, 4], [5, 6]], [3,4]),
    ]
)
def test_daily_mean(test, expected_mean):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected_mean)

@pytest.mark.parametrize(
    "test, expected_max",
    [
        ([[0, 0], [0, 0], [0, 0]], [0,0]),
        ([[1, 2], [3, 4], [5, 6]], [5,6]),
    ]
)
def test_daily_max_integers(test, expected_max):
    
    from inflammation.models import daily_max

    npt.assert_array_equal(daily_max(test), expected_max)

@pytest.mark.parametrize(
    "test, expected_min",
    [
        ([[0, 0], [0, 0], [0, 0]], [0,0]),
        ([[1, 2], [3, 4], [5, 6]], [1,2]),
    ]
)
def test_daily_min(test, expected_min):
    """test integer passing for daily min function
    """
    from inflammation.models import daily_min
    
    npt.assert_array_equal(daily_min(test), expected_min)

def test_daily_min_string():
    """test for type error when pass a string to daily min func.
    """
    from inflammation.models import daily_min
    
    with pytest.raises(TypeError):
        error_expected = daily_min([['abd', 'ads'], ['afs', 'dfs']])