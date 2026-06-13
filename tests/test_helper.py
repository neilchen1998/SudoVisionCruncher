from hypothesis import given, strategies as st
from typing import Callable

def find_max_value_satisfying(low: int, high: int, condition: Callable[[int], bool]) -> int:
    """
    Finds the maximum value that satisfied the given condition.

    Args:
        low: The lower bound of the search range (inclusive).
        high: The uppoer bound of the search range (inclusive).
        condition: A lambda function that takes an integer and returns a boolean.

    Returns:
        int: The maximum value in [low, high] that satisfies the given condition.
    """

    best = low
    while low <= high:
        mid = low + (high - low) // 2

        if condition(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best

MIN_VALUE = 10
MAX_VALUE = 300

@given(st.integers(min_value=MIN_VALUE, max_value=MAX_VALUE))
def test_find_max_value_satisfying(threshold):

    def condition(x):
        return x <= threshold   # this condition is a simplier version of that from scale_font_to_target()

    ret = find_max_value_satisfying(MIN_VALUE, MAX_VALUE, condition)

    # The return value is either:
    # (i) the largest value within the threshold, or
    # (ii) the given max value
    assert ret == min(threshold, MAX_VALUE)
