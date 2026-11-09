from contextlib import nullcontext as no_raises

import numpy as np
import pytest


@pytest.mark.parametrize(
    "input_array, input_array_name, input_err_msg, expected_output, expected_error",
    [
        (np.array(2.3), "name", "err", 2.3, no_raises()),
        (np.ones(0), "name", "err", np.ones(0), no_raises()),
        (np.ones(1), "name", "err", 1, no_raises()),
        (np.ones(3) * 2.1, "name", "err", 2.1, no_raises()),
        (
            np.array([1, 1, 1, 2]),
            "name",
            "err",
            None,
            pytest.raises(RuntimeError, match="is not consistent"),
        ),
        (
            np.array([0, 2, 1]),
            "name",
            "err",
            None,
            pytest.raises(RuntimeError, match="is not consistent"),
        ),
    ],
)
def test_check_if_consistent(
    input_array, input_array_name, input_err_msg, expected_output, expected_error
):
    from xradio._utils.list_and_array import check_if_consistent

    with expected_error:
        result = check_if_consistent(input_array, input_array_name, input_err_msg)
        if input_array.ndim == 0:
            assert result == expected_output
        elif input_array.size == 0:
            assert input_array.size == 0
            assert expected_output.size == 0
        else:
            assert isinstance(result, type(input_array[0]))
            assert result == expected_output


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([]), False),
        (np.array([1.0]), False),
        (np.array([1.0, 2.0, 4.0]), False),
        (np.array([4.0, 2.0, 1.0]), True),
    ],
)
def test_is_decreasing(values, expected):
    from xradio._utils.list_and_array import is_decreasing

    assert is_decreasing(values, "message") is expected


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 3.0, 2.0]),
        np.array([3.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 2.0]),
        np.array([2.0, 2.0, 1.0]),
        np.array([2.0, 2.0, 2.0]),
        np.array([1.0, 2.0, np.nan]),
        np.array([2.0, 1.0, np.nan]),
    ],
)
def test_is_decreasing_any_other_order_raises(values):
    from xradio._utils.list_and_array import NonMonotonicAxisError, is_decreasing

    with pytest.raises(NonMonotonicAxisError, match="^the message$"):
        is_decreasing(values, "the message")


def test_non_monotonic_axis_error_is_a_value_error():
    from xradio._utils.list_and_array import NonMonotonicAxisError

    assert issubclass(NonMonotonicAxisError, ValueError)
