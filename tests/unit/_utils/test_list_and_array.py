from contextlib import nullcontext as no_raises

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize(
    "input_array, input_array_name, input_err_msg, expected_output, expected_error",
    [
        (np.array(2.3), "name", "err", 2.3, no_raises()),
        (np.ones(1), "name", "err", 1, no_raises()),
        (np.ones(3) * 2.1, "name", "err", 2.1, no_raises()),
        (np.array([7, 7, 7], dtype=np.int32), "name", "", 7, no_raises()),
        (np.array(["a", "a"]), "name", "", "a", no_raises()),
        (
            np.ones(0),
            "name",
            "err",
            None,
            pytest.raises(RuntimeError, match=r"name is not consistent: no values"),
        ),
        (
            np.array([], dtype=np.int64),
            "OBSERVATION_ID",
            "",
            None,
            pytest.raises(
                RuntimeError, match=r"OBSERVATION_ID is not consistent.*empty"
            ),
        ),
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
            assert np.ndim(result) == 0
        else:
            assert isinstance(result, type(input_array[0]))
            assert result == expected_output


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(np.asarray, id="ndarray"),
        pytest.param(pd.Series, id="Series"),
        pytest.param(xr.DataArray, id="DataArray"),
    ],
)
def test_check_if_consistent_returns_scalar_for_all_container_types(container):
    """All callers use the result as a scalar (TaQL ids, comparisons)."""
    from xradio._utils.list_and_array import check_if_consistent

    result = check_if_consistent(container(np.array([5, 5, 5])), "ID")

    assert np.ndim(result) == 0
    assert result == 5
    assert not isinstance(result, np.ndarray | pd.Series | xr.DataArray)


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(np.asarray, id="ndarray"),
        pytest.param(lambda arr: pd.Series(arr, dtype=np.int64), id="Series"),
        pytest.param(xr.DataArray, id="DataArray"),
    ],
)
def test_check_if_consistent_empty_input_raises(container):
    """Empty input must raise a clear error, never return the empty container."""
    from xradio._utils.list_and_array import check_if_consistent

    with pytest.raises(RuntimeError) as exc_info:
        check_if_consistent(
            container(np.array([], dtype=np.int64)),
            "Main/configDescriptionId",
            "selection: fieldId in [3]",
        )

    message = str(exc_info.value)
    assert message.startswith("Main/configDescriptionId is not consistent")
    assert "empty selection" in message
    assert "(selection: fieldId in [3])" in message


def test_check_if_consistent_error_lists_distinct_values_and_context():
    from xradio._utils.list_and_array import check_if_consistent

    with pytest.raises(RuntimeError) as exc_info:
        check_if_consistent(np.array([3, 1, 3, 1, 1]), "PROCESSOR_ID", "ddi=2")

    message = str(exc_info.value)
    assert message.startswith("PROCESSOR_ID is not consistent")
    assert "2 distinct values in 5 entries" in message
    assert "(ddi=2)" in message
    assert "[1 3]" in message
    # no stray debug formatting of the err_msg argument
    assert "err_msg=" not in message


def test_check_if_consistent_error_without_context():
    from xradio._utils.list_and_array import check_if_consistent

    with pytest.raises(RuntimeError) as exc_info:
        check_if_consistent(np.array([0, 1]), "FIELD_ID")

    assert "()" not in str(exc_info.value)


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (np.array([3, 1, 2, 3, 1]), np.array([1, 2, 3])),
        (np.array(4), np.array([4])),
        (xr.DataArray(np.array([2.5, 2.5, -1.0])), np.array([-1.0, 2.5])),
        (np.array(["b", "a", "b"]), np.array(["a", "b"])),
    ],
)
def test_unique_1d(input_array, expected):
    from xradio._utils.list_and_array import unique_1d

    result = unique_1d(input_array)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)
