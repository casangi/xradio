import pytest

import pyasdm


@pytest.mark.parametrize(
    "input_slice, default_min, default_max, expected_result",
    [
        (slice(None, None), 0, 33, (0, 33)),
        (slice(0, None), 0, 33, (0, 33)),
        (slice(10, None), 0, 133, (10, 133)),
        (slice(None, 20), 0, 233, (0, 20)),
        (slice(None, 34), 0, 333, (0, 34)),
        (slice(30, 40), 0, 40, (30, 40)),
        (0, 0, 33, (0, 1)),
        (3, 0, 4, (3, 4)),
        (4, 3, 1112, (4, 5)),
        (None, 0, 33, (0, 33)),
        (None, 10, 12, (10, 12)),
    ],
)
def test_min_max_from_dimension_slice(
    input_slice, default_min, default_max, expected_result
):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        min_max_from_dimension_slice,
    )

    result = min_max_from_dimension_slice(input_slice, default_min, default_max)
    assert result == expected_result


time_indices_by_bdf_simple_3 = {
    "bdf_names": ["a", "b", "c"],
    "bdf_start": [0, 10, 15, 25],
}


@pytest.mark.parametrize(
    "input_time_indices_by_bdf, input_time_slice, expected_bdfs, expected_bdf_time_slices",
    [
        (
            time_indices_by_bdf_simple_3,
            None,
            time_indices_by_bdf_simple_3["bdf_names"],
            [slice(None, None)] * len(time_indices_by_bdf_simple_3["bdf_names"]),
        ),
        # single BDF
        (time_indices_by_bdf_simple_3, 0, ["a"], [slice(0, 1)]),
        (time_indices_by_bdf_simple_3, slice(0, 1), ["a"], [slice(0, 1)]),
        (time_indices_by_bdf_simple_3, slice(None, 1), ["a"], [slice(None, 1)]),
        (time_indices_by_bdf_simple_3, slice(2, 8), ["a"], [slice(2, 8)]),
        (time_indices_by_bdf_simple_3, slice(0, 7), ["a"], [slice(0, 7)]),
        (time_indices_by_bdf_simple_3, 4, ["a"], [slice(4, 5)]),
        (time_indices_by_bdf_simple_3, 9, ["a"], [slice(9, 10)]),
        (time_indices_by_bdf_simple_3, 10, ["b"], [slice(0, 1)]),
        (time_indices_by_bdf_simple_3, 14, ["b"], [slice(4, 5)]),
        (time_indices_by_bdf_simple_3, slice(10, 11), ["b"], [slice(0, 1)]),
        (time_indices_by_bdf_simple_3, slice(10, 15), ["b"], [slice(0, 5)]),
        (
            time_indices_by_bdf_simple_3,
            slice(15, None),
            ["c"],
            [slice(0, None)],
        ),
        (time_indices_by_bdf_simple_3, 19, ["c"], [slice(4, 5)]),
        (time_indices_by_bdf_simple_3, 20, ["c"], [slice(5, 6)]),
        (time_indices_by_bdf_simple_3, 21, ["c"], [slice(6, 7)]),
        (time_indices_by_bdf_simple_3, slice(19, None), ["c"], [slice(4, None)]),
        (time_indices_by_bdf_simple_3, slice(15, 16), ["c"], [slice(0, 1)]),
        (time_indices_by_bdf_simple_3, slice(15, 20), ["c"], [slice(0, 5)]),
        (time_indices_by_bdf_simple_3, slice(15, 25), ["c"], [slice(0, 10)]),
        # Multiple BDFs
        (
            time_indices_by_bdf_simple_3,
            slice(10, None),
            ["b", "c"],
            [slice(0, None), slice(None, None)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(14, None),
            ["b", "c"],
            [slice(4, None), slice(None, None)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(0, None),
            ["a", "b", "c"],
            [slice(0, None), slice(None, None), slice(None, None)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(None, None),
            ["a", "b", "c"],
            [slice(None, None), slice(None, None), slice(None, None)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(0, 25),
            ["a", "b", "c"],
            [slice(0, None), slice(None, None), slice(None, 10)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(2, 23),
            ["a", "b", "c"],
            [slice(2, None), slice(None, None), slice(None, 8)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(None, 25),
            ["a", "b", "c"],
            [slice(None, None), slice(None, None), slice(None, 10)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(0, 15),
            ["a", "b"],
            [slice(0, None), slice(None, 5)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(9, 14),
            ["a", "b"],
            [slice(9, None), slice(None, 4)],
        ),
        (
            time_indices_by_bdf_simple_3,
            slice(None, 15),
            ["a", "b"],
            [slice(None, None), slice(None, 5)],
        ),
        # out-of-range:
        (time_indices_by_bdf_simple_3, 28, [], [slice(3, 4)]),
        (time_indices_by_bdf_simple_3, 25, [], [slice(0, 1)]),
    ],
)
def test_find_bdfs_and_indices_in_selected_times(
    input_time_indices_by_bdf, input_time_slice, expected_bdfs, expected_bdf_time_slices
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_bdfs_and_indices_in_selected_times,
    )

    bdfs_in_selected_times, bdf_time_slices = find_bdfs_and_indices_in_selected_times(
        input_time_indices_by_bdf, input_time_slice
    )
    assert bdfs_in_selected_times == expected_bdfs
    assert bdf_time_slices == expected_bdf_time_slices


def test_calc_auto_cross_baseline_slices_wrong_type():
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        calc_auto_cross_baseline_slices,
    )

    with pytest.raises(RuntimeError, match="Unexpected type"):
        _result = calc_auto_cross_baseline_slices(3.4, 10, 5, True)


def test_calc_auto_cross_baseline_slices_wrong_value():
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        calc_auto_cross_baseline_slices,
    )

    with pytest.raises(RuntimeError, match="Unexpected value"):
        _result = calc_auto_cross_baseline_slices(3333, 10, 5, True)


@pytest.mark.parametrize(
    "input_slice, cross_baseline_len, nantennas, cross_data_present, expected_result",
    [
        (slice(None, None), 21, 7, True, (slice(None, None), slice(None, None))),
        (slice(None, None), 21, 7, False, (None, slice(None, None))),
        (slice(0, None), 21, 7, True, (slice(0, None), slice(0, None))),
        (slice(0, None), 21, 7, False, (None, slice(0, None))),
        (slice(10, None), 21, 7, True, (slice(10, None), slice(0, None))),
        (slice(10, None), 21, 7, False, (None, slice(10, None))),
        (slice(None, 20), 21, 7, True, (slice(None, 20), None)),
        (slice(None, 20), 21, 7, False, (None, slice(None, 20))),
        (slice(None, 34), 21, 7, True, (slice(None, 21), slice(None, 13))),
        (slice(None, 34), 21, 7, False, (None, slice(None, 34))),
        (slice(30, 40), 21, 7, True, (None, slice(9, 19))),
        (slice(30, 40), 21, 7, False, (None, slice(30, 40))),
        (0, 21, 7, True, (0, None)),
        (0, 21, 7, False, (None, 0)),
        (3, 21, 7, True, (3, None)),
        (3, 21, 7, False, (None, 3)),
        (24, 21, 7, True, (None, 3)),
        (24, 21, 7, False, (None, 24)),
    ],
)
def test_calc_auto_cross_baseline_slices(
    input_slice, cross_baseline_len, nantennas, cross_data_present, expected_result
):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        calc_auto_cross_baseline_slices,
    )

    result = calc_auto_cross_baseline_slices(
        input_slice, cross_baseline_len, nantennas, cross_data_present
    )
    assert result == expected_result


dummy_bdf_descr_auto = {
    "correlation_mode": pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
    "num_antenna": 11,
}
dummy_bdf_descr_cross = {
    "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
    "num_antenna": 3,
}


@pytest.mark.parametrize(
    "input_slice, input_bdf_descr, expected_result",
    [
        (None, None, ["autoData", "crossData"]),
        ((None, slice(None, None), None, None), dummy_bdf_descr_auto, ["autoData"]),
        ((None, slice(0, 3), None, None), dummy_bdf_descr_auto, ["autoData"]),
        ((None, 2, None, None), dummy_bdf_descr_auto, ["autoData"]),
        ((None, slice(5, 13), None, None), dummy_bdf_descr_auto, ["autoData"]),
        (
            (None, slice(None, None), None, None),
            dummy_bdf_descr_cross,
            ["autoData", "crossData"],
        ),
        (
            (None, slice(0, 2), None, None),
            dummy_bdf_descr_cross,
            ["crossData"],
        ),
        (
            (None, slice(0, 3), None, None),
            dummy_bdf_descr_cross,
            ["autoData", "crossData"],
        ),
        (
            (None, 1, None, None),
            dummy_bdf_descr_cross,
            ["crossData"],
        ),
        (
            (None, 5, None, None),
            dummy_bdf_descr_cross,
            ["autoData"],
        ),
    ],
)
def test_find_data_components_needed(input_slice, input_bdf_descr, expected_result):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        find_data_components_needed,
    )

    result = find_data_components_needed(input_slice, input_bdf_descr)
    assert result == expected_result
