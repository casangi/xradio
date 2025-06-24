import datetime
import numpy as np
import pytest

from xradio.measurement_set._utils._msv2._tables.read import (
    CASACORE_TO_PD_TIME_CORRECTION,
)

time_start = (
    datetime.datetime(2025, 5, 1, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    + CASACORE_TO_PD_TIME_CORRECTION
)


@pytest.mark.parametrize(
    "where, expected_output",
    [
        (
            "",
            (np.arange(1200) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID = 0 AND SCAN_NUMBER = 1 AND STATE_ID = 0",
            (np.arange(300) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID IN [0,1] AND SCAN_NUMBER = 1 AND STATE_ID = 0",
            (np.arange(600) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID IN [0,1,2] AND SCAN_NUMBER = 1 AND STATE_ID = 0",
            (np.arange(900) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID IN [0,1,2,3] AND SCAN_NUMBER = 1 AND STATE_ID = 0",
            (np.arange(1200) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID IN [0,1,2,3,4] AND SCAN_NUMBER = 1 AND STATE_ID = 0",
            (np.arange(1200) + time_start, 0.25),
        ),
        (
            "where DATA_DESC_ID = 0 AND SCAN_NUMBER = 1 AND STATE_ID = 1",
            ([0], 1e-5),
        ),
        (
            "where DATA_DESC_ID IN [0,1] AND SCAN_NUMBER = 1 AND STATE_ID = 1",
            ([0], 1e-5),
        ),
    ],
)
def test_get_utimes_tol(ms_minimal_required, where, expected_output):
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        get_utimes_tol,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    with open_table_ro(ms_minimal_required.fname) as mtable:
        utimes, tol = get_utimes_tol(mtable, where)
        assert all(utimes == expected_output[0])
        assert tol == expected_output[1]


baseline_set_5 = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
    ]
)


@pytest.mark.parametrize(
    "input_msv2_name, expected_output",
    [
        ("ms_empty_required", np.empty(shape=(0, 2))),
        ("ms_minimal_required", baseline_set_5),
        ("ms_minimal_without_opt", baseline_set_5),
    ],
)
def test_get_baselines(input_msv2_name, expected_output, request):
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        get_baselines,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    fixture = request.getfixturevalue(input_msv2_name)
    input_path = fixture.fname

    with open_table_ro(input_path) as mtable:
        baselines = get_baselines(mtable)
        assert np.array_equal(baselines, expected_output)


ms_custom_description = {
    "nrows_per_ddi": 100,
    "nchans": 4,
    "npols": 1,
    "data_cols": ["DATA"],
    "SPECTRAL_WINDOW": {"0": 0},
    "POLARIZATION": {"0": 0},
    "ANTENNA": {"0": 0, "1": 1},
    "FIELD": {"0": 0},
    "SCAN": {"1": {"0": {"intent": "intent#subintent"}}},
    "STATE": {"0": {"id": 0, "intent": None}},
    "OBSERVATION": {"0": 0},
    "FEED": {"0": 0},
    "PROCESSOR": {"0": 0},
    "SOURCE": {},
}


@pytest.mark.parametrize("ms_custom_spec", [ms_custom_description], indirect=True)
def test_get_baselines_custom(ms_custom_spec, request):
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        get_baselines,
    )
    from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro

    with open_table_ro(ms_custom_spec.fname) as mtable:
        baselines = get_baselines(mtable)
        assert np.array_equal(baselines, np.array([[0, 1]]))


baseline_set_3 = np.array(
    [
        [0, 1],
        [1, 2],
        [0, 2],
    ]
)
ant1_3 = [1, 0, 0, 1, 0, 1, 1, 1, 0]
ant2_3 = [2, 2, 1, 2, 1, 2, 2, 2, 1]
baselines_pairs_3 = np.column_stack((ant1_3, ant2_3))
expected_indices_3 = [1, 2, 0, 1, 0, 1, 1, 1, 0]

baselines_pairs_3_disordered = np.column_stack(([1, 1, 0, 1, 0], [2, 2, 2, 2, 1]))
expected_indices_3_disordered = [1, 1, 2, 1, 0]

baselines_pairs_3_reversed = np.column_stack(([0, 1, 0, 1, 1], [1, 2, 2, 2, 2]))
expected_indices_3_reversed = [0, 1, 2, 1, 1]

ant1_5 = [0, 0, 0, 0, 1, 1, 1, 2, 2]
ant2_5 = [1, 2, 3, 4, 2, 3, 4, 3, 4]
baselines_pairs_5 = np.column_stack((ant1_5, ant2_5))
expected_indices_5 = [0, 1, 2, 3, 4, 5, 6, 7, 8]

baselines_pairs_5_disordered = np.column_stack(([2, 0, 3, 1, 0], [3, 2, 4, 2, 4]))
expected_indices_5_disordered = [7, 1, 9, 4, 3]

baselines_pairs_5_reversed = np.column_stack(([0, 1, 3, 0, 2], [4, 2, 4, 2, 3]))
expected_indices_5_reversed = list(reversed(expected_indices_5_disordered))


@pytest.mark.parametrize(
    "unique_baselines, baselines, expected_output",
    [
        (baseline_set_5, baselines_pairs_5, expected_indices_5),
        (baseline_set_5, baselines_pairs_5_disordered, expected_indices_5_disordered),
        (baseline_set_5, baselines_pairs_5_reversed, expected_indices_5_reversed),
        (baseline_set_3, baselines_pairs_3, expected_indices_3),
        (baseline_set_3, baselines_pairs_3_disordered, expected_indices_3_disordered),
        (baseline_set_3, baselines_pairs_3_reversed, expected_indices_3_reversed),
    ],
)
def test_get_baseline_indices(unique_baselines, baselines, expected_output):
    from xradio.measurement_set._utils._msv2._tables.read_main_table import (
        get_baseline_indices,
    )

    indices = get_baseline_indices(unique_baselines, baselines)
    assert np.array_equal(indices, expected_output)
