import itertools

import numpy as np
import pytest

time_indices_by_bdf_simple_3 = {
    "bdf_names": ["a", "b", "c"],
    "bdf_start": [0, 10, 15, 25],
}


@pytest.mark.parametrize(
    "input_time_slice, expected_bdfs, expected_bdf_time_slices",
    [
        (None, ["a", "b", "c"], [slice(0, 10, 1), slice(0, 5, 1), slice(0, 10, 1)]),
        # single BDF
        (0, ["a"], [slice(0, 1, 1)]),
        (slice(0, 1), ["a"], [slice(0, 1, 1)]),
        (slice(None, 1), ["a"], [slice(0, 1, 1)]),
        (slice(2, 8), ["a"], [slice(2, 8, 1)]),
        (slice(0, 7), ["a"], [slice(0, 7, 1)]),
        (slice(0, 10), ["a"], [slice(0, 10, 1)]),
        (4, ["a"], [slice(4, 5, 1)]),
        (9, ["a"], [slice(9, 10, 1)]),
        (10, ["b"], [slice(0, 1, 1)]),
        (14, ["b"], [slice(4, 5, 1)]),
        (slice(10, 11), ["b"], [slice(0, 1, 1)]),
        (slice(10, 15), ["b"], [slice(0, 5, 1)]),
        (slice(15, None), ["c"], [slice(0, 10, 1)]),
        (19, ["c"], [slice(4, 5, 1)]),
        (24, ["c"], [slice(9, 10, 1)]),
        (-1, ["c"], [slice(9, 10, 1)]),
        (-25, ["a"], [slice(0, 1, 1)]),
        (slice(19, None), ["c"], [slice(4, 10, 1)]),
        (slice(15, 16), ["c"], [slice(0, 1, 1)]),
        (slice(15, 25), ["c"], [slice(0, 10, 1)]),
        (slice(-3, None), ["c"], [slice(7, 10, 1)]),
        (slice(20, 100), ["c"], [slice(5, 10, 1)]),
        # Multiple BDFs
        (slice(10, None), ["b", "c"], [slice(0, 5, 1), slice(0, 10, 1)]),
        (slice(14, None), ["b", "c"], [slice(4, 5, 1), slice(0, 10, 1)]),
        (
            slice(0, None),
            ["a", "b", "c"],
            [slice(0, 10, 1), slice(0, 5, 1), slice(0, 10, 1)],
        ),
        (
            slice(None, None),
            ["a", "b", "c"],
            [slice(0, 10, 1), slice(0, 5, 1), slice(0, 10, 1)],
        ),
        (
            slice(0, 25, 1),
            ["a", "b", "c"],
            [slice(0, 10, 1), slice(0, 5, 1), slice(0, 10, 1)],
        ),
        (
            slice(2, 23),
            ["a", "b", "c"],
            [slice(2, 10, 1), slice(0, 5, 1), slice(0, 8, 1)],
        ),
        (slice(0, 15), ["a", "b"], [slice(0, 10, 1), slice(0, 5, 1)]),
        (slice(9, 14), ["a", "b"], [slice(9, 10, 1), slice(0, 4, 1)]),
        (
            slice(9, 16),
            ["a", "b", "c"],
            [slice(9, 10, 1), slice(0, 5, 1), slice(0, 1, 1)],
        ),
        # empty selections, also at the BDF boundaries and at the end (F56)
        (slice(0, 0), [], []),
        (slice(3, 3), [], []),
        (slice(10, 10), [], []),
        (slice(15, 15), [], []),
        (slice(25, 25), [], []),
        (slice(25, None), [], []),
        (slice(30, 40), [], []),
        (slice(12, 4), [], []),
    ],
)
def test_find_bdfs_and_indices_in_selected_times(
    input_time_slice, expected_bdfs, expected_bdf_time_slices
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_bdfs_and_indices_in_selected_times,
    )

    bdfs_in_selected_times, bdf_time_slices = find_bdfs_and_indices_in_selected_times(
        time_indices_by_bdf_simple_3, input_time_slice
    )
    assert bdfs_in_selected_times == expected_bdfs
    assert bdf_time_slices == expected_bdf_time_slices


def test_find_bdfs_and_indices_in_selected_times_exhaustive():
    """For every slice of a partition with BDFs of different lengths (including an
    empty BDF), the BDF-local slices cover exactly the selected integrations."""
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        find_bdfs_and_indices_in_selected_times,
    )

    bdf_lengths = [3, 1, 0, 4, 2]
    bdf_start = np.concatenate([[0], np.cumsum(bdf_lengths)]).tolist()
    names = [f"bdf_{idx}" for idx in range(len(bdf_lengths))]
    owner = [
        names[idx] for idx, length in enumerate(bdf_lengths) for _ in range(length)
    ]
    local = [pos for length in bdf_lengths for pos in range(length)]
    time_len = bdf_start[-1]
    time_indices_by_bdf = {"bdf_names": names, "bdf_start": bdf_start}

    for start, stop in itertools.product(range(-2, time_len + 2), repeat=2):
        selected = range(time_len)[start:stop]
        bdfs, slices = find_bdfs_and_indices_in_selected_times(
            time_indices_by_bdf, slice(start, stop)
        )
        covered = [
            (bdf, pos)
            for bdf, local_slice in zip(bdfs, slices, strict=True)
            for pos in range(local_slice.start, local_slice.stop)
        ]
        assert covered == [(owner[idx], local[idx]) for idx in selected]
        assert all(
            local_slice.step == 1 and local_slice.stop > local_slice.start
            for local_slice in slices
        )
        assert "bdf_2" not in bdfs


@pytest.mark.parametrize(
    "input_time_slice, expected_error",
    [
        (25, IndexError),
        (28, IndexError),
        (-26, IndexError),
        (slice(0, 10, 2), ValueError),
        (slice(None, None, -1), ValueError),
        (1.0, TypeError),
    ],
)
def test_find_bdfs_and_indices_in_selected_times_errors(
    input_time_slice, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        find_bdfs_and_indices_in_selected_times,
    )

    with pytest.raises(expected_error):
        find_bdfs_and_indices_in_selected_times(
            time_indices_by_bdf_simple_3, input_time_slice
        )


def test_find_bdfs_and_indices_in_selected_times_inconsistent():
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        find_bdfs_and_indices_in_selected_times,
    )

    with pytest.raises(ValueError, match="Inconsistent"):
        find_bdfs_and_indices_in_selected_times(
            {"bdf_names": ["a", "b"], "bdf_start": [0, 3]}, slice(None)
        )


KEYS_1D = [
    None,
    slice(None),
    slice(0, 0),
    slice(5, 5),
    slice(2, 5),
    slice(0, 7),
    slice(0, 100),
    slice(-3, None),
    slice(None, -2),
    slice(1, None, 2),
    slice(0, 7, 3),
    slice(None, None, -1),
    slice(5, 1, -2),
    slice(6, None, -3),
    slice(3, 1),
    0,
    6,
    -1,
    -7,
    np.int64(3),
]


@pytest.mark.parametrize("dim_key", KEYS_1D)
def test_split_dim_key(dim_key):
    """block + residual reproduce numpy indexing along one dimension."""
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        apply_residual_keys,
        block_slice_len,
        is_block_slice,
        split_dim_key,
    )

    dim_len = 7
    values = np.arange(dim_len) * 10
    block, residual = split_dim_key(dim_key, dim_len)
    assert is_block_slice(block)
    assert block.step == 1
    assert 0 <= block.start <= block.stop <= dim_len
    loaded = values[block]
    assert len(loaded) == block_slice_len(block)
    expected = values if dim_key is None else values[dim_key]
    np.testing.assert_array_equal(apply_residual_keys(loaded, (residual,)), expected)


@pytest.mark.parametrize("dim_key", [7, -8, 100])
def test_split_dim_key_out_of_range(dim_key):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        split_dim_key,
    )

    with pytest.raises(IndexError):
        split_dim_key(dim_key, 7)


def test_apply_residual_keys_4d():
    """Residuals of several dimensions (ints dropping dimensions, steps) applied to a
    4-D block equal numpy basic indexing of the full array."""
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        apply_residual_keys,
        split_dim_key,
    )

    shape = (5, 6, 7, 4)
    full = np.arange(np.prod(shape)).reshape(shape)
    keys_list = [
        (2, slice(None), slice(1, None, 2), 0),
        (slice(None, None, -2), 3, slice(0, 4), slice(None)),
        (-1, -1, -1, -1),
        (slice(1, 4), slice(0, 6, 5), 6, slice(3, 0, -1)),
        (slice(0, 0), slice(None), 2, slice(None)),
    ]
    for keys in keys_list:
        blocks, residuals = zip(
            *(split_dim_key(key, dim) for key, dim in zip(keys, shape, strict=True)),
            strict=True,
        )
        result = apply_residual_keys(full[blocks], residuals)
        np.testing.assert_array_equal(result, full[keys])


@pytest.mark.parametrize(
    "dim_key, expected",
    [
        (slice(0, 3), True),
        (slice(0, 3, 1), True),
        (slice(2, 2), True),
        (slice(np.int64(1), np.int64(3)), True),
        (slice(None, 3), False),
        (slice(0, None), False),
        (slice(0, 3, 2), False),
        (slice(-1, 3), False),
        (slice(3, 1), False),
        (2, False),
        (None, False),
    ],
)
def test_is_block_slice(dim_key, expected):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        is_block_slice,
    )

    assert is_block_slice(dim_key) is expected
