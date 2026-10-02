"""
Functions related to indexing of the dimensions of the VISIBILITY and FLAG arrays.

The loaders of BDF data work on contiguous blocks: every dimension (time, baseline,
frequency, polarization) is selected with a slice(start, stop, 1) with explicit
non-negative ints ("block slices"). Other (numpy "basic") indices, such as ints,
None, negative indices or slices with steps, are split into a block slice that
bounds the selection plus a "residual" index applied to the loaded block (see
:func:`split_dim_key` and :func:`apply_residual_keys`).
"""

import numbers

import numpy as np


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_block_slice(dim_key) -> bool:
    """
    Whether a key is a slice(start, stop[, 1]) with explicit ints 0 <= start <= stop.

    Parameters
    ----------
    dim_key : slice | int | None
        Index along one dimension.

    Returns
    -------
    bool
        True for block slices (contiguous selection with explicit bounds).
    """
    return bool(
        isinstance(dim_key, slice)
        and dim_key.step in (None, 1)
        and _is_int(dim_key.start)
        and _is_int(dim_key.stop)
        and 0 <= dim_key.start <= dim_key.stop
    )


def block_slice_len(dim_slice: slice) -> int:
    """
    Number of elements selected by a block slice.

    Parameters
    ----------
    dim_slice : slice
        Block slice (see :func:`is_block_slice`).

    Returns
    -------
    int
        stop - start
    """
    return int(dim_slice.stop - dim_slice.start)


def split_dim_key(dim_key, dim_len: int) -> tuple[slice, int | np.ndarray | None]:
    """
    Split a numpy basic index along one dimension into a contiguous block slice and
    a residual index to apply to the block.

    ``array[dim_key]`` equals ``array[block][residual]`` (when residual is not None),
    or ``array[block]`` (when residual is None).

    Parameters
    ----------
    dim_key : slice | int | None
        Index along the dimension. None selects the whole dimension.
    dim_len : int
        Length of the dimension.

    Returns
    -------
    tuple[slice, int | np.ndarray | None]
        - block: slice(start, stop, 1) with explicit ints, 0 <= start <= stop <=
          dim_len (empty when start == stop)
        - residual: None if the block is the selection, 0 for an int key (the
          dimension is to be dropped), or an array of indices within the block
          (slices with step != 1)

    Raises
    ------
    IndexError
        If an int key is out of range.
    TypeError
        If the key is not None, an int or a slice.
    """
    if dim_key is None:
        return slice(0, dim_len, 1), None

    if _is_int(dim_key):
        index = int(dim_key)
        if index < 0:
            index += dim_len
        if not 0 <= index < dim_len:
            raise IndexError(
                f"Index {dim_key} out of range for a dimension of length {dim_len}"
            )
        return slice(index, index + 1, 1), 0

    if isinstance(dim_key, slice):
        indices = range(*dim_key.indices(dim_len))
        if len(indices) == 0:
            return slice(0, 0, 1), None
        if indices.step == 1:
            return slice(indices.start, indices.stop, 1), None
        lowest = min(indices[0], indices[-1])
        highest = max(indices[0], indices[-1])
        residual = np.asarray(indices, dtype=np.intp) - lowest
        return slice(lowest, highest + 1, 1), residual

    raise TypeError(f"Unexpected index type {type(dim_key)} ({dim_key=})")


def apply_residual_keys(
    block: np.ndarray, residuals: tuple[int | np.ndarray | None, ...]
) -> np.ndarray:
    """
    Apply the residual indices (from :func:`split_dim_key`) to a loaded block.

    Parameters
    ----------
    block : np.ndarray
        Loaded block, one dimension per residual.
    residuals : tuple[int | np.ndarray | None, ...]
        Residual index of every dimension of the block.

    Returns
    -------
    np.ndarray
        The selection, with the int-indexed dimensions dropped.
    """
    result = block
    # From the last axis to the first, so that dropping an axis does not shift the
    # axes still to be indexed.
    for axis in reversed(range(len(residuals))):
        residual = residuals[axis]
        if residual is None:
            continue
        result = np.take(result, residual, axis=axis)

    return result


def find_bdfs_and_indices_in_selected_times(
    time_indices_by_bdf: dict, time_slice: slice | int | None
) -> tuple[list[str], list[slice]]:
    """
    Find the BDFs that hold a selection of times (integrations) of a partition, and
    the time indices local to every BDF.

    Parameters
    ----------
    time_indices_by_bdf : dict
        "bdf_names": list of BDF paths; "bdf_start": index of the first integration
        of every BDF in the partition time axis, with len(bdf_names) + 1 elements
        (the last one is the total number of integrations).
    time_slice : slice | int | None
        Selection along the partition time axis. None selects all times. Slices must
        have step 1 (or None); their bounds follow the Python conventions (negative
        values count from the end, out-of-range values are clipped).

    Returns
    -------
    tuple[list[str], list[slice]]
        The BDFs with selected integrations (in time order) and, for every one of
        them, the BDF-local selection as slice(start, stop, 1) with explicit ints
        (never empty). Both lists are empty for an empty selection.

    Raises
    ------
    IndexError
        If an int time index is out of range.
    ValueError
        If time_slice has a step other than 1, or time_indices_by_bdf is
        inconsistent.
    """
    bdf_names = list(time_indices_by_bdf["bdf_names"])
    bdf_start = np.asarray(time_indices_by_bdf["bdf_start"], dtype=np.int64)
    if len(bdf_start) != len(bdf_names) + 1:
        raise ValueError(
            f"Inconsistent time indices by BDF: {len(bdf_names)} BDFs but "
            f"{len(bdf_start)} start indices (expected {len(bdf_names) + 1})"
        )
    time_len = int(bdf_start[-1])

    if time_slice is None:
        start, stop = 0, time_len
    elif _is_int(time_slice):
        time_block, _ = split_dim_key(time_slice, time_len)
        start, stop = time_block.start, time_block.stop
    elif isinstance(time_slice, slice):
        if time_slice.step not in (None, 1):
            raise ValueError(
                f"Only contiguous time selections (step 1) are supported, got {time_slice=}"
            )
        start, stop, _ = time_slice.indices(time_len)
    else:
        raise TypeError(f"Unexpected type of time selection: {type(time_slice)}")

    bdfs_in_selected_times, time_slices_for_bdfs = [], []
    if stop <= start:
        return bdfs_in_selected_times, time_slices_for_bdfs

    first_bdf = int(np.searchsorted(bdf_start, start, side="right")) - 1
    last_bdf = int(np.searchsorted(bdf_start, stop - 1, side="right")) - 1
    for bdf_idx in range(first_bdf, last_bdf + 1):
        bdf_first, bdf_end = int(bdf_start[bdf_idx]), int(bdf_start[bdf_idx + 1])
        local_start = max(start, bdf_first) - bdf_first
        local_stop = min(stop, bdf_end) - bdf_first
        if local_stop > local_start:
            bdfs_in_selected_times.append(bdf_names[bdf_idx])
            time_slices_for_bdfs.append(slice(local_start, local_stop, 1))

    return bdfs_in_selected_times, time_slices_for_bdfs
