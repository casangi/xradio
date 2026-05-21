"""
Functions related to indexing of the dimensions of the VISIBILITY and FLAG arrays.
"""

import pyasdm


def min_max_from_dimension_slice(
    dimension_slice: slice | None, default_min: int, default_max: int
) -> [int, int]:
    if isinstance(dimension_slice, int):
        dimension_idx_min = dimension_slice
        dimension_idx_max = dimension_slice + 1
    elif isinstance(dimension_slice, slice):
        dimension_idx_min = dimension_slice.start or default_min
        dimension_idx_max = dimension_slice.stop or default_max
    else:
        dimension_idx_min = default_min
        dimension_idx_max = default_max

    return dimension_idx_min, dimension_idx_max


def find_bdfs_and_indices_in_selected_times(
    time_indices_by_bdf: dict, time_slice: slice | int
) -> tuple[list[str], list[slice]]:

    if (
        time_slice is None
        or isinstance(time_slice, slice)
        and time_slice.start is None
        and time_slice.stop is None
    ):
        bdf_paths = time_indices_by_bdf["bdf_names"]
        return bdf_paths, [slice(None, None)] * len(bdf_paths)

    bdf_paths = time_indices_by_bdf["bdf_names"]
    bdf_start_indices = time_indices_by_bdf["bdf_start"]

    (start_first_found, start_last_found), bdf_slice = _find_index_in_bdf_start_indices(
        time_slice, bdf_start_indices
    )
    if isinstance(time_slice, int):
        bdfs_in_selected_times = bdf_paths[bdf_slice]
        index_within_bdf = time_slice - start_first_found
        time_slices_for_bdfs = [slice(index_within_bdf, index_within_bdf + 1)]

    elif isinstance(time_slice, slice):
        bdfs_in_selected_times = bdf_paths[bdf_slice]

        bdf_slice_len = bdf_slice.stop - bdf_slice.start
        if bdf_slice_len == 1:
            time_slices_for_bdfs = [
                slice(
                    time_slice.start - start_first_found,
                    time_slice.stop - start_first_found,
                )
            ]
        elif bdf_slice_len > 1:
            first_start = (
                None
                if time_slice.start is None
                else time_slice.start - start_first_found
            )
            time_slices_for_bdfs = [slice(first_start, None)]

            time_slices_for_bdfs.extend((bdf_slice_len - 2) * [slice(None, None)])

            last_stop = (
                None if time_slice.stop is None else time_slice.stop - start_last_found
            )
            time_slices_for_bdfs.append(slice(None, last_stop))
        else:
            time_slices_for_bdfs = []

    return bdfs_in_selected_times, time_slices_for_bdfs


def _find_index_in_bdf_start_indices(
    time_slice: int | slice, bdf_start_indices: list[int]
) -> tuple[tuple[int, int], slice]:
    # binary search through BDF start (time) indices (used as start/stop boundaries)
    # Keep absolute indices, does not shift to relative indices within BDFs

    if isinstance(time_slice, int):
        bdf_index_first = _search_index_in_bdf_time_starts(
            time_slice, bdf_start_indices
        )
        bdf_index_last = bdf_index_first
    elif isinstance(time_slice, slice):
        if time_slice.start is None:
            bdf_index_first = 0
        else:
            bdf_index_first = _search_index_in_bdf_time_starts(
                time_slice.start, bdf_start_indices
            )
        if time_slice.stop is None:
            bdf_index_last = len(bdf_start_indices) - 2
        else:
            bdf_index_last = _search_index_in_bdf_time_starts(
                time_slice.stop - 1, bdf_start_indices
            )

    result = (
        bdf_start_indices[bdf_index_first],
        bdf_start_indices[bdf_index_last],
    ), slice(bdf_index_first, bdf_index_last + 1)
    return result


def _search_index_in_bdf_time_starts(index: int, bdf_start: list[int]) -> int:
    len_bdf_start = len(bdf_start)
    if len_bdf_start <= 1:
        return len(bdf_start) - 1

    left_index = 0
    right_index = len(bdf_start) - 1
    while left_index <= right_index:
        middle_index = (left_index + right_index) // 2
        if bdf_start[middle_index] <= index and (
            middle_index + 1 == len_bdf_start or index < bdf_start[middle_index + 1]
        ):
            return middle_index

        if bdf_start[middle_index] > index:
            right_index = middle_index - 1
        else:
            left_index = middle_index + 1

    return middle_index


def calc_auto_cross_baseline_slices(
    array_slice_baseline: slice | int,
    cross_baseline_len: int,
    nantennas: int,
    cross_data_present: bool,
) -> tuple[slice, slice]:
    """
    Indexing/selecting slices for baseline dimension need special treatment because some baselines are
    loaded from the crossData binary component (cross-correlations, loaded as first block) and some
    others are loaded from the autoData binary components (auto-correlations, loaded as a second block).

    This function maps the overall VISIBILITY baseline dimension indices into separate indices for the
    autoData and crossData binary components of the BDFs.

    Turns an overall slice for indexing/selecting baseline id into separate slices for the
    - cross correlations (to be loaded from crossData binary component)
    - the auto correlations (to be loaded form the autoData binary component).
    When either of them is not used (are not within the overall start/stop), their array selection slices
    are set to None (meaning the selection does not take anything from them, as opposed to a
    slice(None, None) which means all is selected).
    """
    auto_baseline_slice = None
    cross_baseline_slice = None
    skip_cross_slice = skip_auto_slice = False

    if not cross_data_present:
        cross_baseline_slice = None
        auto_baseline_slice = array_slice_baseline

    elif isinstance(array_slice_baseline, slice):
        if not array_slice_baseline.start:
            # All global slice => all cross + all auto slices
            cross_start = auto_start = array_slice_baseline.start
        elif array_slice_baseline.start >= cross_baseline_len:
            # All on the auto (second) half
            skip_cross_slice = True
            auto_start = array_slice_baseline.start - cross_baseline_len
        else:
            # Some on the cross (first) half and more on the auto (second) half
            cross_start = array_slice_baseline.start
            auto_start = 0

        if not array_slice_baseline.stop:
            cross_stop = auto_stop = array_slice_baseline.stop
        elif array_slice_baseline.stop >= cross_baseline_len:
            # Part in cross (first) half, and part in auto (second) half
            cross_stop = cross_baseline_len
            auto_stop = array_slice_baseline.stop - cross_baseline_len
        else:
            # All on the cross (first) half
            cross_stop = array_slice_baseline.stop
            skip_auto_slice = True

        if not skip_auto_slice:
            auto_baseline_slice = slice(auto_start, auto_stop)
        if not skip_cross_slice:
            cross_baseline_slice = slice(cross_start, cross_stop)

    elif isinstance(array_slice_baseline, int):
        if array_slice_baseline < cross_baseline_len:
            cross_baseline_slice = array_slice_baseline
        elif array_slice_baseline >= cross_baseline_len and array_slice_baseline < (
            cross_baseline_len + nantennas
        ):
            auto_baseline_slice = array_slice_baseline - cross_baseline_len
        else:
            raise RuntimeError(
                "Unexpected value (too high) of int {array_slice_baseline=}, with {cross_baseline_len=}, {nantennas=}"
            )

    else:
        raise RuntimeError(
            "Unexpected type of {array_slice_baseline=}, {type(array_slice_baseline)=}"
        )

    return cross_baseline_slice, auto_baseline_slice


def find_data_components_needed(
    array_slice: slice | None, bdf_descr: dict
) -> list[str]:
    # For full partition (MSv4) loading we'd load both cross and auto, but depending on indexing/selection we
    # might not need either auto (if indices are lower than the beginning of the auto-correlations) or cross
    # (if indices are higher)
    #
    # Returns a subset of or {"autoData", "crossData"}

    if not array_slice:
        return ["autoData", "crossData"]

    cross_data_present = (
        bdf_descr["correlation_mode"] != pyasdm.enumerations.CorrelationMode.AUTO_ONLY
    )
    antenna_len = bdf_descr["num_antenna"]
    cross_baseline_len = int(antenna_len * (antenna_len - 1) / 2)

    cross_baseline_slice, auto_baseline_slice = calc_auto_cross_baseline_slices(
        array_slice[1], cross_baseline_len, antenna_len, cross_data_present
    )

    components_needed = []
    if auto_baseline_slice is not None:
        components_needed.append("autoData")
    if cross_data_present and cross_baseline_slice is not None:
        components_needed.append("crossData")

    return components_needed
