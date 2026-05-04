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
