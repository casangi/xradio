"""
Functions related to indexing of the dimensions of the VISIBILITY and FLAG arrays.
"""


def min_max_from_dimension_slice(
    dimension_slice, default_min, default_max
) -> [int, int]:
    if isinstance(dimension_slice, int):
        dimension_idx_min = dimension_slice
        dimension_idx_max = dimension_slice + 1
    elif isinstance(dimension_slice, slice):
        dimension_idx_min = dimension_slice.start or default_min
        dimension_idx_max = dimension_slice.stop or default_max

    return dimension_idx_min, dimension_idx_max
