"""
Sanity checks of various fields of the BDF description dict (BDF header metadata)
"""

import pyasdm


def check_correlation_mode(correlation_mode: pyasdm.enumerations.CorrelationMode):
    if correlation_mode == pyasdm.enumerations.CorrelationMode.CROSS_ONLY:
        raise RuntimeError(f" Unexpected {correlation_mode=}")


def ensure_presence_binary_components(
    data_array_names: list[str], binary_types: list[str], bdf_path: str
):

    for array_name in data_array_names:
        if array_name not in binary_types:
            raise RuntimeError(
                f"When trying to load visibility data from BDF: {bdf_path}, it does not "
                f"have array {array_name}"
            )


def exclude_unsupported_axis_names(
    dims: list[str], exclude_also_for_flags: bool = False
):

    # This effectively assumes we'll always get "POL" from the last 3 possible axes,
    # from BDF doc: "The final three axes, STO, POL and HOL, also appear at the same
    # level in the axis hierarchy; however, only one of these axes will normally
    # appear for a given binary component type.
    unsupported = ["STO", "HOL"]

    if exclude_also_for_flags:
        # TODO: Consider also "BIN"
        unsupported.extend(["APC", "SPP"])

    bad_found = []
    for bad_dim in unsupported:
        if bad_dim in dims:
            bad_found.append(bad_dim)

    if bad_found:
        raise RuntimeError(f"Unsupported dimension(s) {bad_found=} in {dims=}")
