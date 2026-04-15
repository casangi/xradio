"""
Module to load data and flags binary components from BDFs.
Robust in the sense that tolerates (common) inconsistencies in the BDF metadata,
including especially incomplete or incorrect axes definitions.

For the data it loads both autoData and crossData binary components for one SPW.

For both data and flags, re-arranges the values for the different times, baselines,
and polarizations in order to produce MSv4-style ndarrays.
"""

import time
import traceback

import numpy as np

import pyasdm

from .basebands_spws import (
    find_if_different_basebands_pols,
    find_if_different_basebands_spws,
    find_spw_in_basebands_list,
)
from .bdf_description_checks import (
    check_basebands,
    check_correlation_mode,
    ensure_presence_binary_components,
    exclude_unsupported_axis_names,
)

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_subset_arr import (
    define_flag_shape,
    define_visibility_shape,
    load_flags_all_subsets,
    load_visibilities_all_subsets,
)
from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
    load_flags_all_subsets_from_trees,
    load_visibilities_all_subsets_from_trees,
)


def array_slice_to_msv4_indices(array_slice: dict) -> tuple[slice, slice, slice, slice]:
    """
    Translates a dict of slices to a tuple of slices, with order
    (time, baseline, frequency, polarization). This latter style is used in
    xr.backends.BackendArray.__getitem__ / xr.core.indexing.ExplicitIndexer
    """

    time_slice = slice(None)
    baseline_slice = slice(None)
    frequency_slice = slice(None)
    polarization_slice = slice(None)

    if "time" in array_slice:
        time_slice = array_slice["time"]
    if "baseline" in array_slice:
        baseline_slice = array_slice["baseline"]
    if "frequency" in array_slice:
        frequency_slice = array_slice["frequency"]
    if "polarization" in array_slice:
        polarization_slice = array_slice["polarization"]

    return (time_slice, baseline_slice, frequency_slice, polarization_slice)


def load_visibilities_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_vis = []
    for bdf_path in bdf_paths:
        visibility_blob = load_visibilities_from_bdf(bdf_path, spw_id, array_slice)

        cumulative_vis.append(visibility_blob)

    start = time.perf_counter()
    visibility = np.concatenate(cumulative_vis)
    end = time.perf_counter()
    xradio_logger().info(
        f"Loaded visibility, with {visibility.shape=} from {len(bdf_paths)=} blobs, time: {end-start:.6}"
    )

    return visibility


def make_bdf_description(bdf_header: pyasdm.bdf.BDFHeader) -> dict:
    bdf_descr = {
        # packed/TIM: dimensionality == 0
        "dimensionality": bdf_header.getDimensionality(),
        "num_time": bdf_header.getNumTime(),
        "processor_type": bdf_header.getProcessorType(),
        "binary_types": bdf_header.getBinaryTypes(),
        "correlation_mode": bdf_header.getCorrelationMode(),
        "apc": bdf_header.getAPClist(),
        "num_antenna": bdf_header.getNumAntenna(),
        "basebands": bdf_header.getBasebandsList(),
    }

    return bdf_descr


def load_visibilities_from_bdf(
    bdf_path: str,
    spw_id: int,
    array_slice: tuple[slice, ...],
    never_reshape_from_all_spws: bool = False,
) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    bdf_descr = make_bdf_description(bdf_header)

    check_correlation_mode(bdf_descr["correlation_mode"])
    check_basebands(bdf_descr["basebands"])
    ensure_presence_binary_components(
        ["crossData", "autoData"], bdf_descr["binary_types"], bdf_path
    )

    baseband_spw_idxs = find_spw_in_basebands_list(
        spw_id, bdf_descr["basebands"], bdf_path
    )
    different_channels_per_spw = find_if_different_basebands_spws(
        bdf_descr["basebands"]
    )
    guessed_shape = define_visibility_shape(bdf_descr, baseband_spw_idxs)
    try:
        if never_reshape_from_all_spws or different_channels_per_spw:
            # TODO: this is assumed to be slower than the simpler version (simply reshape-based)
            # TO: integrate/replace with pyasdm.
            bdf_vis = load_visibilities_all_subsets_from_trees(
                bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr, array_slice
            )
        else:
            bdf_vis = load_visibilities_all_subsets(
                bdf_reader, guessed_shape, baseband_spw_idxs, bdf_descr, array_slice
            )
    except (RuntimeError, ValueError) as exc:
        trace = traceback.format_exc()
        raise RuntimeError(
            f"Error while loading data/visibilities from a BDF ({bdf_path=}). Details: {exc}."
            + trace
            + "BDF header:\n"
            + str(bdf_header)
        )

    return bdf_vis


def load_flags_from_partition_bdfs(
    bdf_paths: list[str], spw_id: int, array_slice: dict | None = None
) -> np.ndarray:

    cumulative_flag = []
    for bdf_path in bdf_paths:
        flag_blob = load_flags_from_bdf(bdf_path, spw_id, array_slice)
        cumulative_flag.append(flag_blob)

    return np.concatenate(cumulative_flag)


def check_flags_dims(flags_dims: list[str]):
    exclude_unsupported_axis_names(flags_dims, True)


def load_flags_from_bdf(
    bdf_path: list[str],
    spw_id: int,
    array_slice: dict,
    never_reshape_from_all_spws: bool = False,
) -> np.ndarray:

    bdf_reader = pyasdm.bdf.BDFReader()
    bdf_reader.open(bdf_path)
    bdf_header = bdf_reader.getHeader()

    check_flags_dims(bdf_header.getAxesNames("flags"))

    bdf_descr = make_bdf_description(bdf_header)

    check_correlation_mode(bdf_descr["correlation_mode"])
    check_basebands(bdf_descr["basebands"])
    ensure_presence_binary_components(["flags"], bdf_descr["binary_types"], bdf_path)

    baseband_spw_idxs = find_spw_in_basebands_list(
        spw_id, bdf_descr["basebands"], bdf_path
    )
    different_pols_per_spw = find_if_different_basebands_pols(bdf_descr["basebands"])
    guessed_shape = define_flag_shape(bdf_descr, baseband_spw_idxs)
    try:
        if never_reshape_from_all_spws or different_pols_per_spw:
            # TODO: this is assumed to be slower than the simpler version (simply reshape-based)
            # TO: integrate/replace with in pyasdm.
            bdf_flag = load_flags_all_subsets_from_trees(
                bdf_reader,
                guessed_shape,
                bdf_descr,
                baseband_spw_idxs,
                array_slice,
            )
        else:
            bdf_flag = load_flags_all_subsets(
                bdf_reader, guessed_shape, baseband_spw_idxs, array_slice
            )
    except (RuntimeError, ValueError) as exc:
        trace = traceback.format_exc()
        raise RuntimeError(
            f"Error while loading flags from a BDF ({bdf_path=}). Details: {exc}."
            + trace
            + "BDF header:"
            + str(bdf_header)
        )

    bdf_flag_expanded = _expand_frequency_in_flags_subset(
        bdf_flag, bdf_descr, baseband_spw_idxs[0], baseband_spw_idxs[1]
    )

    return bdf_flag_expanded


def _expand_frequency_in_flags_subset(
    flag_subset: np.ndarray, bdf_descr: dict, baseband_idx: int, spw_idx: int
) -> np.ndarray:
    """
    The BDF flags binary components do not give flags per-channel. Expand per-SPW
    ndarray from a BDF, with dimensions (time, baseline, polarization) into an MSv4
    flag array with dimensions (time, baseline, frequency, polarization)
    """
    frequency_len = bdf_descr["basebands"][baseband_idx]["spectralWindows"][spw_idx][
        "numSpectralPoint"
    ]
    expanded_shape = flag_subset.shape[0:2] + (frequency_len,) + flag_subset.shape[-1:]
    flag_subset = np.broadcast_to(flag_subset[:, :, np.newaxis, :], expanded_shape)

    return flag_subset
