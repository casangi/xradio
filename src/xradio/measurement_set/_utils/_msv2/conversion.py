import datetime
import functools
import importlib
import inspect
import os
import pathlib
import time
import traceback
import warnings
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager

import dask.array as da
import numpy as np
import xarray as xr
import zarr.codecs

from xradio._utils.dict_helpers import make_quantity, make_spectral_coord_reference_dict
from xradio._utils.list_and_array import check_if_consistent, unique_1d
from xradio._utils.logging import xradio_logger
from xradio._utils.schema import column_description_casacore_to_msv4_measure
from xradio.measurement_set._utils._msv2 import stream_write
from xradio.measurement_set._utils._msv2._tables.read import (
    convert_casacore_time,
    extract_table_attributes,
    load_generic_table,
    read_col_conversion_dask,
    read_col_conversion_numpy,
)
from xradio.measurement_set._utils._msv2._tables.read_main_table import (
    get_baseline_indices,
    get_baselines,
    utimes_tol_from_times,
)
from xradio.measurement_set._utils._msv2._tables.read_rows import (
    ColumnNotReadableError,
    MainTableRows,
)
from xradio.measurement_set._utils._msv2._tables.subtable_cache import (
    SubtableCache,
    activate_subtable_cache,
    resolve_subtable_cache,
)
from xradio.measurement_set._utils._msv2._tables.table_query import open_table_ro
from xradio.measurement_set._utils._msv2.create_antenna_xds import (
    create_antenna_xds,
    create_gain_curve_xds,
    create_phase_calibration_xds,
)
from xradio.measurement_set._utils._msv2.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set._utils._msv2.msv2_to_msv4_meta import (
    col_dims,
    col_to_data_variable_names,
    create_attribute_metadata,
)
from xradio.measurement_set._utils._msv2.msv4_info_dicts import create_info_dicts
from xradio.measurement_set._utils._msv2.msv4_sub_xdss import (
    create_phased_array_xds,
    create_pointing_xds,
    create_system_calibration_xds,
    create_weather_xds,
)
from xradio.measurement_set._utils._msv2.partition_queries import (
    MainRowRuns,
    PartitionMainRows,
    partition_main_rows,
)
from xradio.measurement_set._utils._msv2.stream_write import (
    DeferredEncodingError,
    DeferredReadError,
    DeferredVariable,
    check_deferred_variables,
    consolidate_msv4,
    deferred_encoding_problems,
    deferred_main_column,
    deferred_ones,
    discard_msv4,
    drop_consolidated_metadata,
    fits_in_memory,
    msv4_members,
    read_deferred_variables,
    write_deferred_variables,
)
from xradio.measurement_set._utils._msv2.subtables import subt_rename_ids
from xradio.measurement_set._utils._utils.stokes_types import stokes_types
from xradio.measurement_set._utils._zarr.encoding import add_encoding
from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION


@contextmanager
def open_partition_main_table(
    in_file: str,
    partition_info: dict,
    main_row_runs: PartitionMainRows | None = None,
) -> Generator[MainTableRows, None, None]:
    """
    Opens the MAIN rows of a partition for reading (no TaQL selection of the
    MAIN table): the base MAIN table, opened here and closed on exit, and the
    partition's row numbers.

    Parameters
    ----------
    in_file : str
        Input MSv2 path.
    partition_info : dict
        Partition description (create_partitions).
    main_row_runs : PartitionMainRows | None, optional
        The partition's MAIN rows from create_partitions_with_main_rows, used
        if they belong to ``partition_info`` (see partition_main_rows).

    Yields
    ------
    MainTableRows
        The partition rows.
    """
    with open_table_ro(in_file) as main_tb:
        main_rows = MainTableRows(
            main_tb, partition_main_rows(main_tb, partition_info, main_row_runs)
        )
        try:
            yield main_rows
        finally:
            main_rows.close()


# Default chunks of the main xds (main_chunksize=None, see default_main_chunksize):
# along time only, with time chunks of about this size (uncompressed) for the
# largest data variable.
DEFAULT_MAIN_CHUNK_BYTES = 128 * 2**20
# Largest buffer the Blosc compressor encodes (numcodecs raises for larger ones),
# i.e. the largest zarr chunk the default compressor can write.
BLOSC_MAX_BUFFER_BYTES = 2**31 - 1


def parse_chunksize(
    chunksize: dict | float | None, xds_type: str, xds: xr.Dataset
) -> dict[str, int] | None:
    """
    Parameters
    ----------
    chunksize : Union[Dict, float, None]
        Desired maximum size of the chunks, either as a dict of per-dimension sizes or as
        an amount of memory
    xds_type : str
        whether to use chunking logic for main or pointing datasets
    xds : xr.Dataset
        dataset to calculate best chunking

    Returns
    -------
    Dict[str, int] | None
        dictionary of chunk sizes (as dim->size), or None for None (the
        defaults: one chunk per variable for pointing, default_main_chunksize
        for main, which needs the data variables)
    """
    if isinstance(chunksize, dict):
        check_chunksize(chunksize, xds_type)
    elif isinstance(chunksize, float):
        chunksize = mem_chunksize_to_dict(chunksize, xds_type, xds)
    elif chunksize is not None:
        raise ValueError(
            f"Chunk size expected as a dict or a float, got: "
            f" {chunksize} (of type {type(chunksize)}"
        )

    return chunksize


def check_chunksize(chunksize: dict, xds_type: str) -> None:
    """
    Rudimentary check of the chunksize parameters to catch obvious errors early before
    more work is done.
    """
    # perphaps start using some TypeDict or/and validator like pydantic?
    if xds_type == "main":
        allowed_dims = [
            "time",
            "baseline_id",
            "antenna_name",
            "frequency",
            "polarization",
        ]
    elif xds_type == "pointing":
        allowed_dims = ["time", "antenna"]

    msg = ""
    for dim in chunksize.keys():
        if dim not in allowed_dims:
            msg += f"dimension {dim} not allowed in {xds_type} dataset:\n"
    if msg:
        raise ValueError(f"Wrong keys found in chunksize: {msg}")


def mem_chunksize_to_dict(
    chunksize: float, xds_type: str, xds: xr.Dataset
) -> dict[str, int]:
    """
    Given a desired 'chunksize' as amount of memory in GB, calculate best chunk sizes
    for every dimension of an xds.

    Parameters
    ----------
    chunksize : float
        Desired maximum size of the chunks
    xds_type : str
        whether to use chunking logic for main or pointing datasets
    xds : xr.Dataset
        dataset to auto-calculate chunking of its dimensions

    Returns
    -------
    Dict[str, int]
        dictionary of chunk sizes (as dim->size)
    """

    if xds_type == "pointing":
        sizes = mem_chunksize_to_dict_pointing(chunksize, xds)
    elif xds_type == "main":
        sizes = mem_chunksize_to_dict_main(chunksize, xds)
    else:
        raise RuntimeError(f"Unexpected type: {xds_type=}")

    return sizes


GiBYTES_TO_BYTES = 1024 * 1024 * 1024


def mem_chunksize_to_dict_main(chunksize: float, xds: xr.Dataset) -> dict[str, int]:
    """
    Checks the assumption that all polarizations can be held in memory, at least for one
    data point (one time, one freq, one channel).

    It presently relies on the logic of mem_chunksize_to_dict_main_balanced() to find a
    balanced list of dimension sizes for the chunks

    Assumes these relevant dims: (time, antenna_name/baseline_id, frequency,
    polarization).
    """

    sizeof_vis = itemsize_spec(xds)
    size_all_pols = sizeof_vis * xds.sizes["polarization"]
    if size_all_pols / GiBYTES_TO_BYTES > chunksize:
        raise RuntimeError(
            "Cannot calculate chunk sizes when memory bound ({chunksize}) does not even allow all polarizations in one chunk"
        )

    baseline_or_antenna_name = find_baseline_or_antenna_var(xds)
    total_size = calc_used_gb(xds.sizes, baseline_or_antenna_name, sizeof_vis)

    ratio = chunksize / total_size
    chunked_dims = ["time", baseline_or_antenna_name, "frequency", "polarization"]
    if ratio >= 1:
        result = {dim: xds.sizes[dim] for dim in chunked_dims}
        xradio_logger().debug(
            f"{chunksize=} GiB is enough to fully hold {total_size=} GiB (for {xds.sizes=}) in memory in one chunk"
        )
    else:
        xds_dim_sizes = {k: xds.sizes[k] for k in chunked_dims}
        result = mem_chunksize_to_dict_main_balanced(
            chunksize, xds_dim_sizes, baseline_or_antenna_name, sizeof_vis
        )

    return result


def mem_chunksize_to_dict_main_balanced(
    chunksize: float,
    xds_dim_sizes: dict,
    baseline_or_antenna_name: str,
    sizeof_vis: int,
) -> dict[str, int]:
    """
    Assumes the ratio is <1 and all pols can fit in memory (from
    mem_chunksize_to_dict_main()).

    What is kept balanced is the fraction of the total size of every dimension included in a
    chunk. For example, time: 10, baseline: 100, freq: 1000, if we can afford about 33% in
    one chunk, the chunksize will be ~ time: 3, baseline: 33, freq: 333.
    The polarization axis is excluded from the calculations.
    Because this can leave a leftover (below or above the desired chunksize limit) and
    adjustment is done to get the final memory use below but as close as possible to
    'chunksize'. This adjustment alters the balance.

    Parameters
    ----------
    chunksize : float
        Desired maximum size of the chunks
    xds_dim_sizes : dict
        Dataset dimension sizes as dim_name->size
    sizeof_vis : int
        Size in bytes of a data point (one visibility / spectrum value)

    Returns
    -------
    Dict[str, int]
        dictionary of chunk sizes (as dim->size)
    """

    dim_sizes = [size for size in xds_dim_sizes.values()]
    # Fix fourth dim (polarization) to all (not free to auto-calculate)
    free_dims_mask = np.array([True, True, True, False])

    total_size = np.prod(dim_sizes) * sizeof_vis / GiBYTES_TO_BYTES
    ratio = chunksize / total_size

    dim_chunksizes = np.array(dim_sizes, dtype="int64")
    factor = ratio ** (1 / np.sum(free_dims_mask))
    dim_chunksizes[free_dims_mask] = np.maximum(
        dim_chunksizes[free_dims_mask] * factor, 1
    )
    used = np.prod(dim_chunksizes) * sizeof_vis / GiBYTES_TO_BYTES

    xradio_logger().debug(
        f"Auto-calculating main chunk sizes. First order approximation {dim_chunksizes=}, used total: {used} GiB (with {chunksize=} GiB)"
    )

    # Iterate through the dims, starting from the dims with lower chunk size
    #  (=bigger impact of a +1)
    # Note the use of np.floor, this iteration can either increase or decrease sizes,
    #  if increasing sizes we want to keep mem use below the upper limit, floor(2.3) = +2
    #  if decreasing sizes we want to take mem use below the upper limit, floor(-2.3) = -3
    indices = np.argsort(dim_chunksizes[free_dims_mask])
    for idx in indices:
        left = chunksize - used
        other_dims_mask = np.ones(free_dims_mask.shape, dtype=bool)
        other_dims_mask[idx] = False
        delta = np.divide(
            left,
            np.prod(dim_chunksizes[other_dims_mask]) * sizeof_vis / GiBYTES_TO_BYTES,
        )
        int_delta = np.floor(delta)
        if abs(int_delta) > 0 and int_delta + dim_chunksizes[idx] > 0:
            dim_chunksizes[idx] += int_delta
        used = np.prod(dim_chunksizes) * sizeof_vis / GiBYTES_TO_BYTES

    chunked_dim_names = ["time", baseline_or_antenna_name, "frequency", "polarization"]
    dim_chunksizes_int = [int(v) for v in dim_chunksizes]
    result = dict(zip(chunked_dim_names, dim_chunksizes_int, strict=False))

    xradio_logger().debug(
        f"Auto-calculated main chunk sizes with {chunksize=}, {total_size=} GiB (for {dim_sizes=}): {result=} which uses {used} GiB."
    )

    return result


def mem_chunksize_to_dict_pointing(chunksize: float, xds: xr.Dataset) -> dict[str, int]:
    """
    Equivalent to mem_chunksize_to_dict_main adapted to pointing xdss.
    Assumes these relevant dims: (time, antenna, direction).
    """

    if not xds.sizes:
        return {}

    sizeof_pointing = itemsize_pointing_spec(xds)
    chunked_dim_names = [name for name in xds.sizes.keys()]
    dim_sizes = [size for size in xds.sizes.values()]
    total_size = np.prod(dim_sizes) * sizeof_pointing / GiBYTES_TO_BYTES

    # Fix third dim (direction) to all
    free_dims_mask = np.array([True, True, False])

    ratio = chunksize / total_size
    if ratio >= 1:
        xradio_logger().debug(
            f"Pointing chunsize: {chunksize=} GiB is enough to fully hold {total_size=} GiB (for {xds.sizes=}) in memory in one chunk"
        )
        dim_chunksizes = dim_sizes
    else:
        # balanced
        dim_chunksizes = np.array(dim_sizes, dtype="int")
        factor = ratio ** (1 / np.sum(free_dims_mask))
        dim_chunksizes[free_dims_mask] = np.maximum(
            dim_chunksizes[free_dims_mask] * factor, 1
        )
        used = np.prod(dim_chunksizes) * sizeof_pointing / GiBYTES_TO_BYTES

        xradio_logger().debug(
            f"Auto-calculating pointing chunk sizes. First order approximation: {dim_chunksizes=}, used total: {used=} GiB (with {chunksize=} GiB"
        )

        indices = np.argsort(dim_chunksizes[free_dims_mask])
        # refine dim_chunksizes
        for idx in indices:
            left = chunksize - used
            other_dims_mask = np.ones(free_dims_mask.shape, dtype=bool)
            other_dims_mask[idx] = False
            delta = np.divide(
                left,
                np.prod(dim_chunksizes[other_dims_mask])
                * sizeof_pointing
                / GiBYTES_TO_BYTES,
            )
            int_delta = np.floor(delta)
            if abs(int_delta) > 0 and int_delta + dim_chunksizes[idx] > 0:
                dim_chunksizes[idx] += int_delta

            used = np.prod(dim_chunksizes) * sizeof_pointing / GiBYTES_TO_BYTES

    dim_chunksizes_int = [int(v) for v in dim_chunksizes]
    result = dict(zip(chunked_dim_names, dim_chunksizes_int, strict=False))

    if ratio < 1:
        xradio_logger().debug(
            f"Auto-calculated pointing chunk sizes with {chunksize=}, {total_size=} GiB (for {xds.sizes=}): {result=} which uses {used} GiB."
        )

    return result


def _chunk_nbytes(var: xr.Variable, chunks: dict[str, int]) -> int:
    """Bytes (uncompressed) of the largest chunk of a variable for the chunk
    sizes ``chunks`` (dimensions not in ``chunks`` whole)."""
    nbytes = int(var.dtype.itemsize)
    for dim, size in zip(var.dims, var.shape, strict=True):
        nbytes *= min(int(size), int(chunks.get(dim, size)))
    return nbytes


def default_main_chunksize(
    xds: xr.Dataset, data_var_names: list[str] | None = None
) -> dict[str, int]:
    """
    The chunk sizes of the main xds for main_chunksize=None: chunks along time
    only (every other dimension whole), with a time chunk length that makes
    the chunk of the largest data variable about DEFAULT_MAIN_CHUNK_BYTES
    (uncompressed), at least one time step.

    Only if one time step of the largest data variable is above the Blosc
    limit (BLOSC_MAX_BUFFER_BYTES, the largest chunk the compressor encodes),
    the frequency axis and then the baseline (antenna) axis are split too, to
    chunks of about DEFAULT_MAIN_CHUNK_BYTES, and a warning is logged: the
    streamed write reads and writes whole time steps (a batch holds every
    chunk of at least one time step), so it then holds one time step of a
    data variable, more than the Blosc limit, in memory.

    The memory of the streamed write is so about DEFAULT_MAIN_CHUNK_BYTES per
    batch only if a time step of the largest data variable is at most that
    size; otherwise it is one time step.

    Parameters
    ----------
    xds : xr.Dataset
        The main xds with its data variables (values, lazy or placeholders:
        only their dimensions, shapes and dtypes are used).
    data_var_names : list[str] | None, optional
        The data variables to size the chunks by, by default all of them.

    Returns
    -------
    dict[str, int]
        Chunk sizes by dimension ({} if there is no data variable along time).
    """
    if data_var_names is None:
        data_var_names = list(xds.data_vars)
    variables = [
        xds[name].variable
        for name in data_var_names
        if name in xds.data_vars and "time" in xds[name].dims
    ]
    if not variables:
        return {}
    n_times = int(xds.sizes["time"])
    per_time = max(_chunk_nbytes(var, {"time": 1}) for var in variables)
    chunks = {
        "time": max(1, min(n_times, DEFAULT_MAIN_CHUNK_BYTES // max(1, per_time)))
    }
    for dim in ("frequency", "baseline_id", "antenna_name"):
        if max(_chunk_nbytes(var, chunks) for var in variables) <= (
            BLOSC_MAX_BUFFER_BYTES
        ):
            break
        with_dim = [var for var in variables if dim in var.dims]
        if not with_dim:
            continue
        per_element = max(_chunk_nbytes(var, chunks | {dim: 1}) for var in with_dim)
        chunks[dim] = max(
            1, min(int(xds.sizes[dim]), DEFAULT_MAIN_CHUNK_BYTES // per_element)
        )
    split = [dim for dim in chunks if dim != "time"]
    if split:
        xradio_logger().warning(
            f"One time step of a main data variable is {per_time / 2**30:.2f} GiB, "
            f"more than the {BLOSC_MAX_BUFFER_BYTES / 2**30:.0f} GiB the Blosc "
            f"compressor encodes in one chunk: the default chunks also split "
            f"{', '.join(split)} ({chunks}). The streamed write still reads and "
            "writes whole time steps (one per batch here), so it holds one time "
            f"step ({per_time / 2**30:.2f} GiB) of a data variable in memory. "
            "Pass main_chunksize to choose the chunks."
        )
    return chunks


def find_baseline_or_antenna_var(xds: xr.Dataset) -> str:
    if "baseline_id" in xds.coords:
        baseline_or_antenna_name = "baseline_id"
    elif "antenna_name" in xds.coords:
        baseline_or_antenna_name = "antenna_name"

    return baseline_or_antenna_name


def itemsize_spec(xds: xr.Dataset) -> int:
    """
    Size in bytes of one visibility (or spectrum) value.
    """
    names = ["SPECTRUM", "VISIBILITY"]
    itemsize = 8
    for var in names:
        if var in xds.data_vars:
            var_name = var
            itemsize = np.dtype(xds.data_vars[var_name].dtype).itemsize
            break

    return itemsize


def itemsize_pointing_spec(xds: xr.Dataset) -> int:
    """
    Size in bytes of one pointing (or spectrum) value.
    """
    pnames = ["BEAM_POINTING"]
    itemsize = 8
    for var in pnames:
        if var in xds.data_vars:
            var_name = var
            itemsize = np.dtype(xds.data_vars[var_name].dtype).itemsize
            break

    return itemsize


def calc_used_gb(
    chunksizes: dict, baseline_or_antenna_name: str, sizeof_vis: int
) -> float:
    return (
        chunksizes["time"]
        * chunksizes[baseline_or_antenna_name]
        * chunksizes["frequency"]
        * chunksizes["polarization"]
        * sizeof_vis
        / GiBYTES_TO_BYTES
    )


def calc_indx_for_row_split(
    tb_tool: MainTableRows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time and baseline index of every row of a partition, and its baselines and
    unique times.

    Parameters
    ----------
    tb_tool : MainTableRows
        The partition rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        tidxs, bidxs (time and baseline index of every row), ANTENNA1 and
        ANTENNA2 of every baseline, and the unique times (seconds from the
        Unix epoch).
    """
    baselines = get_baselines(tb_tool)
    col_names = tb_tool.colnames()
    cshapes = [
        np.array(tb_tool.getcell(col, 0)).shape
        for col in col_names
        if tb_tool.iscelldefined(col, 0)
    ]

    # (raises if no column has 2-D cells in the first row)
    freq_cnt, pol_cnt = [(cc[0], cc[1]) for cc in cshapes if len(cc) == 2][0]
    times = tb_tool.getcol("TIME")
    utimes, tol = utimes_tol_from_times(times)
    tidxs = np.searchsorted(utimes, times)
    del times

    ts_ant1, ts_ant2 = (
        tb_tool.getcol("ANTENNA1"),
        tb_tool.getcol("ANTENNA2"),
    )

    ts_bases = np.column_stack((ts_ant1, ts_ant2))
    bidxs = get_baseline_indices(baselines, ts_bases)

    baseline_ant1_id = baselines[:, 0]
    baseline_ant2_id = baselines[:, 1]

    return (
        tidxs,
        bidxs,
        baseline_ant1_id,
        baseline_ant2_id,
        convert_casacore_time(utimes, False),
    )


def create_coordinates(
    xds: xr.Dataset,
    in_file: str,
    ddi: int,
    utime: np.ndarray,
    interval: np.ndarray,
    baseline_ant1_id: np.ndarray,
    baseline_ant2_id: np.ndarray,
    scan_id: np.ndarray,
    scan_intents: list[str],
) -> tuple[xr.Dataset, int]:
    """
    Creates coordinates of a VisibilityXds/SpectrumXds and assigns them to the input
    correlated dataset.

    Parameters
    ----------
    xds :
        dataset to add the coords to
    in_file :
        path to input MSv2
    ddi :
        DDI index (row) for this MSv4
    utime :
        unique times, for the time coordinate
    interval :
        interval col values from the MSv2, for the integration_time attribute
        of the time coord
    baseline_ant1_id :
        ANTENNA1 ids to be used as coord
    baseline_ant2_id :
        ANTENNA2 ids to be used as coord
    scan_id :
        SCAN_ID values from MSv2, for the scan_name coord
    scan_intents :
        list of SCAN_INTENT values from MSv2, for the scan_intents attribute of the
        scan_name coord

    Returns
    -------
    tuple[xr.Dataset, int]
        A tuple of:
        - The input dataset with coordinates added and populated with all MSv4 schema
          attributes.
        - The MSv2 spectral_window_id of this DDI/MSv4, which is no longer added to
          the frequency coord but is required to create other secondary xdss (antenna,
          gain_curve, phase_calibration, system_calibration, field_and_source).
    """
    coords = {
        "time": utime,
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "baseline_id": np.arange(len(baseline_ant1_id)),
        "scan_name": ("time", scan_id.astype(str)),
        "uvw_label": ["u", "v", "w"],
    }

    ddi_xds = load_generic_table(in_file, "DATA_DESCRIPTION").sel(row=ddi)
    pol_setup_id = ddi_xds.POLARIZATION_ID.values
    spectral_window_id = int(ddi_xds.SPECTRAL_WINDOW_ID.values)

    spectral_window_xds = load_generic_table(
        in_file,
        "SPECTRAL_WINDOW",
        rename_ids=subt_rename_ids["SPECTRAL_WINDOW"],
    ).sel(spectral_window_id=spectral_window_id)
    coords["frequency"] = spectral_window_xds["CHAN_FREQ"].data[
        ~(np.isnan(spectral_window_xds["CHAN_FREQ"].data))
    ]

    pol_xds = load_generic_table(
        in_file,
        "POLARIZATION",
        rename_ids=subt_rename_ids["POLARIZATION"],
    )
    num_corr = int(pol_xds["NUM_CORR"][pol_setup_id].values)
    coords["polarization"] = np.vectorize(stokes_types.get)(
        pol_xds["CORR_TYPE"][pol_setup_id, :num_corr].values
    )

    xds = xds.assign_coords(coords)

    ##### Add scan intents attribute to scan_name coord #####
    xds.scan_name.attrs["scan_intents"] = scan_intents

    ###### Create Frequency Coordinate ######
    freq_column_description = spectral_window_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]

    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_FREQ"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
    )
    xds.frequency.attrs.update(msv4_measure)

    spw_name = spectral_window_xds.NAME.values.item()
    if (spw_name is None) or (spw_name == "none") or (spw_name == ""):
        spw_name = "spw_" + str(spectral_window_id)
    else:
        # spw_name = spectral_window_xds.NAME.values.item()
        spw_name = spw_name + "_" + str(spectral_window_id)

    xds.frequency.attrs["spectral_window_name"] = spw_name
    xds.frequency.attrs["spectral_window_intents"] = ["UNSPECIFIED"]
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["REF_FREQUENCY"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
    )
    xds.frequency.attrs["reference_frequency"] = make_spectral_coord_reference_dict(
        float(spectral_window_xds.REF_FREQUENCY.values),
        msv4_measure["units"],
        msv4_measure["observer"],
    )

    # Add if doppler table is present
    # xds.frequency.attrs["doppler_velocity"] =
    # xds.frequency.attrs["doppler_type"] =

    unique_chan_width = unique_1d(
        spectral_window_xds["CHAN_WIDTH"].data[
            np.logical_not(np.isnan(spectral_window_xds["CHAN_WIDTH"].data))
        ]
    )
    # assert len(unique_chan_width) == 1, "Channel width varies for spectral_window."
    # xds.frequency.attrs["channel_width"] = spectral_window_xds.chan_width.data[
    #    ~(np.isnan(spectral_window_xds.chan_width.data))
    # ]  # unique_chan_width[0]
    msv4_measure = column_description_casacore_to_msv4_measure(
        freq_column_description["CHAN_WIDTH"],
        ref_code=spectral_window_xds["MEAS_FREQ_REF"].data,
    )
    xds.frequency.attrs["channel_width"] = make_quantity(
        np.abs(unique_chan_width[0]), msv4_measure["units"] if msv4_measure else "Hz"
    )

    ###### Create Time Coordinate ######
    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]
    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["TIME"]
    )
    xds.time.attrs.update(msv4_measure)

    msv4_measure = column_description_casacore_to_msv4_measure(
        main_column_descriptions["INTERVAL"]
    )
    xds.time.attrs["integration_time"] = make_quantity(
        interval, msv4_measure["units"] if msv4_measure else "s"
    )

    return xds, spectral_window_id


def find_min_max_times(tb_tool: MainTableRows) -> tuple:
    """
    Find the min/max times in an MSv4, for constraining pointing.

    To avoid numerical comparison issues (leaving out some times at the edges),
    it substracts/adds a tolerance from/to the min and max values. The tolerance
    is a fraction of the difference between times / interval of the MS (see
    utimes_tol_from_times()).

    Parameters
    ----------
    tb_tool : MainTableRows
        The rows of the partition of this MSv4

    Returns
    -------
    tuple
        min/max times (raw time values from the Msv2 table)
    """
    utimes, tol = utimes_tol_from_times(tb_tool.getcol("TIME"))
    time_min = utimes.min() - tol
    time_max = utimes.max() + tol
    return (time_min, time_max)


def data_variables_parallel_mode(
    parallel_mode: str, main_chunksize: dict | None
) -> str:
    """
    The parallel_mode the MAIN columns are read with: "time" needs a time
    chunk size in ``main_chunksize`` and is "none" without one.

    Parameters
    ----------
    parallel_mode : str
        The conversion's parallel_mode.
    main_chunksize : dict | None
        Chunk sizes of the main xds (as parsed by parse_chunksize).

    Returns
    -------
    str
        The parallel_mode of the column reads.
    """
    time_chunksize = main_chunksize.get("time", None) if main_chunksize else None
    if parallel_mode == "time" and time_chunksize is None:
        return "none"
    return parallel_mode


def create_data_variables(
    in_file: str,
    xds: xr.Dataset,
    main_rows: MainTableRows,
    time_baseline_shape: tuple[int, int],
    tidxs: np.ndarray,
    bidxs: np.ndarray,
    parallel_mode: str,
    main_chunksize: dict | None,
    deferred: dict[str, DeferredVariable] | None = None,
    unreadable_columns: frozenset[str] = frozenset(),
):
    """
    Reads the MAIN columns that become data variables of the main xds and adds
    them to ``xds`` (in place). A column that fails to read is skipped (logged
    at debug level); if WEIGHT_SPECTRUM fails, WEIGHT is tried instead.

    With ``deferred`` (streamed write), the columns are not read here: every
    data variable is a placeholder and its description is put in
    ``deferred``; ``write_deferred_variables`` reads and writes the values
    after the MSv4 metadata. A column is skipped here if its storage manager
    tells that the read would fail (see ``check_partition_cells``); if the
    read fails anyway, the partition is converted again with the column in
    ``unreadable_columns``.

    Parameters
    ----------
    in_file : str
        Input MSv2 path.
    xds : xr.Dataset
        Main xds, with its coordinates already set.
    main_rows : MainTableRows
        The partition's MAIN rows.
    time_baseline_shape : tuple[int, int]
        (n_times, n_baselines) of the partition.
    tidxs, bidxs : np.ndarray
        Time and baseline index of every partition row.
    parallel_mode : str
        "time" gives lazy (dask) arrays for the large columns, when the time
        chunk size is set in ``main_chunksize``.
    main_chunksize : dict | None
        Chunk sizes of the main xds.
    deferred : dict[str, DeferredVariable] | None, optional
        Streamed write (parallel_mode "none" or "partition", or "time" without
        a time chunk size): filled with the descriptions of the placeholder
        variables, by name. By default None: the columns are read into the
        xds.
    unreadable_columns : frozenset[str], optional
        Columns skipped as if their read had failed (a read of a previous
        attempt of the streamed write failed), by default none.
    """
    time_chunksize = main_chunksize.get("time", None) if main_chunksize else None
    if parallel_mode != data_variables_parallel_mode(parallel_mode, main_chunksize):
        given = (
            "the default main_chunksize=None"
            if main_chunksize is None
            else f"main_chunksize={main_chunksize!r}"
        )
        how = " (streamed write)" if deferred is not None else ""
        xradio_logger().warning(
            f"parallel_mode='time' needs a 'time' chunk size: with {given}, the "
            f"data variables are converted as with parallel_mode='none'{how}, not "
            "by dask tasks along time. Pass main_chunksize={'time': n} for dask "
            "parallelism along time."
        )
        parallel_mode = "none"

    # Create Data Variables
    col_names = main_rows.colnames()

    target_cols = set(col_names) & set(col_to_data_variable_names.keys())
    if target_cols.issuperset({"WEIGHT", "WEIGHT_SPECTRUM"}):
        target_cols.remove("WEIGHT")

    main_table_attrs = extract_table_attributes(in_file)
    main_column_descriptions = main_table_attrs["column_descriptions"]

    # Use a double-ended queue in case WEIGHT_SPECTRUM conversion fails, and
    # we need to add WEIGHT to list of columns to convert during iteration.
    # Sorted, so that the read order (and with it the data variable order and the
    # memory peak, which depends on what is already held when each column is
    # read) does not depend on the hash seed (set iteration order).
    target_cols = deque(sorted(target_cols))

    if deferred is not None and parallel_mode == "time":
        raise ValueError(
            "The streamed write needs parallel_mode 'none' or 'partition' (got "
            f"{parallel_mode!r} with a time chunk size)"
        )

    while target_cols:
        col = target_cols.popleft()
        datavar_name = col_to_data_variable_names[col]
        if deferred is None:
            read_col_conversion = get_read_col_conversion_function(col, parallel_mode)

        try:
            start = time.time()
            if col in unreadable_columns:
                raise ColumnNotReadableError(
                    f"Column {col}: its read failed in a previous attempt"
                )
            if deferred is None:
                read_args = (main_rows, col, time_baseline_shape, tidxs, bidxs)
                if read_col_conversion is read_col_conversion_dask:
                    read_args += (time_chunksize,)
                col_data = read_col_conversion(*read_args)
                col_data = postprocess_main_column(
                    col, col_data, parallel_mode, xds.sizes, main_chunksize
                )
            else:
                # The same conversion, applied to every batch when it is written
                transform = functools.partial(
                    postprocess_main_column,
                    col,
                    parallel_mode="none",
                    main_sizes={"frequency": xds.sizes["frequency"]},
                    main_chunksize=None,
                )
                col_data, spec = deferred_main_column(
                    main_rows,
                    col,
                    datavar_name,
                    time_baseline_shape,
                    transform,
                    # repeated along frequency: the same in either order
                    frequency_constant=col == "WEIGHT",
                )

            xds[datavar_name] = xr.DataArray(
                col_data,
                dims=col_dims[col],
                attrs=create_attribute_metadata(col, main_column_descriptions),
            )
            if deferred is not None:
                deferred[datavar_name] = spec
            xradio_logger().debug(f"Time to read column {col} : {time.time() - start}")

        except Exception as exc:
            xradio_logger().debug(f"Could not load column {col}, exception: {exc}")
            xradio_logger().debug(traceback.format_exc())

            if col == "WEIGHT_SPECTRUM" and "WEIGHT" in col_names:
                xradio_logger().debug(
                    "Failed to convert WEIGHT_SPECTRUM column: "
                    "will attempt to use WEIGHT instead"
                )
                target_cols.append("WEIGHT")

    # The grid plan and time-chunk rows (8-24 bytes per row) are not needed
    # after the reads: do not keep them alive through to_zarr. The lazy
    # (parallel_mode="time") blocks keep their own reference.
    main_rows.release_plans()


# The MAIN columns read lazily (dask) with parallel_mode="time"
DASK_COLUMNS = frozenset(
    {
        "DATA",
        "CORRECTED_DATA",
        "MODEL_DATA",
        "WEIGHT_SPECTRUM",
        "WEIGHT",
        "FLAG",
    }
)


def get_read_col_conversion_function(col_name: str, parallel_mode: str) -> Callable:
    """
    Returns the appropriate read_col_conversion function: use the dask version
    for large columns and parallel_mode="time", or the numpy version otherwise.
    """
    if parallel_mode == "time" and col_name in DASK_COLUMNS:
        return read_col_conversion_dask
    return read_col_conversion_numpy


def repeat_weight_array(
    weight_arr,
    parallel_mode: str,
    main_sizes: dict[str, int],
    main_chunksize: dict[str, int],
):
    """
    Repeat the weights read from the WEIGHT column along the frequency dimension.
    Returns a dask array if parallel_mode="time", or a numpy array otherwise.
    """
    reshaped_arr = weight_arr[:, :, None, :]
    repeats = (1, 1, main_sizes["frequency"], 1)

    if parallel_mode == "time":
        result = da.tile(reshaped_arr, repeats)
        # da.tile() adds each repeat as a separate chunk, so rechunking is necessary
        chunksizes = tuple(
            main_chunksize.get(dim, main_sizes[dim])
            for dim in ("time", "baseline_id", "frequency", "polarization")
        )
        return result.rechunk(chunksizes)

    return np.tile(reshaped_arr, repeats)


def postprocess_main_column(
    col: str,
    col_data,
    parallel_mode: str = "none",
    main_sizes: dict[str, int] | None = None,
    main_chunksize: dict[str, int] | None = None,
):
    """
    The conversion applied to the values of a MAIN column after reading them
    onto the (time, baseline) grid: TIME_CENTROID to seconds from the Unix
    epoch, WEIGHT repeated along frequency (see repeat_weight_array); other
    columns are returned as they are. Used for whole columns and for the
    batches of the streamed write.

    Parameters
    ----------
    col : str
        MAIN column name.
    col_data : np.ndarray | da.Array
        Column values on the (time, baseline, ...) grid.
    parallel_mode : str, optional
        As for repeat_weight_array.
    main_sizes : dict[str, int] | None, optional
        Sizes of the main xds (WEIGHT only).
    main_chunksize : dict[str, int] | None, optional
        Chunk sizes of the main xds (WEIGHT with parallel_mode="time" only).

    Returns
    -------
    np.ndarray | da.Array
        The converted values.
    """
    if col == "TIME_CENTROID":
        return convert_casacore_time(col_data, False)
    if col == "WEIGHT":
        return repeat_weight_array(col_data, parallel_mode, main_sizes, main_chunksize)
    return col_data


def add_missing_data_var_attrs(xds):
    """
    Adds in the xds attributes expected metadata that cannot be found in the input MSv2.
    For now:
    - missing single-dish/SPECTRUM metadata
    - missing interferometry/VISIBILITY_MODEL metadata
    """
    data_var_names = ["SPECTRUM", "SPECTRUM_CORRECTED"]
    for var_name in data_var_names:
        if var_name in xds.data_vars:
            xds.data_vars[var_name].attrs["units"] = ""

    vis_var_names = ["VISIBILITY_MODEL"]
    for var_name in vis_var_names:
        if var_name in xds.data_vars and "units" not in xds.data_vars[var_name].attrs:
            # Assume MODEL uses the same units
            if "VISIBILITY" in xds.data_vars:
                xds.data_vars[var_name].attrs["units"] = xds.data_vars[
                    "VISIBILITY"
                ].attrs["units"]
            else:
                xds.data_vars[var_name].attrs["units"] = ""

    return xds


def create_taql_query_where(partition_info: dict) -> str:
    """
    The TaQL WHERE clause of the MAIN rows of a partition, as recorded in the
    partition_info of the MSv4 ("taql_where"). The rows are selected without
    TaQL, by the numpy twin of this clause (partition_queries.select_main_rows)
    or the row runs of create_partitions_with_main_rows.

    Parameters
    ----------
    partition_info : dict
        Partition description.

    Returns
    -------
    str
        The WHERE clause.
    """
    main_par_table_cols = [
        "DATA_DESC_ID",
        "OBSERVATION_ID",
        "STATE_ID",
        "FIELD_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "ANTENNA1",
    ]

    taql_where = "WHERE "
    for col_name in main_par_table_cols:
        if col_name in partition_info:
            if partition_info[col_name][0] is not None:
                taql_where = (
                    taql_where
                    + f"({col_name} IN [{','.join(map(str, partition_info[col_name]))}]) AND"
                )
                if col_name == "ANTENNA1":
                    taql_where = (
                        taql_where
                        + f"(ANTENNA2 IN [{','.join(map(str, partition_info[col_name]))}]) AND"
                    )
    taql_where = taql_where[:-3]

    return taql_where


def fix_uvw_frame(
    xds: xr.Dataset, field_and_source_xds: xr.Dataset, is_single_dish: bool
) -> xr.Dataset:
    """
    Fix UVW frame

    From CASA fixvis docs: clean and the im tool ignore the reference frame claimed by the UVW column (it is often
    mislabelled as ITRF when it is really FK5 (J2000)) and instead assume the (u, v, w)s are in the same frame as the phase
    tracking center. calcuvw does not yet force the UVW column and field centers to use the same reference frame!
    Blank = use the phase tracking frame of vis.
    """
    if xds.UVW.attrs["frame"] == "ITRF":
        if is_single_dish:
            center_var = "FIELD_REFERENCE_CENTER_DIRECTION"
        else:
            center_var = "FIELD_PHASE_CENTER_DIRECTION"

        xds.UVW.attrs["frame"] = field_and_source_xds[center_var].attrs["frame"]

    return xds


def estimate_memory_for_partition(
    in_file: str, partition: dict, main_row_runs: PartitionMainRows | None = None
) -> float:
    """
    Aim: given a partition description, estimates a safe maximum memory value, but avoiding overestimation
    (at least not adding not well understood factors).

    Parameters
    ----------
    in_file : str
        Input MSv2 path.
    partition : dict
        Partition description (create_partitions).
    main_row_runs : PartitionMainRows | None, optional
        The partition's MAIN rows from create_partitions_with_main_rows, used
        if they belong to ``partition`` (see partition_main_rows). Otherwise
        the rows are selected from the MAIN key columns.

    Returns
    -------
    float
        Estimated memory in GiB.
    """

    def calculate_term_all_data(
        tb_tool: MainTableRows, ntimes: float, nbaselines: float
    ) -> tuple[list[float], bool]:
        """
        Size that DATA vars from MS will have in the MSv4, whether this MS has FLOAT_DATA
        """
        sizes_all_data_vars = []
        col_names = tb_tool.colnames()
        for data_col in ["DATA", "CORRECTED_DATA", "MODEL_DATA", "FLOAT_DATA"]:
            if data_col in col_names:
                col_descr = tb_tool.table.getcoldesc(data_col)
                if "shape" in col_descr and isinstance(col_descr["shape"], np.ndarray):
                    # example: "shape": array([15,  4]) => gives pols x channels
                    cells_in_row = col_descr["shape"].prod()
                else:
                    # the first row of the partition
                    first_row = np.array(tb_tool.getcell(data_col, 0))
                    cells_in_row = np.prod(first_row.shape)

                if col_descr["valueType"] == "complex":
                    # Assume. Otherwise, read first column and get the itemsize:
                    # col_dtype = np.array(mtable.col(data_col)[0]).dtype
                    # cell_size = col_dtype.itemsize
                    cell_size = 4
                    if data_col != "FLOAT_DATA":
                        cell_size *= 2
                elif col_descr["valueType"] == "float":
                    cell_size = 4

                # cells_in_row should account for the polarization and frequency dims
                size_data_var = ntimes * nbaselines * cells_in_row * cell_size

                sizes_all_data_vars.append(size_data_var)

        is_float_data = "FLOAT_DATA" in col_names

        return sizes_all_data_vars, is_float_data

    def calculate_term_weight_flag(size_largest_data, is_float_data) -> float:
        """
        Size that WEIGHT and FLAG will have in the MSv4, derived from the size of the
        MSv2 DATA col=> MSv4 VIS/SPECTRUM data var.
        """
        # Factors of the relative "cell_size" wrt the DATA var
        # WEIGHT_SPECTRUM size: DATA (IF), DATA/2 (SD)
        factor_weight = 1.0 if is_float_data else 0.5
        factor_flag = 1.0 / 4.0 if is_float_data else 1.0 / 8.0

        return size_largest_data * (factor_weight + factor_flag)

    def calculate_term_other_data_vars(
        ntimes: int, nbaselines: int, is_float_data: bool
    ) -> float:
        """
        Size all data vars other than the DATA (visibility/spectrum) vars will have in the MSv4

        For the rest of columns, including indices/iteration columns and other
        scalar columns could say approx ->5% of the (large) data cols

        """
        # Small ones, but as they are loaded into data arrays, why not including,
        # For example: UVW (3xscalar), EXPOSURE, TIME_CENTROID
        # assuming float64 in output MSv4
        item_size = 8
        return ntimes * nbaselines * (3 + 1 + 1) * item_size

    def calculate_term_calc_indx_for_row_split(msv2_nrows: int) -> float:
        """
        Account for the per-row indices of a partition: tidxs and bidxs
        (calc_indx_for_row_split()) and the partition's MAIN row numbers.

        In terms of amount of memory represented by this term relative to the
        total, it becomes relevant proportionally to the ratio between
           nrows / (chans x pols)
        - for example LOFAR long scans/partitions with few channels,
        but its value is independent from # chans, pols.
        """
        item_size = 8
        # 3 are: tidxs, bidxs, MAIN row numbers
        return msv2_nrows * 3 * item_size

    def calculate_term_other_msv2_indices(msv2_nrows: int) -> float:
        """
        Account for the allocations to load ID, etc. columns from input MSv2.
        The converter needs to load: OBSERVATION_ID, INTERVAL, SCAN_NUMBER.
        These are loaded one after another (allocations do not stack up).
        Also, in most memory profiles these allocations are released once we
        get to create_data_variables(). As such, adding this term will most
        likely lead to overestimation (but adding it for safety).

        Simlarly as with calculate_term_calc_indx_for_row_split() this term
        becomes relevant when the ratio 'nrows / (chans x pols)' is high.
        """
        # assuming float64/int64 in input MSv2, which seems to be the case,
        # except for OBSERVATION_ID (int32)
        item_size = 8
        return msv2_nrows * item_size

    def calculate_term_attrs(size_estimate_main_xds: float) -> float:
        """Rough guess which seems to be more than enough"""
        # could also account for info_dicts (which seem to require typically ~1 MB)
        return 10 * 1024 * 1024

    def calculate_term_sub_xds(size_estimate_main_xds: float) -> float:
        """
        This is still very rough. Just seemingly working for now. Not taking into account the dims
        of the sub-xdss, interpolation options used, etc.
        """
        # Most cases so far 1% seems enough
        return 0.015 * size_estimate_main_xds

    def calculate_term_to_zarr(size_estimate_main_xds: float) -> float:
        """
        The to_zarr call on the main_xds seems to allocate 10s or 100s of MBs, presumably for buffers.
        That adds on top of the expected main_xds size.
        This is currently a very rough extrapolation and is being (mis)used to give a safe up to 5-6%
        overestimation. Perhaps we should drop this term once other sub-xdss are accounted for (and
        this term could be replaced by a similar, smaller but still safe over-estimation percentage).
        """
        return 0.05 * size_estimate_main_xds

    with open_partition_main_table(in_file, partition, main_row_runs) as tb_tool:
        # Do not feel tempted to rely on nrows. nrows tends to underestimate memory when baselines are missing.
        # For some EVN datasets that can easily underestimate by a 50%
        utimes, _tol = utimes_tol_from_times(tb_tool.getcol("TIME"))
        ntimes = len(utimes)
        nbaselines = len(get_baselines(tb_tool))

        # Still, use nrwos for estimations related to sizes of input (MSv2)
        # columns, not sizes of output (MSv4) data vars
        msv2_nrows = tb_tool.nrows()

        sizes_all_data, is_float_data = calculate_term_all_data(
            tb_tool, ntimes, nbaselines
        )

    size_largest_data = np.max(sizes_all_data)
    sum_sizes_data = np.sum(sizes_all_data)
    estimate_main_xds = (
        sum_sizes_data
        + calculate_term_weight_flag(size_largest_data, is_float_data)
        + calculate_term_other_data_vars(ntimes, nbaselines, is_float_data)
    )
    estimate = (
        estimate_main_xds
        + calculate_term_calc_indx_for_row_split(msv2_nrows)
        + calculate_term_other_msv2_indices(msv2_nrows)
        + calculate_term_sub_xds(estimate_main_xds)
        + calculate_term_to_zarr(estimate_main_xds)
        + calculate_term_attrs(estimate_main_xds)
    )
    estimate /= GiBYTES_TO_BYTES

    return estimate


def estimate_memory_and_cores_for_partitions(
    in_file: str, partitions: list, main_row_runs: MainRowRuns | None = None
) -> tuple[float, int, int]:
    """
    Estimates approximate memory required to convert an MSv2 to MSv4, given
    a predefined set of partitions (and optionally their MAIN rows,
    ``main_row_runs[i]`` for ``partitions[i]``, from
    create_partitions_with_main_rows).
    """
    max_cores = len(partitions)

    size_estimates = [
        estimate_memory_for_partition(
            in_file,
            part_description,
            None if main_row_runs is None else main_row_runs[idx],
        )
        for idx, part_description in enumerate(partitions)
    ]
    max_estimate = np.max(size_estimates) if size_estimates else 0.0

    recommended_cores = np.ceil(max_cores / 4).astype("int")

    return float(max_estimate), int(max_cores), int(recommended_cores)


USE_TABLE_ITER_DEPRECATION = (
    "use_table_iter is deprecated and has no effect: the MAIN table is always read "
    "in bounded calls, without the table iterator"
)


def warn_use_table_iter(use_table_iter: bool, stacklevel: int = 3) -> None:
    """
    Emit the DeprecationWarning of use_table_iter=True (a no-op since the MAIN
    table is read in bounded calls).

    Parameters
    ----------
    use_table_iter : bool
        The value given.
    stacklevel : int, optional
        Stack level of the warning (the caller of the caller by default).
    """
    if use_table_iter:
        warnings.warn(
            USE_TABLE_ITER_DEPRECATION, DeprecationWarning, stacklevel=stacklevel
        )


def convert_and_write_partition(*args, **kwargs):
    """
    Converts one partition of an MSv2 into an MSv4 and writes it (see
    ``_convert_and_write_partition`` for the parameters, the signature is
    the same without its internal ones; use_table_iter is a deprecated no-op,
    True emits a DeprecationWarning).

    With the streamed write of the MAIN data variables, a column whose read
    fails after the MSv4 metadata was written (see ``DeferredReadError``) makes
    the partition be converted again without that column (the MSv4 written so
    far is removed first): the result is that of the non-streamed path, which
    skips a column whose read fails. If the zarr metadata declares an encoding
    that the streamed write does not apply (``DeferredEncodingError``, raised
    before any value is written), the partition is converted again without
    the streamed write. When the failed attempt itself created the MSv4 store
    (the error carries ``created_msv4``), the next attempt overwrites it:
    persistence mode "w-" (the default, fail if the MSv4 exists) becomes "w",
    so an MSv4 that ``discard_msv4`` could not remove does not make the next
    attempt fail. A failure before the store was written (for example a read
    of a partition that is read whole before its single to_zarr) keeps "w-",
    so an MSv4 that existed before the conversion is never overwritten.
    """
    arguments = convert_and_write_partition.__signature__.bind(*args, **kwargs)
    arguments = dict(arguments.arguments)
    warn_use_table_iter(arguments.get("use_table_iter", False))
    unreadable: set[str] = set()
    allow_stream_write = True
    for _ in range(len(col_to_data_variable_names) + 2):
        try:
            return _convert_and_write_partition(
                **arguments,
                unreadable_columns=frozenset(unreadable),
                allow_stream_write=allow_stream_write,
            )
        except DeferredEncodingError as exc:
            if not allow_stream_write:  # not streamed again: cannot happen
                raise
            xradio_logger().warning(
                f"{exc}: converting the partition again without the streamed write"
            )
            allow_stream_write = False
            created_msv4 = getattr(exc, "created_msv4", False)
        except DeferredReadError as exc:
            if exc.col in unreadable:  # not read again: cannot happen
                raise
            xradio_logger().info(
                f"Column {exc.col} could not be read ({exc}: {exc.__cause__!r}): "
                "converting the partition again without it"
            )
            unreadable.add(exc.col)
            created_msv4 = getattr(exc, "created_msv4", False)
        if created_msv4 and arguments.get("persistence_mode", "w-") == "w-":
            arguments["persistence_mode"] = "w"
    raise RuntimeError(f"Columns {sorted(unreadable)} could not be read")


def _convert_and_write_partition(
    in_file: str,
    out_file: str,
    ms_v4_id: int | str,
    partition_info: dict,
    use_table_iter: bool,
    partition_scheme: str = "ddi_intent_field",
    main_chunksize: dict | float | None = None,
    with_pointing: bool = True,
    pointing_chunksize: dict | float | None = None,
    pointing_interpolate: bool = False,
    ephemeris_interpolate: bool = False,
    phase_cal_interpolate: bool = False,
    sys_cal_interpolate: bool = False,
    # the codec default is an immutable config object, safe to build once here
    compressor: zarr.abc.codec.BytesBytesCodec | None = zarr.codecs.BloscCodec(  # noqa: B008
        cname="lz4", clevel=5, shuffle="noshuffle"
    ),
    add_reshaping_indices: bool = False,
    storage_backend="zarr",
    parallel_mode: str = "none",
    persistence_mode: str = "w-",
    subtable_cache: SubtableCache | None = None,
    main_row_runs: PartitionMainRows | None = None,
    unreadable_columns: frozenset[str] = frozenset(),
    allow_stream_write: bool = True,
):
    """_summary_

    Parameters
    ----------
    in_file : str
        _description_
    out_file : str
        _description_
    use_table_iter : bool
        Deprecated, has no effect (the MAIN table is read in bounded calls).
    scan_intents : str
        _description_
    ddi : int, optional
        _description_, by default 0
    state_ids : _type_, optional
        _description_, by default None
    field_id : int, optional
        _description_, by default None
    main_chunksize : Union[Dict, float, None], optional
        Chunk sizes of the main xds (see convert_msv2_to_processing_set), by
        default None: chunks along time only (default_main_chunksize).
    with_pointing: bool, optional
        _description_, by default True
    pointing_chunksize : Union[Dict, float, None], optional
        _description_, by default None
    pointing_interpolate : bool, optional
        _description_, by default None
    ephemeris_interpolate : bool, optional
        _description_, by default None
    phase_cal_interpolate : bool, optional
        _description_, by default None
    sys_cal_interpolate : bool, optional
        _description_, by default None
    compressor : zarr.abc.codec.BytesBytesCodec, optional
        _description_, by default zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle="noshuffle")
    add_reshaping_indices : bool, optional
        _description_, by default False
    storage_backend : str, optional
        _description_, by default "zarr"
    parallel_mode : _type_, optional
        _description_
    persistence_mode: str = "w-",
        _description_, by default "w-"
    subtable_cache : SubtableCache | None, optional
        Sub-table data shared by the partitions of a conversion (see
        _tables/subtable_cache.py), by default None: a cache for this partition
        only.
    main_row_runs : PartitionMainRows | None, optional
        MAIN rows of the partition from create_partitions_with_main_rows, by
        default None. Used only if they were computed for the same row
        selection as ``partition_info``; otherwise the rows are selected from
        the MAIN key columns (the rows create_taql_query_where describes).
    unreadable_columns : frozenset[str], optional
        MAIN columns skipped as if their read had failed (set by
        convert_and_write_partition after a failed read of the streamed
        write), by default none.
    allow_stream_write : bool, optional
        False disables the streamed write of the MAIN data variables (set by
        convert_and_write_partition after a DeferredEncodingError), by
        default True.

    Returns
    -------
    _type_
        _description_
    """
    from toolviper.utils.memory_management import free_memory, memory_setup

    memory_setup(131072)

    ms_xdt = xr.DataTree()  # MSv4 as a Data Tree

    taql_where = create_taql_query_where(partition_info)
    # Streamed write of the MAIN data variables (see stream_write.py): they are
    # written after the MSv4 metadata, one at a time, in batches of whole zarr
    # chunks along time. parallel_mode="time" with a time chunk size already
    # writes the large ones lazily (dask); without one it reads like "none"
    # (decided below).
    use_stream_write = (
        allow_stream_write
        and storage_backend == "zarr"
        and parallel_mode in ("none", "partition", "time")
    )
    stream_batch_bytes = stream_write.STREAM_BATCH_BYTES
    ddi = partition_info["DATA_DESC_ID"][0]
    scan_intents = str(partition_info["OBS_MODE"][0]).split(",")

    start = time.time()
    with (
        activate_subtable_cache(resolve_subtable_cache(subtable_cache)),
        open_partition_main_table(in_file, partition_info, main_row_runs) as tb_tool,
    ):
        if tb_tool.nrows() == 0:
            return xr.Dataset(), {}, {}

        xradio_logger().debug("Starting a real convert_and_write_partition")
        (
            tidxs,
            bidxs,
            baseline_ant1_id,
            baseline_ant2_id,
            utime,
        ) = calc_indx_for_row_split(tb_tool)
        time_baseline_shape = (len(utime), len(baseline_ant1_id))
        xradio_logger().debug("Calc indx for row split " + str(time.time() - start))

        observation_id = check_if_consistent(
            tb_tool.getcol("OBSERVATION_ID"), "OBSERVATION_ID"
        )

        def get_observation_info(in_file, observation_id, scan_intents):
            generic_observation_xds = load_generic_table(
                in_file,
                "OBSERVATION",
                taql_where=f" where (ROWID() IN [{str(observation_id)}])",
            )

            if scan_intents == "None":
                scan_intents = "obs_" + str(observation_id)

            return generic_observation_xds["TELESCOPE_NAME"].values[0], scan_intents

        telescope_name, scan_intents = get_observation_info(
            in_file, observation_id, scan_intents
        )

        start = time.time()
        xds = xr.Dataset(
            attrs={
                "schema_version": MSV4_SCHEMA_VERSION,
                "creator": {
                    "software_name": "xradio",
                    "version": importlib.metadata.version("xradio"),
                },
                "creation_date": datetime.datetime.now(datetime.UTC).isoformat(),
                "type": "visibility",
            }
        )

        # interval = check_if_consistent(tb_tool.getcol("INTERVAL"), "INTERVAL")
        interval = tb_tool.getcol("INTERVAL")

        interval_unique = unique_1d(interval)
        if len(interval_unique) > 1:
            xradio_logger().debug(
                "Integration time (interval) not consitent in partition, using median."
            )
            interval = np.median(interval)
        else:
            interval = interval_unique[0]

        scan_id = np.full(time_baseline_shape, -42, dtype=int)
        scan_id[tidxs, bidxs] = tb_tool.getcol("SCAN_NUMBER")
        scan_id = np.max(scan_id, axis=1)

        xds, spectral_window_id = create_coordinates(
            xds,
            in_file,
            ddi,
            utime,
            interval,
            baseline_ant1_id,
            baseline_ant2_id,
            scan_id,
            scan_intents,
        )
        xradio_logger().debug("Time create coordinates " + str(time.time() - start))

        start = time.time()
        main_chunksize = parse_chunksize(main_chunksize, "main", xds)
        deferred: dict[str, DeferredVariable] | None = (
            {}
            if use_stream_write
            and data_variables_parallel_mode(parallel_mode, main_chunksize) != "time"
            else None
        )
        create_data_variables(
            in_file,
            xds,
            tb_tool,
            time_baseline_shape,
            tidxs,
            bidxs,
            parallel_mode,
            main_chunksize,
            deferred=deferred,
            unreadable_columns=unreadable_columns,
        )

        # Add data_groups
        xds, is_single_dish = add_data_groups(xds)
        xds = add_missing_data_var_attrs(xds)

        if (
            "WEIGHT" not in xds.data_vars
        ):  # Some single dish datasets don't have WEIGHT.
            if deferred is not None:  # streamed write: written after the metadata
                like = xds.SPECTRUM if is_single_dish else xds.VISIBILITY
                ones, deferred["WEIGHT"] = deferred_ones("WEIGHT", like.shape)
                xds["WEIGHT"] = xr.DataArray(ones, dims=like.dims)
            elif is_single_dish:
                xds["WEIGHT"] = xr.DataArray(
                    np.ones(xds.SPECTRUM.shape, dtype=np.float64),
                    dims=xds.SPECTRUM.dims,
                )
            else:
                xds["WEIGHT"] = xr.DataArray(
                    np.ones(xds.VISIBILITY.shape, dtype=np.float64),
                    dims=xds.VISIBILITY.dims,
                )

        xradio_logger().debug("Time create data variables " + str(time.time() - start))

        # To constrain the time range to load (in pointing, ephemerides, phase_cal data_vars)
        time_min_max = find_min_max_times(tb_tool)

        # Create ant_xds
        start = time.time()
        feed_id = unique_1d(
            np.concatenate(
                [
                    unique_1d(tb_tool.getcol("FEED1")),
                    unique_1d(tb_tool.getcol("FEED2")),
                ]
            )
        )
        antenna_id = unique_1d(
            np.concatenate(
                [xds["baseline_antenna1_id"].data, xds["baseline_antenna2_id"].data]
            )
        )

        ant_xds = create_antenna_xds(
            in_file,
            spectral_window_id,
            antenna_id,
            feed_id,
            telescope_name,
            xds.polarization,
        )
        xradio_logger().debug("Time antenna xds  " + str(time.time() - start))

        start = time.time()
        gain_curve_xds = create_gain_curve_xds(in_file, spectral_window_id, ant_xds)
        xradio_logger().debug("Time gain_curve xds  " + str(time.time() - start))

        start = time.time()
        if phase_cal_interpolate:
            phase_cal_interp_time = xds.time.values
        else:
            phase_cal_interp_time = None
        phase_calibration_xds = create_phase_calibration_xds(
            in_file,
            spectral_window_id,
            ant_xds,
            time_min_max,
            phase_cal_interp_time,
        )
        xradio_logger().debug("Time phase_calibration xds  " + str(time.time() - start))

        # Create system_calibration_xds
        start = time.time()
        if sys_cal_interpolate:
            sys_cal_interp_time = xds.time.values
        else:
            sys_cal_interp_time = None
        system_calibration_xds = create_system_calibration_xds(
            in_file,
            spectral_window_id,
            xds.frequency,
            ant_xds,
            sys_cal_interp_time,
        )
        xradio_logger().debug("Time system_calibation " + str(time.time() - start))

        # Change antenna_ids to antenna_names
        with_antenna_partitioning = "ANTENNA1" in partition_info
        xds = antenna_ids_to_names(
            xds, ant_xds, is_single_dish, with_antenna_partitioning
        )
        # but before, keep the name-id arrays, we need them for the pointing and weather xds
        ant_xds_name_ids = ant_xds["antenna_name"].set_xindex("antenna_id")
        ant_position_xds_with_ids = ant_xds["ANTENNA_POSITION"].set_xindex("antenna_id")
        # No longer needed after converting to name.
        ant_xds = ant_xds.drop_vars("antenna_id")

        # Create weather_xds
        start = time.time()
        weather_xds = create_weather_xds(in_file, ant_position_xds_with_ids)
        xradio_logger().debug("Time weather " + str(time.time() - start))

        # Create pointing_xds
        pointing_xds = xr.Dataset()
        if with_pointing:
            start = time.time()
            if pointing_interpolate:
                pointing_interp_time = xds.time
            else:
                pointing_interp_time = None
            pointing_xds = create_pointing_xds(
                in_file, ant_xds_name_ids, time_min_max, pointing_interp_time
            )
            pointing_chunksize = parse_chunksize(
                pointing_chunksize, "pointing", pointing_xds
            )
            add_encoding(pointing_xds, compressor=compressor, chunks=pointing_chunksize)
            xradio_logger().debug(
                "Time pointing (with add compressor and chunking) "
                + str(time.time() - start)
            )

        # Create phased array xds
        phased_array_xds = create_phased_array_xds(
            in_file,
            ant_xds.antenna_name,
            ant_xds.receptor_label,
            ant_xds.polarization_type,
        )

        start = time.time()

        # Time and frequency should always be increasing
        reverse_frequency = False
        if len(xds.frequency) > 1 and xds.frequency[1] - xds.frequency[0] < 0:
            xds = xds.sel(frequency=slice(None, None, -1))
            reverse_frequency = True  # the streamed write reverses every batch

        if len(xds.time) > 1 and xds.time[1] - xds.time[0] < 0:
            if deferred is not None:
                # The times are sorted (np.unique), so this is not reached;
                # the batches are not reversed along time.
                raise RuntimeError(
                    "The streamed write needs increasing times, got decreasing ones"
                )
            xds = xds.sel(time=slice(None, None, -1))

        # Create field_and_source_xds (combines field, source and ephemeris data into one super dataset)
        start = time.time()

        # if "FIELD_ID" not in partition_scheme:
        #     field_id = np.full(time_baseline_shape, -42, dtype=int)
        #     field_id[tidxs, bidxs] = tb_tool.getcol("FIELD_ID")
        #     field_id = np.max(field_id, axis=1)
        #     field_times = utime
        # else:
        #     field_id = check_if_consistent(tb_tool.getcol("FIELD_ID"), "FIELD_ID")
        #     field_times = None

        field_id = np.full(
            time_baseline_shape, -42, dtype=int
        )  # -42 used for missing baselines
        field_id[tidxs, bidxs] = tb_tool.getcol("FIELD_ID")
        field_id = np.max(field_id, axis=1)
        field_times = xds.time.values

        # col_unique = unique_1d(col)
        # assert len(col_unique) == 1, col_name + " is not consistent."
        # return col_unique[0]

        field_and_source_xds, source_id, _num_lines, field_names = (
            create_field_and_source_xds(
                in_file,
                field_id,
                spectral_window_id,
                field_times,
                is_single_dish,
                time_min_max,
                ephemeris_interpolate,
            )
        )

        xradio_logger().debug("Time field_and_source_xds " + str(time.time() - start))

        xds = fix_uvw_frame(xds, field_and_source_xds, is_single_dish)
        xds = xds.assign_coords({"field_name": ("time", field_names)})

        partition_info_misc_fields = {
            "scan_name": xds.coords["scan_name"].data,
            "taql_where": taql_where,
        }
        if with_antenna_partitioning:
            partition_info_misc_fields["antenna_name"] = xds.coords[
                "antenna_name"
            ].data[0]
        info_dicts = create_info_dicts(
            in_file, xds, field_and_source_xds, partition_info_misc_fields, tb_tool
        )
        xds.attrs.update(info_dicts)

        # xds ready, prepare to write
        start = time.time()
        if main_chunksize is None:
            # (UVW is dropped from single dish xdss)
            main_chunksize = default_main_chunksize(
                xds,
                [
                    name
                    for name in xds.data_vars
                    if not (is_single_dish and name == "UVW")
                ],
            )
        add_encoding(xds, compressor=compressor, chunks=main_chunksize)
        xradio_logger().debug(
            "Time add compressor and chunk " + str(time.time() - start)
        )

        os.path.join(
            out_file,
            pathlib.Path(in_file).name.replace(".ms", "") + "_" + str(ms_v4_id),
        )

        if is_single_dish:
            xds.attrs["type"] = "spectrum"
            xds = xds.drop_vars("UVW")
            xds = xds.drop_dims("uvw_label")
        else:
            if xds.attrs["processor_info"]["type"] == "RADIOMETER":
                xds.attrs["type"] = "radiometer"
            else:
                xds.attrs["type"] = "visibility"

        # Add tidxs and bidxs for testing
        if add_reshaping_indices:
            xds["tidxs"] = tidxs
            xds["bidxs"] = bidxs
            xds["row_id"] = tb_tool.rownumbers()  # tb_tool.getcol("row_id")

        if deferred is not None:
            # Read the data variables whole and write the MSv4 with one to_zarr,
            # as the non-streamed path, if they are small (a fraction of a batch
            # together: the streamed write costs a few ms per variable) or if
            # xarray's encoding of a variable would change its values (the
            # streamed write writes them to the zarr arrays directly)
            problems = deferred_encoding_problems(xds, deferred)
            if problems:
                xradio_logger().warning(
                    "The MAIN data variables are not streamed (their encoding "
                    f"changes the values): {'; '.join(problems)}"
                )
            if problems or fits_in_memory(xds, deferred, stream_batch_bytes):
                read_deferred_variables(
                    xds, deferred, tb_tool, tidxs, bidxs, reverse_frequency
                )
                deferred = None

        start = time.time()
        ms_v4_name = pathlib.Path(in_file).name.replace(".ms", "") + "_" + str(ms_v4_id)
        ms_xdt.ds = xds

        ms_xdt["/antenna_xds"] = ant_xds
        for group_name in xds.attrs["data_groups"]:
            ms_xdt["/" + f"field_and_source_{group_name}_xds"] = field_and_source_xds

        if with_pointing and len(pointing_xds.data_vars) > 0:
            ms_xdt["/pointing_xds"] = pointing_xds

        if system_calibration_xds:
            ms_xdt["/system_calibration_xds"] = system_calibration_xds

        if gain_curve_xds:
            ms_xdt["/gain_curve_xds"] = gain_curve_xds

        if phase_calibration_xds:
            ms_xdt["/phase_calibration_xds"] = phase_calibration_xds

        if weather_xds:
            ms_xdt["/weather_xds"] = weather_xds

        if phased_array_xds:
            ms_xdt["/phased_array_xds"] = phased_array_xds

        if storage_backend == "zarr":
            from xradio._utils.zarr.config import ZARR_FORMAT

            store_path = os.path.join(out_file, ms_v4_name)
            if deferred is None:
                ms_xdt.to_zarr(
                    store=store_path,
                    mode=persistence_mode,
                    zarr_format=ZARR_FORMAT,
                )
            else:
                # Streamed write: all metadata (encodings), coordinates, small
                # variables and sub-datasets now; the deferred (placeholder)
                # variables are never computed but written next, one at a time,
                # in batches of whole zarr chunks along time. The consolidated
                # metadata is written last: until then (and after a hard kill)
                # the MSv4 opens as incomplete, as one whose to_zarr was
                # interrupted (see stream_write.py).
                check_deferred_variables(xds, deferred)
                members_before = msv4_members(store_path)
                store_existed = members_before is not None
                ms_xdt.to_zarr(
                    store=store_path,
                    mode=persistence_mode,
                    zarr_format=ZARR_FORMAT,
                    compute=False,
                    consolidated=False,
                )
                try:
                    drop_consolidated_metadata(store_path)
                    write_deferred_variables(
                        store_path,
                        xds,
                        deferred,
                        tb_tool,
                        tidxs,
                        bidxs,
                        time_baseline_shape[1],
                        reverse_frequency,
                        stream_batch_bytes,
                    )
                    consolidate_msv4(store_path)
                except BaseException as exc:
                    # No MSv4 with unwritten (fill value) data variables is left:
                    # after a failed read the partition is converted again without
                    # the column, after an encoding the streamed write does not
                    # apply without the streamed write (convert_and_write_partition),
                    # otherwise the error is raised.
                    done = discard_msv4(
                        store_path,
                        set(deferred),
                        remove_store=persistence_mode in ("w", "w-")
                        or not store_existed,
                        members_before=members_before,
                    )
                    # Lets convert_and_write_partition overwrite, on its next
                    # attempt, only an MSv4 this attempt created.
                    exc.created_msv4 = not store_existed
                    log = xradio_logger().debug
                    if not isinstance(exc, DeferredReadError | DeferredEncodingError):
                        log = xradio_logger().error
                    log(f"Writing the data variables of {store_path} failed: {done}")
                    raise
        elif storage_backend == "netcdf":
            # xds.to_netcdf(path=file_name+"/MAIN", mode=mode) #Does not work
            raise
        xradio_logger().debug("Write data  " + str(time.time() - start))

        # get_logger().info("Saved ms_v4 " + file_name + " in " + str(time.time() - start_with) + "s")

        # Drop the dataset reference and trigger explicit cleanup to help release
        # memory retained by Dask task graphs and large NumPy-backed arrays after writing.
        ms_xdt = None
        free_memory()


# convert_and_write_partition has the signature of _convert_and_write_partition
# (for help(), inspect and the binding of its arguments) without the parameters
# it sets itself on every attempt
_inner_signature = inspect.signature(_convert_and_write_partition)
convert_and_write_partition.__signature__ = _inner_signature.replace(
    parameters=[
        param
        for param in _inner_signature.parameters.values()
        if param.name not in ("unreadable_columns", "allow_stream_write")
    ]
)
del _inner_signature


def antenna_ids_to_names(
    xds: xr.Dataset,
    ant_xds: xr.Dataset,
    is_single_dish: bool,
    with_antenna_partitioning,
) -> xr.Dataset:
    """
    Turns the antenna_ids that we get from MSv2 into MSv4 antenna_name

    Parameters
    ----------
    xds: xr.Dataset
        A main xds (MSv4)
    ant_xds: xr.Dataset
        The antenna_xds for this MSv4
    is_single_dish: bool
        Whether a single-dish ("spectrum" data) dataset
    with_antenna_partitioning: bool
        Whether the MSv4 partitions include the antenna axis => only
        one antenna (and implicitly one 'baseline' - auto-correlation)

    Returns
    ----------
    xr.Dataset
        The main xds with antenna_id replaced with antenna_name
    """
    ant_xds = ant_xds.set_xindex(
        "antenna_id"
    )  # Allows for non-dimension coordinate selection.

    if not is_single_dish:  # Interferometer
        xds["baseline_antenna1_id"].data = ant_xds["antenna_name"].sel(
            antenna_id=xds["baseline_antenna1_id"].data
        )
        xds["baseline_antenna2_id"].data = ant_xds["antenna_name"].sel(
            antenna_id=xds["baseline_antenna2_id"].data
        )
        xds = xds.rename(
            {
                "baseline_antenna1_id": "baseline_antenna1_name",
                "baseline_antenna2_id": "baseline_antenna2_name",
            }
        )
    else:
        if not with_antenna_partitioning:
            # baseline_antenna1_id will be removed soon below, but it is useful here to know the actual antenna_ids,
            # as opposed to the baseline_ids which can mismatch when data is missing for some antennas
            xds["baseline_id"] = ant_xds["antenna_name"].sel(
                antenna_id=xds["baseline_antenna1_id"]
            )
        else:
            xds["baseline_id"] = ant_xds["antenna_name"]

        unwanted_coords_from_ant_xds = [
            "antenna_id",
            "antenna_name",
            "mount",
            "station_name",
        ]
        for unwanted_coord in unwanted_coords_from_ant_xds:
            xds = xds.drop_vars(unwanted_coord)

        # Rename a dim coord started generating warnings (index not re-created). Swap dims, create coord
        # https://github.com/pydata/xarray/pull/6999
        xds = xds.swap_dims({"baseline_id": "antenna_name"})
        xds = xds.assign_coords({"antenna_name": xds["baseline_id"].data})
        xds = xds.drop_vars("baseline_id")

        # drop more vars that seem unwanted in main_sd_xds, but there should be a better way
        # of not creating them in the first place
        unwanted_coords_sd = ["baseline_antenna1_id", "baseline_antenna2_id"]
        for unwanted_coord in unwanted_coords_sd:
            xds = xds.drop_vars(unwanted_coord)

    return xds


def add_group_to_data_groups(
    data_groups: dict, what_group: str, correlated_data_name: str, uvw: bool = True
):
    """
    Adds one correlated_data variable to the data_groups dict.
    A utility function to use when creating/updating data_groups from MSv2 data columns
    / data variables.

    Parameters
    ----------
    data_groups: str
        The data_groups dict of an MSv4 xds. It is updated in-place
    what_group: str
        Name of the data group: "base", "corrected", "model", etc.
    correlated_data_name: str
        Name of the correlated_data var: "VISIBILITY", "VISIBILITY_CORRECTED", "SPECTRUM", etc.
    uvw: bool
        Whether to add a uvw field to the data group (assume True = interferometric data).
    """
    data_groups[what_group] = {
        "correlated_data": correlated_data_name,
        "flag": "FLAG",
        "weight": "WEIGHT",
        "field_and_source": f"field_and_source_{what_group}_xds",
        "description": f"Data group derived from the data column '{correlated_data_name}' of an MSv2 converted to MSv4",
        "date": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if uvw:
        data_groups[what_group]["uvw"] = "UVW"


def add_data_groups(xds):
    xds.attrs["data_groups"] = {}

    data_groups = xds.attrs["data_groups"]
    if "VISIBILITY" in xds:
        add_group_to_data_groups(data_groups, "base", "VISIBILITY")

    if "VISIBILITY_CORRECTED" in xds:
        add_group_to_data_groups(data_groups, "corrected", "VISIBILITY_CORRECTED")

    if "VISIBILITY_MODEL" in xds:
        add_group_to_data_groups(data_groups, "model", "VISIBILITY_MODEL")

    is_single_dish = False
    if "SPECTRUM" in xds:
        add_group_to_data_groups(data_groups, "base", "SPECTRUM", False)
        is_single_dish = True

    if "SPECTRUM_MODEL" in xds:
        add_group_to_data_groups(data_groups, "model", "SPECTRUM_MODEL", False)
        is_single_dish = True

    if "SPECTRUM_CORRECTED" in xds:
        add_group_to_data_groups(data_groups, "corrected", "SPECTRUM_CORRECTED", False)
        is_single_dish = True

    return xds, is_single_dish
