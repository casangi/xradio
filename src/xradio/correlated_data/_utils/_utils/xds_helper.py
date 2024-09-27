from importlib.metadata import version
import toolviper.utils.logger as logger, multiprocessing, psutil
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from .cds import CASAVisSet
from .stokes_types import stokes_types
from xradio._utils.list_and_array import get_pad_value


def make_coords(
    xds: xr.Dataset, ddi: int, subtables: Tuple[xr.Dataset, ...]
) -> Dict[str, np.ndarray]:
    """
    Make the coords to be added to a partition or chunk (besides
    the time, baseline) basic structure

    Grabs:
    - channel (center) frequency values from the spw subtable
    - pol idxs from the pol+ddi subtables -> pol names via the stokes_types
    - antenna IDs from antenna subtable

    Parameters
    ----------
    xds : xr.Dataset

    ddi : int

    subtables: Tuple[xr.Dataset, ...]


    Returns
    -------
    Dict[str, np.ndarray]
    """
    ant_xds, ddi_xds, spw_xds, pol_xds = subtables
    freq = spw_xds.CHAN_FREQ.values[
        ddi_xds.SPECTRAL_WINDOW_ID.values[ddi], : xds.freq.shape[0]
    ]
    pol_ids = pol_xds.CORR_TYPE.values[
        ddi_xds.POLARIZATION_ID.values[ddi], : xds.pol.shape[0]
    ]
    pol_names = np.vectorize(stokes_types.get)(pol_ids)
    ant_id = ant_xds.antenna_id.values
    coords = {
        "freq": freq,
        "pol": pol_names,
        "antenna_id": ant_id,
        # These will be metainfo in partitions
        # "spw_id": [ddi_xds["spectral_window_id"].values[ddi]],
        # "pol_id": [ddi_xds["polarization_id"].values[ddi]],
    }
    return coords


def vis_xds_packager_cds(
    subtables: List[Tuple[str, xr.Dataset]],
    partitions: Dict[Any, xr.Dataset],
    descr_add: str = "",
) -> CASAVisSet:
    """
    Takes a a list of subtable xds datasets and a dictionary of data
    partition xds datasets and and packages them as a CASA vis dataset
    (cds)

    Parameters
    ----------
    partitions : List[Tuple[str, xr.Dataset]]
        data partiions as xds datasets
    subtables : Dict[Any, xr.Dataset]
        subtables as xds datasets
    descr_add : str (Default value = "")
        substring to add to the short descr string of the cds

    Returns
    -------
    CASAVisSet
        A "cds" - container for the metainfo subtables and data partitions
    """
    vers = version("xradio")

    return CASAVisSet(
        subtables,
        partitions,
        f"CASA vis set produced by xradio {vers}/{descr_add}",
    )


def vis_xds_packager_mxds(
    partitions: Dict[Any, xr.Dataset],
    subtables: List[Tuple[str, xr.Dataset]],
    add_global_coords: bool = True,
) -> xr.Dataset:
    """
    Takes a dictionary of data partition xds datasets and a list of
    subtable xds datasets and packages them as a dataset of datasets
    (mxds)

    Parameters
    ----------
    partitions : Dict[Any, xr.Dataset]
        data partiions as xds datasets
    subtables : List[Tuple[str, xr.Dataset]]
        subtables as xds datasets
        :add_global_coords: whether to add coords to the output mxds
    add_global_coords: bool (Default value = True)

    Returns
    -------
    xr.Dataset
        A "mxds" - xr.dataset of datasets
    """
    mxds = xr.Dataset(attrs={"metainfo": subtables, "partitions": partitions})

    if add_global_coords:
        mxds = mxds.assign_coords(make_global_coords(mxds))

    return mxds


def make_global_coords(mxds: xr.Dataset) -> Dict[str, xr.DataArray]:
    coords = {}
    metainfo = mxds.attrs["metainfo"]
    if "antenna" in metainfo:
        coords["antenna_ids"] = metainfo["antenna"].antenna_id.values
        coords["antennas"] = xr.DataArray(
            metainfo["antenna"].NAME.values, dims=["antenna_ids"]
        )
    if "field" in metainfo:
        coords["field_ids"] = metainfo["field"].field_id.values
        coords["fields"] = xr.DataArray(
            metainfo["field"].NAME.values, dims=["field_ids"]
        )
    if "feed" in mxds.attrs:
        coords["feed_ids"] = metainfo["feed"].FEED_ID.values
    if "observation" in metainfo:
        coords["observation_ids"] = metainfo["observation"].observation_id.values
        coords["observations"] = xr.DataArray(
            metainfo["observation"].PROJECT.values, dims=["observation_ids"]
        )
    if "polarization" in metainfo:
        coords["polarization_ids"] = metainfo["polarization"].pol_setup_id.values
    if "source" in metainfo:
        coords["source_ids"] = metainfo["source"].SOURCE_ID.values
        coords["sources"] = xr.DataArray(
            metainfo["source"].NAME.values, dims=["source_ids"]
        )
    if "spectral_window" in metainfo:
        coords["spw_ids"] = metainfo["spectral_window"].spw_id.values
    if "state" in metainfo:
        coords["state_ids"] = metainfo["state"].STATE_ID.values

    return coords


def expand_xds(xds: xr.Dataset) -> xr.Dataset:
    """
    expand single (row) dimension of xds to (time, baseline)

    Parameters
    ----------
    xds : xr.Dataset
        "flat" dataset (with row dimension - without (time, baseline) dimensions)

    Returns
    -------
    xr.Dataset
        expanded dataset, with (time, baseline) dimensions
    """
    assert "baseline" not in xds.coords

    txds = xds.copy()

    unique_baselines, baselines = np.unique(
        [txds.baseline_ant1_id.values, txds.baseline_ant2_id.values],
        axis=1,
        return_inverse=True,
    )

    txds["baseline"] = xr.DataArray(baselines.astype("int32"), dims=["row"])

    try:
        txds = (
            txds.set_index(row=["time", "baseline"])
            .unstack("row")
            .transpose("time", "baseline", ...)
        )
        # unstack changes type to float when it needs to introduce NaNs, so
        # we need to reset to the proper type. Avoid if possible, as the
        # astype are costly
        for dv in txds.data_vars:
            if txds[dv].dtype != xds[dv].dtype:
                txds[dv] = txds[dv].astype(xds[dv].dtype)
    except Exception as exc:
        logger.warning(
            f"WARNING: Cannot expand rows to (time, baseline), "
            f"possibly duplicate values in (time, baseline). Exception: {exc}."
            f"\nDataset: {txds=}"
        )
        txds = xds.copy()

    return txds


def flatten_xds(xds: xr.Dataset) -> xr.Dataset:
    """
    flatten (time, baseline) dimensions of xds back to single dimension (row)

    Parameters
    ----------
    xds : xr.Dataset


    Returns
    -------
    xr.Dataset
        Dataset in flat form (back to 'row' dimension as read by casacore tables)
    """
    txds = xds.copy()

    # flatten the time x baseline dimensions of main table
    if ("time" in xds.sizes) and ("baseline" in xds.sizes):
        txds = xds.stack({"row": ("time", "baseline")}).transpose("row", ...)
        # compute for issue https://github.com/hainegroup/oceanspy/issues/332
        # drop=True silently does compute (or at least used to)

        fill_value_int32 = get_pad_value(np.int32)
        txds = txds.where(
            (
                (txds.STATE_ID != fill_value_int32)
                & (txds.FIELD_ID != fill_value_int32)
            ).compute(),
            drop=True,
        )  # .unify_chunks()

        # re-assigning (implicitly dropping index coords) one by one produces
        # DeprecationWarnings: https://github.com/pydata/xarray/issues/6505
        astyped_data_vars = dict(xds.data_vars)
        for dv in list(txds.data_vars):
            if txds[dv].dtype != xds[dv].dtype:
                astyped_data_vars[dv] = txds[dv].astype(xds[dv].dtype)
            else:
                astyped_data_vars[dv] = txds[dv]

        flat_xds = xr.Dataset(astyped_data_vars, coords=txds.coords, attrs=txds.attrs)
        flat_xds = flat_xds.reset_index(["time", "baseline"])

    else:
        flat_xds = txds

    return flat_xds


####################################
# xautomatically compute best data chunking
def optimal_chunking(
    ndim: Union[int, None] = None,
    didxs: Union[Tuple[int], List[int], None] = None,
    chunk_size: str = "auto",
    data_shape: Union[tuple, None] = None,
) -> tuple:
    """
    Determine the optimal chunk shape for reading an MS or Image based
    on machine resources and intended operations

    Parameters
    ----------
    ndim : Union[int, None] = None
        number of dimensions to chunk. An MS is 3, an
        expanded MS is 4. An image could be anywhere from 2 to 5. Not
        needed if data_shape is given.
    didxs : Union[Tuple[int], List[int], None] = None
        dimension indices over which subsequent operations
        will be performed. Values should be less than ndim. Tries to
        reduce inter-process communication of data contents. Needs to
        know the shape to do this well. Default None balances chunk size
        across all dimensions.
    chunk_size : str (Default value = "auto")
        target chunk size ('large', 'small', 'auto').
        Default 'auto' tries to guess by looking at CPU core count and
        available memory.
    data_shape : Union[tuple, None] = None
        shape of the total MS DDI or Image data. Helps
        to know. Default None does not optimize based on shape

    Returns
    -------
    tuple
        optimal chunking for reading the ms (row, chan, pol)
    """
    assert (ndim is not None) or (
        data_shape is not None
    ), "either ndim or data_shape must be given"
    assert chunk_size in ["large", "small", "auto"], "invalid chunk_size parameter"
    if ndim is None:
        ndim = len(data_shape)

    opt_dims = (
        didxs if (didxs is not None) and (len(didxs) > 0) else np.arange(ndim)
    )  # maximize these dim chunk sizes
    nonopt_dims = np.setdiff1d(np.arange(ndim), opt_dims)  # at the expense of these

    max_chunk_sizes = (
        data_shape
        if data_shape is not None
        else [dd for ii, dd in enumerate([10000, 10000, 10000, 4, 10]) if ii < ndim]
    )
    min_chunk_sizes = (
        np.ceil(np.array(data_shape) / 80).astype(int)
        if data_shape is not None
        else (
            [1000, 1, 1]
            if ndim == 3
            else [dd for ii, dd in enumerate([10, 10, 1, 1, 1]) if ii < ndim]
        )
    )
    target_size = 175 * 1024**2 / 8  # ~175 MB chunk worst case with 8-byte DATA column
    bytes_per_core = int(
        round(
            ((psutil.virtual_memory().available * 0.10) / multiprocessing.cpu_count())
        )
    )
    if data_shape is not None:
        bytes_per_core = min(
            bytes_per_core, np.prod(data_shape) * 8 / 2
        )  # ensure at least two chunks
    if chunk_size == "large":
        target_size = target_size * 6  # ~1 GB
    if chunk_size == "auto":
        target_size = max(min(target_size * 6, bytes_per_core / 8), target_size)

    # start by setting the optimized dims to their max size and non-optimized dims to their min size
    chunks = np.zeros((ndim), dtype="int")
    chunks[opt_dims] = np.array(max_chunk_sizes)[opt_dims]
    chunks[nonopt_dims] = np.array(min_chunk_sizes)[nonopt_dims]

    # iteratively walk towards an optimal chunk size
    # iteration is needed because rounding to nearest integer index can make a big different (2x) in chunk size
    # for small dimensions like pol
    for ii in range(10):
        # if the resulting size is too big, reduce the sizes of the optimized dimensions
        if (np.prod(chunks) > target_size) and (len(opt_dims) > 0):
            chunks[opt_dims] = np.round(
                chunks[opt_dims]
                * (target_size / np.prod(chunks)) ** (1 / len(opt_dims))
            )
        # else if the resulting size is too small, increase the sizes of the non-optimized dimensions
        elif (np.prod(chunks) < target_size) and (len(nonopt_dims) > 0):
            chunks[nonopt_dims] = np.round(
                chunks[nonopt_dims]
                * (target_size / np.prod(chunks)) ** (1 / len(nonopt_dims))
            )
        chunks = np.min((chunks, max_chunk_sizes), axis=0)
        chunks = np.max((chunks, min_chunk_sizes), axis=0)

    return tuple(chunks)


def calc_optimal_ms_chunk_shape(
    memory_available_in_bytes, shape, element_size_in_bytes, column_name
) -> int:
    """
    Calculates the max number of rows (1st dim in shape) of a variable
    that can be fit in the memory for a thread.

    Parameters
    ----------
    memory_available_in_bytes :

    shape :

    element_size_in_bytes :

    column_name :


    Returns
    -------
    int
    """
    factor = 0.8  # Account for memory used by other objects in thread.
    # total_mem = np.prod(shape)*element_size_in_bytes
    single_row_mem = np.prod(shape[1:]) * element_size_in_bytes

    if not single_row_mem < factor * memory_available_in_bytes:
        msg = (
            "Not engough memory in a thread to contain a row of "
            f"{column_name}. Need at least {single_row_mem / factor}"
            " bytes."
        )
        raise RuntimeError(msg)

    rows_chunk_size = int((factor * memory_available_in_bytes) / single_row_mem)

    if rows_chunk_size > shape[0]:
        rows_chunk_size = shape[0]

    logger.debug(
        "Numbers of rows in chunk for " + column_name + ": " + str(rows_chunk_size)
    )

    return rows_chunk_size
