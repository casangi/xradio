from importlib.metadata import version
import toolviper.utils.logger as logger, multiprocessing, psutil
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr

from .cds import CASAVisSet
from .stokes_types import stokes_types
from xradio._utils.list_and_array import get_pad_value


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
