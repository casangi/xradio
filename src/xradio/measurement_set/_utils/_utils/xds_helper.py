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
