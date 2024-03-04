import copy
import dask.array as da

# import graphviper.utils.logger as logger
import numpy as np
import os
import xarray as xr
from .common import _np_types, _top_level_sub_xds
from ..common import _coords_to_numpy, _dask_arrayize_dv, _numpy_arrayize_dv


def _read_zarr(
    zarr_store: str, output: dict, selection: dict = {}
) -> (xr.Dataset, bool):
    # supported key/values in output are:
    # "dv"
    #    what data variables should be returned as.
    #    "numpy": numpy arrays
    #    "dask": dask arrays
    # "coords"
    #    what coords should be returned as
    #    "numpy": numpy arrays
    # it's easiest just to copy the object rather than figuring out
    # if it should be copied at some point or not
    # xds = xr.open_zarr(zarr_store).isel(selection).copy(deep=True)
    xds = xr.open_zarr(zarr_store).isel(selection)
    do_dask = False
    do_numpy = False
    do_np_coords = False
    if "dv" in output:
        dv = output["dv"]
        if dv in ["dask", "numpy"]:
            do_dask = dv == "dask"
            do_numpy = not do_dask
        else:
            raise ValueError(
                f"Unsupported value {output[dv]} for output[dv]. "
                "Supported values are 'dask' and 'numpy'"
            )
    if "coords" in output:
        c = output["coords"]
        if c == "numpy":
            do_np_coords = True
        else:
            raise ValueError(
                f"Unexpected value {c} for output[coords]. "
                "The supported value is 'numpy'"
            )
    # do not pass selection, because that is only for the top level data vars
    xds = _decode(xds, zarr_store, output)
    if do_np_coords:
        xds = _coords_to_numpy(xds)
    if do_dask:
        xds = _dask_arrayize_dv(xds)
    elif do_numpy:
        xds = _numpy_arrayize_dv(xds)
    return xds


def _decode(xds: xr.Dataset, zarr_store: str, output: dict) -> (xr.Dataset, bool):
    xds.attrs = _decode_dict(xds.attrs, "")
    sub_xdses = _decode_sub_xdses(zarr_store, output)
    for k, v in sub_xdses.items():
        xds.attrs[k] = v
    return xds


def _decode_dict(my_dict: dict, top_key: str) -> dict:
    # Decodes numpy arrays
    my_dict
    for k, v in my_dict.items():
        if isinstance(v, dict):
            if (
                "_type" in v
                and v["_type"] == "numpy.ndarray"
                and "_value" in v
                and "_dtype" in v
            ):
                my_dict[k] = np.array(v["_value"], dtype=_np_types[v["_dtype"]])
            else:
                z = os.sep.join([top_key, k]) if top_key else k
                my_dict[k] = _decode_dict(v, z)
    return my_dict


def _decode_sub_xdses(zarr_store: str, output: dict) -> dict:
    sub_xdses = {}
    for root, dirs, files in os.walk(zarr_store):
        # top down walk
        for d in dirs:
            if d.startswith(_top_level_sub_xds):
                xds = _read_zarr(os.sep.join([root, d]), output)
                # for k, v in xds.data_vars.items():
                #    xds = xds.drop_vars([k]).assign({k: v.compute()})
                ky = d[len(_top_level_sub_xds) + 1 :]
                sub_xdses[ky] = xds
    return sub_xdses
