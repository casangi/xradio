import dask.array as da
import numpy as np
import os
import xarray as xr
from .common import __np_types, __top_level_sub_xds
from ..common import __dask_arrayize, __numpy_arrayize


def __read_zarr(zarr_store: str, do_dask: bool) -> xr.Dataset:
    tmp_xds = xr.open_zarr(zarr_store)
    xds = __decode(tmp_xds, zarr_store, do_dask)
    if do_dask:
        return __dask_arrayize(xds)
    else:
        return __numpy_arrayize(xds)


def __decode(xds: xr.Dataset, zarr_store: str, do_dask: bool) -> xr.Dataset:
    xds.attrs = __decode_dict(xds.attrs, "")
    sub_xdses = __decode_sub_xdses(zarr_store, do_dask)
    for k, v in sub_xdses.items():
        xds.attrs[k] = v
    return xds


def __decode_dict(my_dict: dict, top_key: str) -> dict:
    for k, v in my_dict.items():
        if isinstance(v, dict):
            if (
                "__type" in v
                and v["__type"] == "numpy.ndarray"
                and "__value" in v
                and "__dtype" in v
            ):
                my_dict[k] = np.array(v["__value"], dtype=__np_types[v["__dtype"]])
            else:
                z = os.sep.join([top_key, k]) if top_key else k
                my_dict[k] = __decode_dict(v, z)
    return my_dict


def __decode_sub_xdses(zarr_store: str, do_dask: bool) -> dict:
    sub_xdses = {}
    for root, dirs, files in os.walk(zarr_store):
        for d in dirs:
            if d.startswith(__top_level_sub_xds):
                xds = __read_zarr(os.sep.join([root, d]), do_dask)
                for k, v in xds.data_vars.items():
                    xds = xds.drop_vars([k]).assign({k: v.compute()})
                k = d[len(__top_level_sub_xds) + 2 :]
                sub_xdses[k] = xds
    return sub_xdses
