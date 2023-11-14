import dask.array as da
import numpy as np
import os
import xarray as xr
from .common import _np_types, _top_level_sub_xds
from ..common import _dask_arrayize, _numpy_arrayize


def _read_zarr(zarr_store: str, do_dask: bool) -> xr.Dataset:
    tmp_xds = xr.open_zarr(zarr_store)
    xds = _decode(tmp_xds, zarr_store, do_dask)
    if do_dask:
        return _dask_arrayize(xds)
    else:
        return _numpy_arrayize(xds)


def _decode(xds: xr.Dataset, zarr_store: str, do_dask: bool) -> xr.Dataset:
    xds.attrs = _decode_dict(xds.attrs, "")
    sub_xdses = _decode_sub_xdses(zarr_store, do_dask)
    for k, v in sub_xdses.items():
        xds.attrs[k] = v
    return xds


def _decode_dict(my_dict: dict, top_key: str) -> dict:
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


def _decode_sub_xdses(zarr_store: str, do_dask: bool) -> dict:
    sub_xdses = {}
    for root, dirs, files in os.walk(zarr_store):
        for d in dirs:
            if d.startswith(_top_level_sub_xds):
                xds = _read_zarr(os.sep.join([root, d]), do_dask)
                for k, v in xds.data_vars.items():
                    xds = xds.drop_vars([k]).assign({k: v.compute()})
                k = d[len(_top_level_sub_xds) + 1 :]
                sub_xdses[k] = xds
    return sub_xdses
