import numpy as np
import xarray as xr
import os
from .common import _np_types, _top_level_sub_xds


def _write_zarr(xds: xr.Dataset, zarr_store: str):
    xds_copy = xds.copy(deep=True)
    xds_copy, xds_dict = _encode(xds_copy)
    z_obj = xds_copy.to_zarr(store=zarr_store, compute=True)
    _write_sub_xdses(zarr_store, xds_dict, _top_level_sub_xds)


def _encode(xds: xr.Dataset):
    # encode attrs
    xds.attrs, xds_dict = _encode_dict(xds.attrs)
    return xds, xds_dict


def _encode_dict(my_dict: dict, top_key="") -> tuple:
    xds_dict = {}
    del_keys = []
    for k, v in my_dict.items():
        if isinstance(v, dict):
            z = os.sep.join([top_key, k]) if top_key else k
            my_dict[k], ret_xds_dict = _encode_dict(v, z)
            if ret_xds_dict:
                xds_dict[k] = ret_xds_dict
        elif isinstance(v, np.ndarray):
            my_dict[k] = {}
            my_dict[k]["_type"] = "numpy.ndarray"
            my_dict[k]["_value"] = v.tolist()
            my_dict[k]["_dtype"] = str(v.dtype)
        elif isinstance(v, xr.Dataset):
            xds_dict[k] = v.copy(deep=True)
            del_keys.append(k)
    for k in del_keys:
        del my_dict[k]
    return my_dict, xds_dict


def _write_sub_xdses(zarr_store: str, xds_dict: dict, path: str):
    for k, v in xds_dict.items():
        my_path = f"{path}_{k}" if path else f"{k}"
        if isinstance(v, dict):
            _write_sub_xdses(zarr_store, xds_dict[k], my_path)
        elif isinstance(v, xr.Dataset):
            zs = os.sep.join([zarr_store, my_path])
            z_obj = v.to_zarr(store=zs, compute=True)
