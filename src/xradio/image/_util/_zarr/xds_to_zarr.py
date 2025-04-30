import dask.array as da
import logging
import numpy as np
import xarray as xr
import os
from .common import _np_types, _top_level_sub_xds


def _write_zarr(xds: xr.Dataset, zarr_store: str):
    max_chunk_size = 0.95 * 2**30
    for dv in xds.data_vars:
        obj = xds[dv]
        if isinstance(obj, xr.core.dataarray.DataArray) and isinstance(
            obj.data, da.Array
        ):
            # get chunk size to make sure it is small enough to be compressed
            ary = obj.data
            chunk_size_bytes = np.prod(ary.chunksize) * np.dtype(ary.dtype).itemsize
            if chunk_size_bytes > max_chunk_size:
                raise ValueError(
                    f"Chunk size of {chunk_size_bytes/1e9} GB for data variable {dv} "
                    "bytes is too large for compression. To fix this, "
                    "reduce the chunk size of the dask array in the data variable "
                    f"by at least a factor of {chunk_size_bytes/max_chunk_size}."
                )
    xds_copy = xds.copy(deep=True)
    sub_xds_dict = _encode(xds_copy, zarr_store)
    z_obj = xds_copy.to_zarr(store=zarr_store, compute=True)
    if sub_xds_dict:
        _write_sub_xdses(sub_xds_dict)


def _encode(xds: xr.Dataset, top_path: str) -> dict:
    # encode attrs
    sub_xds_dict = {}
    _encode_dict(xds.attrs, top_path, sub_xds_dict)
    for dv in xds.data_vars:
        _encode_dict(xds[dv].attrs, os.sep.join([top_path, dv]), sub_xds_dict)
    logging.debug(f"Encoded sub_xds_dict: {sub_xds_dict}")
    return sub_xds_dict


def _encode_dict(my_dict: dict, top_path: str, sub_xds_dict) -> tuple:
    del_keys = []
    for k, v in my_dict.items():
        if isinstance(v, dict):
            z = os.sep.join([top_path, k])
            _encode_dict(v, z, sub_xds_dict)
        elif isinstance(v, np.ndarray):
            my_dict[k] = {}
            my_dict[k]["_type"] = "numpy.ndarray"
            my_dict[k]["_value"] = v.tolist()
            my_dict[k]["_dtype"] = str(v.dtype)
        elif isinstance(v, xr.Dataset):
            sub_xds_dict[os.sep.join([top_path, f"{_top_level_sub_xds}{k}"])] = v.copy(
                deep=True
            )
            del_keys.append(k)
    for k in del_keys:
        del my_dict[k]


def _write_sub_xdses(sub_xds: dict):
    for k, v in sub_xds.items():
        z_obj = v.to_zarr(store=k, compute=True)
