import copy
import logging
import os

import dask.array as da
import numpy as np
import xarray as xr

from xradio._utils.zarr.config import ZARR_FORMAT
from xradio.image._util._zarr.common import _top_level_sub_xds

# Encodings inherited from a source zarr store that describe its chunking;
# they are dropped before writing, so that the new store's chunks follow the
# dask chunks of the data being written
_CHUNK_ENCODINGS = ("chunks", "preferred_chunks", "shards")


def _uniform_chunks(chunks: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    """Return {axis: chunk size} for the axes whose dask chunks zarr cannot
    store (zarr needs equal chunks, except for a smaller last chunk), with
    the largest chunk of the axis as the new uniform size (so the largest
    chunk does not grow)."""
    rechunk = {}
    for axis, sizes in enumerate(chunks):
        if len(sizes) > 1 and (len(set(sizes[:-1])) > 1 or sizes[-1] > sizes[0]):
            rechunk[axis] = max(sizes)
    return rechunk


def _codecs_compatible(encoding: dict, key: str, zarr_format: int) -> bool:
    """Whether the codecs in encoding[key] (inherited from the source store)
    can be used to write a store of zarr_format."""
    value = encoding[key]
    if value is None or isinstance(value, str):
        return True
    codecs = value if isinstance(value, list | tuple) else (value,)
    if zarr_format == 2:
        import numcodecs.abc

        return key != "serializer" and all(
            isinstance(codec, numcodecs.abc.Codec) for codec in codecs
        )
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec

    expected = {
        "compressors": BytesBytesCodec,
        "filters": ArrayArrayCodec,
        "serializer": ArrayBytesCodec,
    }.get(key)
    return expected is not None and all(isinstance(codec, expected) for codec in codecs)


def _prepare_variables_for_zarr(xds: xr.Dataset, zarr_format: int) -> None:
    """Adapt the variables of xds (a shallow copy of the dataset to write) to
    the store being written: drop inherited chunk encodings, rechunk dask
    arrays whose chunks zarr cannot store to uniform chunks, and drop
    inherited codecs that do not apply to zarr_format (for example zarr v2
    codecs of a store written by an older xradio, when writing zarr v3)."""
    for variable in xds.variables.values():
        encoding = variable.encoding
        if isinstance(variable.data, da.Array):
            for key in _CHUNK_ENCODINGS:
                encoding.pop(key, None)
            rechunk = _uniform_chunks(variable.data.chunks)
            if rechunk:
                variable.data = variable.data.rechunk(rechunk)
        for key in ("compressor", "compressors", "filters", "serializer"):
            if key in encoding and (
                (key == "compressor" and zarr_format != 2)
                or not _codecs_compatible(encoding, key, zarr_format)
            ):
                del encoding[key]


def _write_zarr(xds: xr.Dataset, zarr_store: str):
    max_chunk_size = 0.95 * 2**30
    for dv in xds.data_vars:
        obj = xds[dv]
        if isinstance(obj, xr.core.dataarray.DataArray) and isinstance(
            obj.data, da.Array
        ):
            # get chunk size to make sure it is small enough to be compressed
            # (dask's chunksize is the largest chunk along each axis, which
            # the uniform rechunking below does not change)
            ary = obj.data
            chunk_size_bytes = np.prod(ary.chunksize) * np.dtype(ary.dtype).itemsize
            if chunk_size_bytes > max_chunk_size:
                raise ValueError(
                    f"Chunk size of {chunk_size_bytes / 1e9} GB for data variable {dv} "
                    "bytes is too large for compression. To fix this, "
                    "reduce the chunk size of the dask array in the data variable "
                    f"by at least a factor of {chunk_size_bytes / max_chunk_size}."
                )
    # _encode only mutates dataset and data variable attrs, so shallow copy
    # the dataset (sharing the pixel data buffers) and deep copy just the
    # attrs; a deep dataset copy would duplicate every data array in memory.
    # The shallow copy has its own variable objects and encoding dicts, so
    # adapting them does not change the caller's dataset.
    xds_copy = xds.copy(deep=False)
    xds_copy.attrs = copy.deepcopy(xds.attrs)
    for dv in xds_copy.data_vars:
        xds_copy[dv].attrs = copy.deepcopy(xds_copy[dv].attrs)
    _prepare_variables_for_zarr(xds_copy, ZARR_FORMAT)
    sub_xds_dict = _encode(xds_copy, zarr_store)
    xds_copy.to_zarr(store=zarr_store, compute=True, zarr_format=ZARR_FORMAT)
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
        # v is a deep copy (see _encode_dict), so it can be adapted in place
        _prepare_variables_for_zarr(v, ZARR_FORMAT)
        v.to_zarr(store=k, compute=True, zarr_format=ZARR_FORMAT)
