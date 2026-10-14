import copy
import logging
import os

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from xradio._utils.logging import xradio_logger
from xradio._utils.zarr.config import ZARR_FORMAT
from xradio.image._util._zarr.common import _top_level_sub_xds

# Encodings inherited from a source zarr store that describe its chunking;
# they are dropped before writing, so that the new store's chunks follow the
# dask chunks of the data being written (or the chunks passed to write_image)
_CHUNK_ENCODINGS = ("chunks", "preferred_chunks", "shards")

#: Default compressor of write_image's zarr stores. Byte shuffle matters for
#: float pixels: without it Blosc barely compresses them.
DEFAULT_COMPRESSOR = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def _check_zarr_options(
    xds: xr.Dataset, chunks: dict | None, shards: dict | None, compressor
) -> None:
    """Raise for chunks, shards or a compressor that write_image cannot apply."""
    if compressor is not None and not isinstance(
        compressor, zarr.abc.codec.BytesBytesCodec
    ):
        raise TypeError(
            "compressor must be a zarr v3 bytes-to-bytes codec "
            f"(zarr.abc.codec.BytesBytesCodec) or None, not {type(compressor).__name__}"
        )
    if shards is not None and chunks is None:
        raise ValueError("shards needs chunks: the length of the chunks in a shard")
    for kind, lengths in (("chunks", chunks), ("shards", shards)):
        for dim, length in (lengths or {}).items():
            if dim not in xds.dims:
                raise ValueError(f"{kind}: the dataset has no dimension {dim!r}")
            if not (
                isinstance(length, int | np.integer) and (length > 0 or length == -1)
            ):
                raise ValueError(
                    f"{kind}: the length of {dim!r} must be a positive integer "
                    f"or -1 (the whole axis), not {length!r}"
                )


def _chunk_shape(dims: tuple, shape: tuple, chunks: dict) -> tuple[int, ...]:
    """Zarr chunk shape of a variable: a dimension missing from chunks, or
    given as -1, is one chunk over the whole axis."""
    return tuple(
        max(min(size if chunks.get(dim, -1) == -1 else int(chunks[dim]), size), 1)
        for dim, size in zip(dims, shape, strict=True)
    )


def _shard_shape(
    name: str, dims: tuple, shape: tuple, chunk_shape: tuple, shards: dict
) -> tuple[int, ...]:
    """Zarr shard shape of a variable: a dimension missing from shards is not
    sharded (its shard length is the chunk length, as in zarr's automatic
    shards), and -1, or a length of at least the axis, is one shard of whole
    chunks covering the axis."""
    shard_shape = []
    for dim, size, chunk in zip(dims, shape, chunk_shape, strict=True):
        shard = int(shards.get(dim, chunk))
        if shard == -1 or shard >= size:
            shard = max(-(-size // chunk), 1) * chunk
        elif shard % chunk:
            raise ValueError(
                f"The shard length {shard} of dimension {dim!r} of data "
                f"variable {name} is not a multiple of its chunk length {chunk}"
            )
        shard_shape.append(shard)
    return tuple(shard_shape)


def _align_dask_chunks(
    name: str, data: da.Array, dims: tuple, write_shape: tuple, unit: str
) -> da.Array:
    """Rechunk the axes of data whose dask chunks do not start on boundaries
    of write_shape (zarr chunks, or shards, which are written as a unit, so
    two dask chunks in one shard would overwrite each other's chunks). A
    misaligned axis gets the multiple of write_shape closest to its largest
    dask chunk."""
    rechunk = {}
    for axis, (sizes, length) in enumerate(zip(data.chunks, write_shape, strict=True)):
        if any(start % length for start in np.cumsum(sizes[:-1])):
            rechunk[axis] = max(1, round(max(sizes) / length)) * length
    if not rechunk:
        return data
    xradio_logger().info(
        f"Rechunking data variable {name} along "
        + ", ".join(f"{dims[axis]} to {length}" for axis, length in rechunk.items())
        + f": its dask chunks must be whole numbers of zarr {unit}s. Open the "
        f"image with such chunks to avoid this."
    )
    return data.rechunk(rechunk)


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


def _prepare_variables_for_zarr(
    xds: xr.Dataset,
    zarr_format: int,
    compressor,
    chunks: dict | None = None,
    shards: dict | None = None,
) -> None:
    """Adapt the variables of xds (a shallow copy of the dataset to write) to
    the store being written: drop inherited chunk encodings, rechunk dask
    arrays whose chunks zarr cannot store to uniform chunks, set the chunks
    and shards of the data variables (if given; their dask chunks are then
    aligned to them instead), set the compressor, and drop inherited codecs
    that do not apply to zarr_format (for example zarr v2 codecs of a store
    written by an older xradio, when writing zarr v3; a zarr v3 compressor
    then gives zarr's default)."""
    for name, variable in xds.variables.items():
        encoding = variable.encoding
        layout = chunks is not None and name in xds.data_vars
        if isinstance(variable.data, da.Array) or layout:
            for key in _CHUNK_ENCODINGS:
                encoding.pop(key, None)
        if layout:
            encoding["chunks"] = _chunk_shape(variable.dims, variable.shape, chunks)
            write_shape, unit = encoding["chunks"], "chunk"
            if shards is not None:
                encoding["shards"] = _shard_shape(
                    name, variable.dims, variable.shape, encoding["chunks"], shards
                )
                write_shape, unit = encoding["shards"], "shard"
            if isinstance(variable.data, da.Array):
                variable.data = _align_dask_chunks(
                    name, variable.data, variable.dims, write_shape, unit
                )
        elif isinstance(variable.data, da.Array):
            rechunk = _uniform_chunks(variable.data.chunks)
            if rechunk:
                variable.data = variable.data.rechunk(rechunk)
        encoding["compressors"] = () if compressor is None else (compressor,)
        for key in ("compressor", "compressors", "filters", "serializer"):
            if key in encoding and (
                (key == "compressor" and zarr_format != 2)
                or not _codecs_compatible(encoding, key, zarr_format)
            ):
                del encoding[key]


def _write_zarr(
    xds: xr.Dataset,
    zarr_store: str,
    chunks: dict | None = None,
    shards: dict | None = None,
    compressor=DEFAULT_COMPRESSOR,
):
    _check_zarr_options(xds, chunks, shards, compressor)
    max_chunk_size = 0.95 * 2**30
    for dv in xds.data_vars:
        obj = xds[dv]
        if chunks is not None:
            # the compressor encodes the zarr chunk (the inner chunk of a shard)
            chunk_shape = _chunk_shape(obj.dims, obj.shape, chunks)
            what = "the chunks passed to write_image"
        elif isinstance(obj.data, da.Array):
            # get chunk size to make sure it is small enough to be compressed
            # (dask's chunksize is the largest chunk along each axis, which
            # the uniform rechunking below does not change)
            chunk_shape = obj.data.chunksize
            what = "the chunk size of the dask array in the data variable"
        else:
            continue
        chunk_size_bytes = np.prod(chunk_shape) * np.dtype(obj.dtype).itemsize
        if chunk_size_bytes > max_chunk_size:
            raise ValueError(
                f"Chunk size of {chunk_size_bytes / 1e9} GB for data variable {dv} "
                f"bytes is too large for compression. To fix this, reduce {what} "
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
    _prepare_variables_for_zarr(xds_copy, ZARR_FORMAT, compressor, chunks, shards)
    sub_xds_dict = _encode(xds_copy, zarr_store)
    xds_copy.to_zarr(store=zarr_store, compute=True, zarr_format=ZARR_FORMAT)
    if sub_xds_dict:
        _write_sub_xdses(sub_xds_dict, compressor)


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


def _write_sub_xdses(sub_xds: dict, compressor):
    for k, v in sub_xds.items():
        # v is a deep copy (see _encode_dict), so it can be adapted in place
        _prepare_variables_for_zarr(v, ZARR_FORMAT, compressor)
        v.to_zarr(store=k, compute=True, zarr_format=ZARR_FORMAT)
