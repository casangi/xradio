"""Shared zarr v3 encoding utilities (compressor + sharding)."""

import zarr.abc.codec
import xarray as xr


def add_encoding(
    xds: xr.Dataset,
    compressor: zarr.abc.codec.BytesBytesCodec,
    chunks: dict | None = None,
    shards: dict | int | None = None,
) -> None:
    """Set zarr encoding on every data variable in *xds*.

    Parameters
    ----------
    xds :
        Dataset whose data variables will have their ``.encoding`` modified in-place.
    compressor :
        A zarr v3 ``BytesBytesCodec`` (e.g. ``zarr.codecs.ZstdCodec(level=2)``)
        used for compressing inner chunks.
    chunks :
        Inner-chunk sizes keyed by dimension name.  Missing dimensions default
        to the full axis length.  When *shards* is ``None`` these are the
        on-disk chunk sizes; when *shards* is provided they are the inner-chunk
        sizes inside each shard.
    shards :
        Controls zarr v3 sharding.

        - ``dict[str, int]`` — per-dimension absolute shard sizes, keyed by
          dimension name (same keys as *chunks*).  Dimensions absent from the
          dict default to the full axis length.
        - ``int`` — uniform factor applied to every dimension:
          ``shard_size = factor × chunk_size``.  Must be a positive integer;
          divisibility is guaranteed by construction.
        - ``None`` (default) — no sharding.

    .. note::

       This function mutates *xds* in-place by setting the ``.encoding``
       attribute on each data variable.  It does **not** return a new dataset.
    """
    # Fill in any missing dimensions with full axis lengths.
    chunks = {**dict(xds.sizes), **(chunks or {})}

    for da_name in list(xds.data_vars):
        da_chunks = [chunks[dim_name] for dim_name in xds[da_name].sizes]
        encoding = {"chunks": da_chunks, "compressors": (compressor,)}

        if shards is not None:
            if isinstance(shards, int):
                if shards < 1:
                    raise ValueError(
                        f"Shard factor must be a positive integer, got {shards}."
                    )
                # Note: shard_size may exceed the axis length (e.g.
                # axis_len=6, chunk=4, factor=3 → shard=12).  zarr v3 allows
                # shards larger than the array dimension.
                shard_shape = [c * shards for c in da_chunks]
            else:
                shard_shape = [
                    shards.get(dim, xds.sizes[dim]) for dim in xds[da_name].dims
                ]
                for dim, inner, outer in zip(xds[da_name].dims, da_chunks, shard_shape):
                    if outer % inner != 0:
                        raise ValueError(
                            f'Shard size {outer} for dimension "{dim}" must be an '
                            f"exact multiple of the inner chunk size {inner}."
                        )
            encoding["shards"] = shard_shape

        xds[da_name].encoding = encoding
