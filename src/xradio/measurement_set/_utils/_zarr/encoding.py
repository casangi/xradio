import zarr.codecs


def add_encoding(
    xds,
    compressor,
    chunks: dict | None = None,
    shards: dict | int | None = None,
) -> None:
    """Set zarr encoding on every data variable in *xds*.

    Parameters
    ----------
    xds :
        Dataset whose data variables will have their ``.encoding`` set in-place.
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
    """
    if chunks is None:
        chunks = xds.sizes

    chunks = {**dict(xds.sizes), **chunks}  # Add missing sizes if present.

    for da_name in list(xds.data_vars):
        da_chunks = [chunks[dim_name] for dim_name in xds[da_name].sizes]

        if shards is None:
            xds[da_name].encoding = {"compressors": (compressor,), "chunks": da_chunks}
        else:
            if isinstance(shards, int):
                if shards < 1:
                    raise ValueError(
                        f"Shard factor must be a positive integer, got {shards}."
                    )
                shard_shape = [c * shards for c in da_chunks]
            else:
                shard_shape = [
                    shards.get(dim_name, xds.sizes[dim_name])
                    for dim_name in xds[da_name].sizes
                ]
                # Validate: inner chunks must divide evenly into shards.
                for dim_name, inner, outer in zip(
                    xds[da_name].dims, da_chunks, shard_shape
                ):
                    if outer % inner != 0:
                        raise ValueError(
                            f'Shard size {outer} for dimension "{dim_name}" must be an '
                            f"exact multiple of the inner chunk size {inner}."
                        )
            # Each Dask write task must map to exactly one shard to avoid concurrent
            # read-modify-write races on the same shard file.  Only rechunk when the
            # existing Dask chunk shape does not already match the shard shape; an
            # unnecessary rechunk inserts merge tasks in the graph and spikes memory.
            arr = xds[da_name].data
            already_aligned = hasattr(arr, "chunks") and all(
                # Every Dask chunk along this axis equals shard_size (the last
                # chunk may be smaller at the array edge — that is fine).
                all(c == s or c == xds.sizes[dim] % s for c in arr.chunks[i])
                for i, (dim, s) in enumerate(zip(xds[da_name].dims, shard_shape))
            )
            if not already_aligned:
                shard_rechunk = dict(zip(xds[da_name].dims, shard_shape))
                xds[da_name] = xds[da_name].chunk(shard_rechunk)
            xds[da_name].encoding = {
                "codecs": [
                    zarr.codecs.ShardingCodec(
                        chunk_shape=tuple(da_chunks),
                        codecs=[zarr.codecs.BytesCodec(), compressor],
                    )
                ],
                "chunks": shard_shape,
            }
