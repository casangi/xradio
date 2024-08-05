import dask.array as da


def add_encoding(xds, compressor, chunks=None):
    if chunks is None:
        chunks = xds.sizes

    chunks = {**dict(xds.sizes), **chunks}  # Add missing sizes if presents.

    encoding = {}
    for da_name in list(xds.data_vars):
        if isinstance(xds[da_name].data, da.Array):
            da_chunks = dict(xds[da_name].chunksizes)

            # Loop over dimensions that can be chunked
            for dim_name in chunks.keys():
                # Only add user specified chunks to dimensions that aren't already chunked
                # ie don't touch time on VISIBILITY when using `read_col_conversion_dask`
                if len(da_chunks.get(dim_name, (1,))):
                    da_chunks[dim_name] = chunks[dim_name]

            xds[da_name].encoding = {"compressor": compressor, "chunks": da_chunks}

        else:
            for da_name in list(xds.data_vars):
                da_chunks = [chunks[dim_name] for dim_name in xds[da_name].sizes]
                xds[da_name].encoding = {"compressor": compressor, "chunks": da_chunks}
