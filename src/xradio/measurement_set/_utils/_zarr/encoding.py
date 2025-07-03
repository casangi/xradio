def add_encoding(xds, compressor, chunks=None):
    if chunks is None:
        chunks = xds.sizes

    chunks = {**dict(xds.sizes), **chunks}  # Add missing sizes if presents.

    for da_name in list(xds.data_vars):
        da_chunks = [chunks[dim_name] for dim_name in xds[da_name].sizes]
        xds[da_name].encoding = {"compressor": compressor, "chunks": da_chunks}
