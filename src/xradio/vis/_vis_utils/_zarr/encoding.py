def add_encoding(xds, compressor, chunks=None):
    if chunks is None:
        chunks = xds.dims

    chunks = {**dict(xds.dims), **chunks}  # Add missing dims if presents.

    encoding = {}
    for da_name in list(xds.data_vars):
        if chunks:
            da_chunks = [chunks[dim_name] for dim_name in xds[da_name].dims]
            xds[da_name].encoding = {"compressor": compressor, "chunks": da_chunks}
            # print(xds[da_name].encoding)
        else:
            xds[da_name].encoding = {"compressor": compressor}
