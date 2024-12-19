import numcodecs
import xarray as xr

from xradio.measurement_set._utils._zarr.encoding import add_encoding


def test_add_encoding_wo_chunks():
    xds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                [0, 2, 3],
                coords=[[0, 1, 2]],
                dims=["x"],
            ),
            "bool_var": ("time", [True, False, True]),
        }
    )

    encoding = numcodecs.Zstd(level=2)

    add_encoding(xds, encoding)
    assert xds
    assert xds.da.encoding == {"compressor": encoding, "chunks": [3]}


def test_add_encoding_with_wrong_chunks():
    xds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                [0, 2, 3],
                coords=[[0, 1, 2]],
                dims=["x"],
            ),
            "bool_var": ("time", [True, False, True]),
        }
    )

    encoding = numcodecs.Zstd(level=2)

    chunks_size = 1
    add_encoding(xds, encoding, chunks={"zz_not_there": chunks_size})
    assert xds
    assert xds.da.encoding == {"compressor": encoding, "chunks": [3]}


def test_add_encoding_with_chunks():
    xds = xr.Dataset(
        data_vars={
            "da": xr.DataArray(
                [0, 2, 3],
                coords=[[0, 1, 2]],
                dims=["x"],
            ),
            "bool_var": ("time", [True, False, True]),
        }
    )

    encoding = numcodecs.Zstd(level=2)

    chunks_size = 1
    add_encoding(xds, encoding, chunks={"x": chunks_size})
    assert xds
    assert xds.da.encoding == {"compressor": encoding, "chunks": [chunks_size]}
