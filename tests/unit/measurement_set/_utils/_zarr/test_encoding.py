import numcodecs
import xarray as xr

from xradio.measurement_set._utils._zarr.encoding import add_encoding

single_encoding = numcodecs.Zstd(level=2)


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

    add_encoding(xds, single_encoding)
    assert xds
    assert xds.da.encoding == {"compressor": single_encoding, "chunks": [3]}


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

    chunks_size = 1
    add_encoding(xds, single_encoding, chunks={"zz_not_there": chunks_size})
    assert xds
    assert xds.da.encoding == {"compressor": single_encoding, "chunks": [3]}


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

    chunks_size = 1
    add_encoding(xds, single_encoding, chunks={"x": chunks_size})
    assert xds
    assert xds.da.encoding == {"compressor": single_encoding, "chunks": [chunks_size]}
