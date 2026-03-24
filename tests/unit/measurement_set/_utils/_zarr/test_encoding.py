import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr.codecs

from xradio.measurement_set._utils._zarr.encoding import add_encoding

compressor = zarr.codecs.ZstdCodec(level=2)


def _make_xds(time=6, freq=8) -> xr.Dataset:
    """Small 2-D dataset with a dask-backed data variable."""
    return xr.Dataset(
        {
            "vis": xr.DataArray(
                da.zeros((time, freq), chunks=(time, freq), dtype=np.complex64),
                dims=["time", "frequency"],
            ),
        }
    )


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

    add_encoding(xds, compressor)
    assert xds
    assert xds.da.encoding == {"compressors": (compressor,), "chunks": [3]}


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

    add_encoding(xds, compressor, chunks={"zz_not_there": 1})
    assert xds
    assert xds.da.encoding == {"compressors": (compressor,), "chunks": [3]}


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
    add_encoding(xds, compressor, chunks={"x": chunks_size})
    assert xds
    assert xds.da.encoding == {
        "compressors": (compressor,),
        "chunks": [chunks_size],
    }


def test_sharding_sets_shards_and_chunks():
    """With sharding, encoding has 'shards' (outer) and 'chunks' (inner)."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(
        xds,
        compressor,
        chunks={"time": 2, "frequency": 4},
        shards={"time": 6, "frequency": 8},
    )

    enc = xds["vis"].encoding
    assert enc["chunks"] == [2, 4]  # inner chunk shape
    assert enc["shards"] == [6, 8]  # outer shard shape
    assert enc["compressors"] == (compressor,)
    assert "codecs" not in enc


def test_sharding_inner_chunk_defaults_to_full_axis():
    """Omitting chunks= makes inner chunks equal to full axis (1 inner chunk/shard)."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(xds, compressor, shards={"time": 6, "frequency": 8})

    enc = xds["vis"].encoding
    assert enc["chunks"] == [6, 8]  # inner == full axis
    assert enc["shards"] == [6, 8]


def test_sharding_absent_dim_defaults_to_full_axis():
    """A dimension absent from shards= gets shard size == full axis length."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(
        xds, compressor, chunks={"time": 2, "frequency": 4}, shards={"time": 6}
    )

    enc = xds["vis"].encoding
    assert enc["chunks"] == [2, 4]  # inner chunks
    assert enc["shards"] == [6, 8]  # frequency shard spans full axis (8)


def test_sharding_compressor_always_present():
    """The compressor is in encoding even when sharding is enabled."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(
        xds,
        compressor,
        chunks={"time": 2, "frequency": 4},
        shards={"time": 6, "frequency": 8},
    )

    assert xds["vis"].encoding["compressors"] == (compressor,)


def test_sharding_raises_on_non_divisible_inner_chunk():
    """ValueError when inner chunk does not divide evenly into shard."""
    xds = _make_xds(time=6, freq=8)
    with pytest.raises(ValueError, match="exact multiple"):
        add_encoding(xds, compressor, chunks={"time": 4}, shards={"time": 6})


def test_no_sharding_unchanged_when_shards_is_none():
    """Passing shards=None produces plain encoding without 'shards' key."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(xds, compressor, chunks={"time": 2, "frequency": 4}, shards=None)

    enc = xds["vis"].encoding
    assert "shards" not in enc
    assert enc == {"compressors": (compressor,), "chunks": [2, 4]}


def test_sharding_factor_sets_shard_shape():
    """Integer factor: shard_size == factor * chunk_size for every dim."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(xds, compressor, chunks={"time": 2, "frequency": 2}, shards=3)

    enc = xds["vis"].encoding
    assert enc["chunks"] == [2, 2]  # inner chunks unchanged
    assert enc["shards"] == [6, 6]  # 3 * 2 = 6 for both dims
    assert enc["compressors"] == (compressor,)


def test_sharding_factor_shard_equals_factor_times_chunk():
    """Shard sizes are exactly factor * chunk for every dim, uncapped."""
    xds = _make_xds(time=6, freq=8)
    add_encoding(xds, compressor, chunks={"time": 2, "frequency": 2}, shards=2)

    enc = xds["vis"].encoding
    assert enc["chunks"] == [2, 2]  # inner
    assert enc["shards"] == [4, 4]  # 2*2=4 for both dims


def test_sharding_factor_divisibility_always_holds():
    """Factor path must never raise a divisibility error."""
    for factor in (1, 2, 3):
        xds = _make_xds(time=6, freq=8)
        add_encoding(xds, compressor, chunks={"time": 1, "frequency": 1}, shards=factor)
        assert xds["vis"].encoding["chunks"] == [1, 1]
        assert xds["vis"].encoding["shards"] == [factor, factor]


def test_sharding_factor_invalid_raises():
    """Non-positive factor must raise ValueError."""
    xds = _make_xds(time=6, freq=8)
    with pytest.raises(ValueError, match="positive integer"):
        add_encoding(xds, compressor, shards=0)
    with pytest.raises(ValueError, match="positive integer"):
        add_encoding(xds, compressor, shards=-2)
