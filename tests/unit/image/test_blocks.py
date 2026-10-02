"""Tests of the region reader the CASA and FITS writers compute pixels with."""

import itertools
import os

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from xradio.image import open_image, write_image
from xradio.image._util import _blocks
from xradio.image._util._blocks import RegionReader

_DIMS = ("frequency", "polarization", "m", "l")


def _read_chunk(values: np.ndarray, start: int) -> np.ndarray:
    return values[start : start + 1].copy()


def reader_like_array(values: np.ndarray) -> da.Array:
    """A dask array built like the image readers build theirs: one delayed
    read (and one graph layer) per chunk along the first axis, concatenated."""
    pieces = [
        da.from_delayed(
            dask.delayed(_read_chunk)(values, start),
            (1, *values.shape[1:]),
            values.dtype,
        )
        for start in range(values.shape[0])
    ]
    return da.concatenate(pieces, axis=0)


def regions_of(chunks) -> list[tuple]:
    """The regions of a chunk grid, one per chunk."""
    edges = [np.concatenate([[0], np.cumsum(sizes)]) for sizes in chunks]
    return [
        tuple(
            slice(int(e[i]), int(e[i + 1])) for e, i in zip(edges, index, strict=True)
        )
        for index in itertools.product(*(range(len(sizes)) for sizes in chunks))
    ]


class RecordingScheduler:
    """A synchronous dask scheduler that records the graphs it runs."""

    def __init__(self):
        self.graph_sizes = []

    def __call__(self, dsk, keys, **kwargs):
        self.graph_sizes.append(len(dsk))
        return dask.local.get_sync(dsk, keys, **kwargs)


@pytest.fixture
def values():
    rng = np.random.default_rng(3)
    return rng.normal(size=(6, 2, 5, 4)).astype(np.float32)


class TestRegionReader:
    @pytest.mark.parametrize(
        "chunks, regions",
        [
            ((1, 2, 5, 4), None),
            ((2, 1, 3, 4), None),
            # regions that cut through chunks
            (
                (4, 2, 5, 4),
                [
                    (slice(0, 3), slice(0, 2), slice(0, 5), slice(0, 4)),
                    (slice(3, 6), slice(0, 2), slice(1, 5), slice(0, 2)),
                ],
            ),
        ],
    )
    def test_regions_hold_the_values(self, values, chunks, regions):
        lazy = xr.DataArray(da.from_array(values, chunks=chunks), dims=_DIMS)
        eager = xr.DataArray(values, dims=_DIMS)
        if regions is None:
            regions = regions_of(lazy.chunks)
        reader = RegionReader([lazy, None, eager])
        seen = []
        with dask.config.set(scheduler="synchronous"):
            for region, (from_lazy, nothing, from_eager) in reader.regions(regions):
                seen.append(region)
                np.testing.assert_array_equal(from_lazy, values[region])
                np.testing.assert_array_equal(from_eager, values[region])
                assert nothing is None
        assert seen == list(regions)

    @pytest.mark.parametrize("batch_bytes", [1, 3 * 2 * 5 * 4 * 4, None])
    def test_each_chunk_computed_once(self, values, batch_bytes):
        calls = []

        def record(block, block_id=None):
            calls.append(block_id)
            return block

        lazy = xr.DataArray(
            da.from_array(values, chunks=(1, 2, 5, 4)).map_blocks(
                record, dtype=values.dtype, meta=np.array((), values.dtype)
            ),
            dims=_DIMS,
        )
        # flags derived from the pixels share their tasks
        flags = np.isnan(lazy)
        reader = RegionReader([lazy, flags])
        with dask.config.set(scheduler="synchronous"):
            out = list(reader.regions(regions_of(lazy.chunks), batch_bytes))
        assert sorted(calls) == sorted(np.ndindex(6, 1, 1, 1))
        assert len(out) == 6

    def test_batches_respect_the_budget(self, values):
        lazy = xr.DataArray(da.from_array(values, chunks=(1, 2, 5, 4)), dims=_DIMS)
        scheduler = RecordingScheduler()
        region_bytes = 2 * 5 * 4 * values.itemsize
        with dask.config.set(scheduler=scheduler):
            list(
                RegionReader([lazy]).regions(regions_of(lazy.chunks), 2 * region_bytes)
            )
        assert len(scheduler.graph_sizes) == 3

    def test_a_batch_runs_only_its_own_tasks(self):
        """The cost of a batch does not grow with the number of chunks of the
        array (the writers were quadratic in the number of chunks when every
        region was computed with dask.compute on the whole graph)."""
        values = np.arange(400 * 3 * 2, dtype=np.float32).reshape(400, 1, 3, 2)
        lazy = xr.DataArray(reader_like_array(values), dims=_DIMS)
        assert len(dict(lazy.data.__dask_graph__())) > 400
        scheduler = RecordingScheduler()
        regions = regions_of(lazy.chunks)
        with dask.config.set(scheduler=scheduler):
            for region, (block,) in RegionReader([lazy]).regions(regions, 1):
                np.testing.assert_array_equal(block, values[region])
        assert len(scheduler.graph_sizes) == 400
        assert max(scheduler.graph_sizes) <= 5


def _cube(path: str, nchan: int) -> None:
    """A small single-polarization FITS cube with one NaN pixel per plane."""
    data = np.arange(nchan * 4 * 3, dtype=np.float32).reshape(nchan, 1, 4, 3)
    data[:, :, 0, 0] = np.nan
    header = fits.Header()
    for key, value in {
        "CTYPE1": "RA---SIN",
        "CRVAL1": 10.0,
        "CDELT1": -1 / 3600,
        "CRPIX1": 2.0,
        "CUNIT1": "deg",
        "CTYPE2": "DEC--SIN",
        "CRVAL2": -30.0,
        "CDELT2": 1 / 3600,
        "CRPIX2": 2.0,
        "CUNIT2": "deg",
        "CTYPE3": "STOKES",
        "CRVAL3": 1.0,
        "CDELT3": 1.0,
        "CRPIX3": 1.0,
        "CTYPE4": "FREQ",
        "CRVAL4": 1e11,
        "CDELT4": 1e6,
        "CRPIX4": 1.0,
        "CUNIT4": "Hz",
        "RESTFRQ": 1e11,
        "SPECSYS": "LSRK",
        "RADESYS": "FK5",
        "EQUINOX": 2000.0,
        "DATE-OBS": "2020-01-01T00:00:00",
        "TIMESYS": "UTC",
        "TELESCOP": "ALMA",
        "BUNIT": "Jy/beam",
    }.items():
        header[key] = value
    fits.PrimaryHDU(data=data, header=header).writeto(path)


@pytest.mark.parametrize("out_format", ["fits", "casa"])
def test_writers_run_one_small_graph_per_batch(tmp_path, monkeypatch, out_format):
    """The writers compute the regions of a reader graph (one layer per
    chunk) through the region reader: one small graph per batch."""
    if out_format == "casa":
        pytest.importorskip("casacore.tables")
    nchan = 60
    path = str(tmp_path / "cube.fits")
    _cube(path, nchan)
    xds = open_image(path, chunks={"frequency": 1})
    monkeypatch.setattr(_blocks, "BATCH_BYTES", 1)
    scheduler = RecordingScheduler()
    out = str(tmp_path / f"out.{out_format}")
    with dask.config.set(scheduler=scheduler):
        write_image(xds, out, out_format)
    # one batch per channel (and, for CASA, none for the masks, which come
    # with the pixels)
    assert len(scheduler.graph_sizes) == nchan
    assert max(scheduler.graph_sizes) < 20
    back = open_image(out)
    np.testing.assert_array_equal(back.SKY.values, xds.SKY.values)
    np.testing.assert_array_equal(back.FLAG_SKY.values, xds.FLAG_SKY.values)


@pytest.mark.parametrize("out_format", ["fits", "casa"])
def test_writers_under_a_distributed_client(tmp_path, dask_client_module, out_format):
    """With a distributed client the batches run on its workers and the
    writing process writes the results."""
    if out_format == "casa":
        pytest.importorskip("casacore.tables")
    path = str(tmp_path / "cube.fits")
    _cube(path, 12)
    xds = open_image(path, chunks={"frequency": 2})
    out = str(tmp_path / f"out.{out_format}")
    write_image(xds, out, out_format)
    assert os.path.exists(out)
    back = open_image(out)
    np.testing.assert_array_equal(back.SKY.values, xds.SKY.values)
    np.testing.assert_array_equal(back.FLAG_SKY.values, xds.FLAG_SKY.values)
