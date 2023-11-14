from ._zarr.xds_to_zarr import _write_zarr
from ._zarr.xds_from_zarr import _read_zarr
import xarray as xr


def _xds_to_zarr(xds: xr.Dataset, zarr_store: str):
    _write_zarr(xds, zarr_store)


def _xds_from_zarr(zarr_store: str, do_dask: bool):
    # if do_dask is False, all coordinate and data value arrays must
    # be numpy arrays. If True, some may (but aren't required to be)
    # dask arrays
    return _read_zarr(zarr_store, do_dask)
