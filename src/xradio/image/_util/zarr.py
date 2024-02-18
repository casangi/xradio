from ._zarr.xds_to_zarr import _write_zarr
from ._zarr.xds_from_zarr import _read_zarr
import xarray as xr


def _xds_to_zarr(xds: xr.Dataset, zarr_store: str):
    _write_zarr(xds, zarr_store)


def _xds_from_zarr(
    zarr_store: str, output: dict = {}, selection: dict = {}
) -> xr.Dataset:
    # supported key/values in output are:
    # "dv"
    #    what data variables should be returned as.
    #    "numpy": numpy arrays
    #    "dask": dask arrays
    # "coords"
    #    what coordinates should be returned as
    #    "numpy": numpy arrays
    return _read_zarr(zarr_store, output, selection)
