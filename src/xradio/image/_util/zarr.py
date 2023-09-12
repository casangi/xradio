from ._zarr.xds_to_zarr import (__write_zarr)
from ._zarr.xds_from_zarr import (__read_zarr)
import xarray as xr

def __xds_to_zarr(xds: xr.Dataset, zarr_store:str):
    __write_zarr(xds, zarr_store)


def __xds_from_zarr(zarr_store: str):
    return __read_zarr(zarr_store)

