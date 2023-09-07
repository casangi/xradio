from ._zarr.zarr_from_xds import (__write_zarr)
import xarray as xr

def __xds_to_zarr(xds: xr.Dataset, imagename: str):
    __write_zarr(xds, imagename)

