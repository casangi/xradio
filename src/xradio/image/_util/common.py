import astropy.units as u
import dask
import dask.array as da
import xarray as xr


__c = 2.99792458e+08 * u.m/u.s


def __get_xds_dim_order(has_sph:bool) -> list:
    dimorder = ['time', 'pol', 'freq']
    dir_lin = ['l', 'm'] if has_sph else ['u', 'v']
    dimorder.extend(dir_lin)
    return dimorder


def __coords_to_numpy(xds):
    for k, v in xds.coords.items():
        if dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.assign_coords({k: (v.dims, v.to_numpy())})
            xds[k].attrs = attrs
    return xds


def __dask_arrayize(xds):
    """
    If necessary, change coordinates to numpy arrays and data
    variables to dask arrays
    """
    xds = __coords_to_numpy(xds)
    for k, v in xds.data_vars.items():
        if not dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.drop_vars([k])
            # may need to add dims to this call as in numpy method analogs in this file
            xds = xds.assign({k: da.array(v)})
            xds[k].attrs = attrs
    for k, v in xds.attrs.items():
        if isinstance(v, xr.Dataset):
            xds.attrs[k] = __dask_arrayize(v)
    return xds


def __numpy_arrayize(xds):
    xds = __coords_to_numpy(xds)
    for k, v in xds.data_vars.items():
        if dask.is_dask_collection(v):
            attrs = xds[k].attrs
            xds = xds.drop_vars([k])
            xds = xds.assign({k: (v.dims, v.to_numpy())})
            xds[k].attrs = attrs
    for k, v in xds.attrs.items():
        if isinstance(v, xr.Dataset):
            xds.attrs[k] = __dask_arrayize(v)
    return xds


