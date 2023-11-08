import xarray as xr
import zarr
import copy
import os
from ._processing_set import processing_set

DIMENSION_KEY = "_ARRAY_DIMENSIONS"  # Used by xarray to store array labeling info in zarr meta data.
# from xradio._utils._logger import _get_logger


def _get_attrs(zarr_obj):
    """
    get attributes of zarr obj (groups or arrays)
    """
    return {k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")}


def _load_ms_xds(
    ps_name, ms_xds_name, slice_dict={}, cache_dir=None, chunk_id=None, date_time=""
):
    # logger = _get_logger()
    if cache_dir:
        xds_cached_name = (
            os.path.join(cache_dir, ms_xds_name) + "_" + str(chunk_id) + "_" + date_time
        )

        # Check if already chached:
        try:
            ms_xds = _load_ms_xds_core(
                ms_xds_name=xds_cached_name, slice_dict=slice_dict
            )

            # logger.debug(ms_xds_name + ' chunk ' + str(slice_dict) + ' was found in cache: ' + xds_cached)
            found_in_cache = True
            return xds, found_in_cache
        except:
            # logger.debug(xds_cached + ' chunk ' + str(slice_dict) + ' was not found in cache or failed to load. Retrieving chunk from ' + ms_xds_name + ' .')
            ms_xds = _load_ms_xds_core(
                ms_xds_name=os.path.join(ps_name, ms_xds_name), slice_dict=slice_dict
            )
            write_ms_xds(ms_xds, xds_cached_name)

            found_in_cache = False
            return xds, found_in_cache
    else:
        found_in_cache = None
        ms_xds = _load_ms_xds_core(
            ms_xds_name=os.path.join(ps_name, ms_xds_name), slice_dict=slice_dict
        )
        return ms_xds, found_in_cache


def _write_ms_xds(ms_xds, ms_xds_name):
    ms_xds_temp = ms_xds
    xr.Dataset.to_zarr(
        ms_xds.attrs["ANTENNA"],
        os.path.join(xds_cached_name, "ANTENNA"),
        consolidated=True,
    )
    ms_xds_temp = ms_xds
    ms_xds_temp.attrs["ANTENNA"] = {}
    xr.Dataset.to_zarr(
        ms_xds_temp, os.path.join(xds_cached_name, "MAIN"), consolidated=True
    )


def _load_ms_xds_core(ms_xds_name, slice_dict):
    ms_xds = _load_no_dask_zarr(
        zarr_name=os.path.join(ms_xds_name, "MAIN"), slice_dict=slice_dict
    )
    ms_xds.attrs["antenna_xds"] = _load_no_dask_zarr(
        zarr_name=os.path.join(ms_xds_name, "ANTENNA")
    )
    return ms_xds


def _load_no_dask_zarr(zarr_name, slice_dict={}):
    """
    Alternative to xarray open_zarr where the arrays are not Dask Arrays.

    slice_dict: A dictionary of slice objects for which values to read form a dimension.
                For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                If a dim is not specified all values are retruned.
    return:
        xarray.Dataset()

    #Should go into general utils.
    """

    # logger = _get_logger()
    zarr_group = zarr.open_group(store=zarr_name, mode="r")
    group_attrs = _get_attrs(zarr_group)

    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()
    for var_name, var in zarr_group.arrays():
        var_attrs = _get_attrs(var)

        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None)  # No slicing.

        if (var_attrs[DIMENSION_KEY][0] == var_name) and (
            len(var_attrs[DIMENSION_KEY]) == 1
        ):
            coords[var_name] = var[
                slice_dict_complete[var_attrs[DIMENSION_KEY][0]]
            ]  # Dimension coordinates.
        else:
            # Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)
            xds[var_name] = xr.DataArray(
                var[slicing_tuple], dims=var_attrs[DIMENSION_KEY]
            )

    xds = xds.assign_coords(coords)

    xds.attrs = group_attrs

    return xds


def load_processing_set(ps_name, sel_parms):
    """
    sel_parms
        A dictionary where the keys are the names of the ms_xds's and the values are slice_dicts.
        slice_dicts: A dictionary where the keys are the dimension names and the values are slices.
    """
    ps = processing_set()
    for name_ms_xds, ms_xds_sel_parms in sel_parms.items():
        ps[name_ms_xds] = _load_ms_xds(ps_name, name_ms_xds, ms_xds_sel_parms)[0]
    return ps
