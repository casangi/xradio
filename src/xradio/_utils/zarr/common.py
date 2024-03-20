import copy
import xarray as xr
import zarr


def _open_dataset(store, xds_isel=None, data_variables=None, load=False):
    """

    Parameters
    ----------
    store : _type_
        _description_
    xds_isel : _type_, optional
        Example {'time':slice(0,10), 'frequency':slice(5,7)}, by default None
    data_variables : _type_, optional
        Example ['VISIBILITY','WEIGHT'], by default None
    load : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    import dask

    xds = xr.open_zarr(store)

    if xds_isel is not None:
        xds = xds.isel(xds_isel)

    if data_variables is not None:
        xds_sub = xr.Dataset()
        for dv in data_variables:
            xds_sub[dv] = xds[dv]
        xds_sub.attrs = xds.attrs
        xds = xds_sub

    if load:
        with dask.config.set(scheduler="synchronous"):
            xds = xds.load()
    return xds


# Code to depricate:
def _get_attrs(zarr_obj):
    """
    get attributes of zarr obj (groups or arrays)
    """
    return {k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")}


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
    # Used by xarray to store array labeling info in zarr meta data.
    DIMENSION_KEY = "_ARRAY_DIMENSIONS"

    # logger = _get_logger()
    zarr_group = zarr.open_group(store=zarr_name, mode="r")
    group_attrs = _get_attrs(zarr_group)

    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()
    for var_name, var in zarr_group.arrays():
        print("Hallo 3", var_name, var.shape)
        var_attrs = _get_attrs(var)

        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None)  # No slicing.

        if (var_attrs[DIMENSION_KEY][0] == var_name) and (
            len(var_attrs[DIMENSION_KEY]) == 1
        ):
            coord = var[
                slice_dict_complete[var_attrs[DIMENSION_KEY][0]]
            ]  # Dimension coordinates.
            del var_attrs["_ARRAY_DIMENSIONS"]
            xds = xds.assign_coords({var_name: coord})
            xds[var_name].attrs = var_attrs
        else:
            # Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)

            print(var_attrs[DIMENSION_KEY])

            xds[var_name] = xr.DataArray(
                var[slicing_tuple], dims=var_attrs[DIMENSION_KEY]
            )

            if "coordinates" in var_attrs:
                del var_attrs["coordinates"]
            del var_attrs["_ARRAY_DIMENSIONS"]
            xds[var_name].attrs = var_attrs

    xds.attrs = group_attrs

    return xds
