import copy
import xarray as xr
import zarr
import s3fs
import os
from botocore.exceptions import NoCredentialsError

# from xradio.vis._vis_utils._ms.msv2_to_msv4_meta import (
#     column_description_casacore_to_msv4_measure,
# )


def _get_file_system_and_items(ps_store: str):

    # default to assuming the data are accessible on local file system
    if os.path.isdir(ps_store):
        # handle a common shell convention
        if ps_store.startswith("~"):
            ps_store = os.path.expanduser(ps_store)
        items = os.listdir(ps_store)
        file_system = os

    elif ps_store.startswith("s3"):
        # only if not found locally, check if dealing with an S3 bucket URL
        # if not ps_store.endswith("/"):
        #     # just for consistency, as there is no os.path equivalent in s3fs
        #     ps_store = ps_store + "/"

        try:
            # initialize the S3 "file system", first attempting to use pre-configured credentials
            file_system = s3fs.S3FileSystem(anon=False, requester_pays=False)
            items = [
                bd.split(sep="/")[-1]
                for bd in file_system.listdir(ps_store, detail=False)
            ]

        except (NoCredentialsError, PermissionError) as e:
            # only public, read-only buckets will be accessible
            # we will want to add messaging and error handling here
            file_system = s3fs.S3FileSystem(anon=True)
            items = [
                bd.split(sep="/")[-1]
                for bd in file_system.listdir(ps_store, detail=False)
            ]
    else:
        raise FileNotFoundError(
            f"Could not find {ps_store} either locally or in the cloud."
        )

    items = [
        item for item in items if not item.startswith(".")
    ]  # Mac OS likes to place hidden files in the directory (.DStore).
    return file_system, items


def _open_dataset(
    store, file_system=os, xds_isel=None, data_variables=None, load=False
):
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

    if isinstance(file_system, s3fs.core.S3FileSystem):
        mapping = s3fs.S3Map(root=store, s3=file_system, check=False)
        xds = xr.open_zarr(store=mapping)
    else:
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
