import copy
import xarray as xr
import zarr
import s3fs
import os
from botocore.exceptions import NoCredentialsError
from xradio.vis._vis_utils._ms.msv2_to_msv4_meta import (
    column_description_casacore_to_msv4_measure,
)


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
        raise (
            FileNotFoundError,
            f"Could not find {ps_store} either locally or in the cloud.",
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


def convert_generic_xds_to_xradio_schema(
    generic_xds: xr.Dataset,
    msv4_xds: xr.Dataset,
    to_new_data_variables: dict,
    to_new_coords: dict,
) -> xr.Dataset:
    """Converts a generic xarray Dataset to the xradio schema.

    This function takes a generic xarray Dataset and converts it to an xradio schema
    represented by the msv4_xds Dataset. It performs the conversion based on the provided
    mappings in the to_new_data_variables and to_new_coords dictionaries.

    Parameters
    ----------
    generic_xds : xr.Dataset
        The generic xarray Dataset to be converted.
    msv4_xds : xr.Dataset
        The xradio schema represented by the msv4_xds Dataset.
    to_new_data_variables : dict
        A dictionary mapping the data variables/coordinates in the generic_xds Dataset to the new data variables
        in the msv4_xds Dataset. The keys are the old data variables/coordinates and the values are a list of the new name and a list of the new dimension names.
    to_new_coords : dict
        A dictionary mapping  data variables/coordinates in the generic_xds Dataset to the new coordinates
        in the msv4_xds Dataset. The keys are the old data variables/coordinates and the values are a list of the new name and a list of the new dimension names.

    Returns
    -------
    xr.Dataset
        The converted xradio schema represented by the msv4_xds Dataset.

    Notes
    -----
    Example to_new_data_variables:
    to_new_data_variables = {
        "POSITION": ["ANTENNA_POSITION",["name", "cartesian_pos_label"]],
        "OFFSET": ["ANTENNA_FEED_OFFSET",["name", "cartesian_pos_label"]],
        "DISH_DIAMETER": ["ANTENNA_DISH_DIAMETER",["name"]],
    }

    Example to_new_coords:
    to_new_coords = {
        "NAME": ["name",["name"]],
        "STATION": ["station",["name"]],
        "MOUNT": ["mount",["name"]],
        "PHASED_ARRAY_ID": ["phased_array_id",["name"]],
        "antenna_id": ["antenna_id",["name"]],
    }
    """

    column_description = generic_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    coords = {}

    name_keys = list(generic_xds.data_vars.keys()) + list(generic_xds.coords.keys())

    for key in name_keys:

        if key in column_description:
            msv4_measure = column_description_casacore_to_msv4_measure(
                column_description[key]
            )
        else:
            msv4_measure = None

        if key in to_new_data_variables:
            new_dv = to_new_data_variables[key]
            msv4_xds[new_dv[0]] = xr.DataArray(generic_xds[key].data, dims=new_dv[1])

            if msv4_measure:
                msv4_xds[new_dv[0]].attrs.update(msv4_measure)

        if key in to_new_coords:
            new_coord = to_new_coords[key]
            coords[new_coord[0]] = (
                new_coord[1],
                generic_xds[key].data,
            )
    msv4_xds = msv4_xds.assign_coords(coords)
    return msv4_xds
