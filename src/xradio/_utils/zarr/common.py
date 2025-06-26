import xarray as xr
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

        except (NoCredentialsError, PermissionError):
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
