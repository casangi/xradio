import os
import xarray as xr
from ._processing_set import processing_set
import graphviper.utils.logger as logger
from xradio._utils.zarr.common import _open_dataset
import s3fs
from botocore.exceptions import NoCredentialsError


def read_processing_set(
    ps_store: str, intents: list = None, 
) -> processing_set:
    """Creates a lazy representation of a Processing Set (only meta-data is loaded into memory).

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set. For example '/users/user_1/uid___A002_Xf07bba_Xbe5c_target.lsrk.vis.zarr'.
    intents : list, optional
        A list of the intents to be read for example ['OBSERVE_TARGET#ON_SOURCE']. The intents in a processing set can be seem by calling processing_set.summary().
        By default None, which will read all intents.

    Returns
    -------
    processing_set
        Lazy representation of processing set (data is represented by Dask.arrays).
    """
    s3 = None
    ps_store_is_s3dir = None

    if os.path.isdir(ps_store):
        # default to assuming the data are accessible on local file system
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

            items = [bd.split(sep="/")[-1] for bd in file_system.listdir(ps_store, detail=False)]

        except (NoCredentialsError, PermissionError) as e:
            # only public, read-only buckets will be accessible
            # we will want to add messaging and error handling here
            file_system = s3fs.S3FileSystem(anon=True)

            items = [bd.split(sep="/")[-1] for bd in file_system.listdir(ps_store, detail=False)]

    else:
        raise (
            FileNotFoundError,
            f"Could not find {ps_store} either locally or in the cloud.",
        )

    ps = processing_set()
    data_group = "base"
    for ms_dir_name in items:
        #try:              
        store_path = os.path.join(ps_store, ms_dir_name)
        store_path_main = os.path.join(store_path, "MAIN")

        xds = _open_dataset(store_path_main, file_system)
        data_groups = xds.attrs["data_groups"]

        if (intents is None) or (xds.attrs["intent"] in intents):
            sub_xds_dict, field_and_source_xds_dict = _read_sub_xds(store_path, file_system=file_system, data_groups=data_groups)
            
            xds.attrs = {
                    **xds.attrs,
                    **sub_xds_dict,
                }
            
            for data_group_name, data_group_vals in data_groups.items():
                xds[data_group_vals['visibility']].attrs['field_and_source_xds'] = field_and_source_xds_dict[data_group_name]
            
            ps[ms_dir_name] = xds
        # except Exception as e:  
        #     logger.warning(f"Could not read {ms_dir_name} due to {e}")  
        #     continue
                

    return ps


def _read_sub_xds(ms_store, file_system, data_groups, load=False):
    sub_xds_dict = {}
    field_and_source_xds_dict = {}
    
    xds_names = {
        "ANTENNA":"antenna_xds",
        "WEATHER":"weather_xds",
        "POINTING":"pointing_xds",
    }
    
    if  isinstance(file_system, s3fs.core.S3FileSystem):
        file_names = [bd.split(sep="/")[-1] for bd in file_system.listdir(ms_store, detail=False)]
    else:
        file_names = file_system.listdir(ms_store)
        
    file_names.remove("MAIN")
        
    field_dict = {"FIELD_AND_SOURCE_" + key.upper(): key for key in data_groups.keys()}
        
    # field_and_source_xds_name_start = "FIELD"
    for n in file_names:
        xds = _open_dataset(os.path.join(ms_store, n), load=load, file_system=file_system)
        if n in field_dict.keys():
            field_and_source_xds_dict[field_dict[n]] = xds
        else:
            sub_xds_dict[xds_names[n]] = xds
            
    
    return sub_xds_dict, field_and_source_xds_dict
        
    
    
    

    # sub_xds = {
    #     "antenna_xds": "ANTENNA",
    # }
    # for sub_xds_key, sub_xds_name in sub_xds.items():
    #     if "s3" in kwargs.keys():
    #         joined_store = ms_store + "/" + sub_xds_name
    #         sub_xds_dict[sub_xds_key] = _open_dataset(
    #             joined_store, load=load, s3=kwargs["s3"]
    #         )
    #     else:
    #         sub_xds_dict[sub_xds_key] = _open_dataset(
    #             os.path.join(ms_store, sub_xds_name), load=load
    #         )

    # optional_sub_xds = {
    #     "weather_xds": "WEATHER",
    #     "pointing_xds": "POINTING",
    # }
    # for sub_xds_key, sub_xds_name in optional_sub_xds.items():
    #     sub_xds_path = os.path.join(ms_store, sub_xds_name)
    #     if os.path.isdir(sub_xds_path):
    #         sub_xds_dict[sub_xds_key] = _open_dataset(sub_xds_path, load=load)
    #     elif "s3" in kwargs.keys():
    #         joined_store = ms_store + "/" + sub_xds_name
    #         if kwargs["s3"].isdir(joined_store):
    #             sub_xds_dict[sub_xds_key] = _open_dataset(
    #                 joined_store, load=load, s3=kwargs["s3"]
    #             )
    
    # if  "s3" in kwargs.keys():
    #     files = os.listdir(os.path.join(ms_store))
    # else:
    # field_and_source_xds_name_start = "FIELD"
 
    # for f in files:
    #     if "FIELD" in 
    
    

    return sub_xds_dict


def _get_data_name(xds, data_group):
    if "visibility" in xds.attrs["data_groups"][data_group]:
        data_name = xds.attrs["data_groups"][data_group]["visibility"]
    elif "spectrum" in xds.attrs["data_groups"][data_group]:
        data_name = xds.attrs["data_groups"][data_group]["spectrum"]
    else:
        error_message = (
            "No Visibility or Spectrum data variable found in data_group "
            + data_group
            + "."
        )
        logger.exception(error_message)
        raise ValueError(error_message)
    return data_name
