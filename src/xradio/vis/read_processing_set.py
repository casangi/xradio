import os
import xarray as xr
from ._processing_set import processing_set
import graphviper.utils.logger as logger
from xradio._utils.zarr.common import _open_dataset, _get_ms_stores_and_file_system
import s3fs


def read_processing_set(
    ps_store: str,
    obs_modes: list = None,
) -> processing_set:
    """Creates a lazy representation of a Processing Set (only meta-data is loaded into memory).

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set. For example '/users/user_1/uid___A002_Xf07bba_Xbe5c_target.lsrk.vis.zarr'.
    obs_modes : list, optional
        A list of obs_mode to be read for example ['OBSERVE_TARGET#ON_SOURCE']. The obs_mode in a processing set can be seem by calling processing_set.summary().
        By default None, which will read all obs_mode.

    Returns
    -------
    processing_set
        Lazy representation of processing set (data is represented by Dask.arrays).
    """
    file_system, ms_store_list = _get_ms_stores_and_file_system(ps_store)

    ps = processing_set()
    data_group = "base"
    for ms_name in ms_store_list:
        # try:
        ms_store = os.path.join(ps_store, ms_name)
        ms_main_store = os.path.join(ms_store, "MAIN")

        xds = _open_dataset(ms_main_store, file_system)
        data_groups = xds.attrs["data_groups"]

        if (obs_modes is None) or (
            xds.attrs["partition_info"]["obs_mode"] in obs_modes
        ):
            sub_xds_dict, field_and_source_xds_dict = _read_sub_xds(
                ms_store, file_system=file_system, data_groups=data_groups
            )

            xds.attrs = {
                **xds.attrs,
                **sub_xds_dict,
            }

            for data_group_name, data_group_vals in data_groups.items():
                if "visibility" in data_group_vals:
                    xds[data_group_vals["visibility"]].attrs["field_and_source_xds"] = (
                        field_and_source_xds_dict[data_group_name]
                    )
                elif "spectrum" in data_group_vals:
                    xds[data_group_vals["spectrum"]].attrs["field_and_source_xds"] = (
                        field_and_source_xds_dict[data_group_name]
                    )

            ps[ms_name] = xds
        # except Exception as e:
        #     logger.warning(f"Could not read {ms_name} due to {e}")
        #     continue

    return ps


def _read_sub_xds(ms_store, file_system, data_groups, load=False):
    sub_xds_dict = {}
    field_and_source_xds_dict = {}

    xds_names = {
        "ANTENNA": "antenna_xds",
        "WEATHER": "weather_xds",
        "POINTING": "pointing_xds",
    }

    if isinstance(file_system, s3fs.core.S3FileSystem):
        file_names = [
            bd.split(sep="/")[-1] for bd in file_system.listdir(ms_store, detail=False)
        ]
    else:
        file_names = file_system.listdir(ms_store)
    file_names = [item for item in file_names if not item.startswith(".")]

    file_names.remove("MAIN")

    field_dict = {"FIELD_AND_SOURCE_" + key.upper(): key for key in data_groups.keys()}

    # field_and_source_xds_name_start = "FIELD"
    for n in file_names:
        xds = _open_dataset(
            os.path.join(ms_store, n), load=load, file_system=file_system
        )
        if n in field_dict.keys():
            field_and_source_xds_dict[field_dict[n]] = xds
        else:
            sub_xds_dict[xds_names[n]] = xds

    return sub_xds_dict, field_and_source_xds_dict


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
