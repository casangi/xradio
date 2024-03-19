import os
import xarray as xr
from ._processing_set import processing_set
import graphviper.utils.logger as logger
from xradio._utils.zarr.common import _open_dataset


def read_processing_set(ps_name, intents=None, data_group="base", fields=None):
    items = os.listdir(ps_name)
    ms_xds = xr.Dataset()
    ps = processing_set()
    for ms_dir_name in items:
        if "ddi" in ms_dir_name:
            xds = _open_dataset(os.path.join(ps_name, ms_dir_name, "MAIN"))
            if (intents is None) or (xds.attrs["intent"] in intents):
                data_name = _get_data_name(xds, data_group)

                if (fields is None) or (
                    xds[data_name].attrs["field_info"]["name"] in fields
                ):
                    xds.attrs = {
                        **xds.attrs,
                        **_read_sub_xds(os.path.join(ps_name, ms_dir_name)),
                    }
                    ps[ms_dir_name] = xds
    return ps


def _read_sub_xds(ms_store, load=False):
    sub_xds_dict = {}

    sub_xds = {
        "antenna_xds": "ANTENNA",
    }
    for sub_xds_key, sub_xds_name in sub_xds.items():
        sub_xds_dict[sub_xds_key] = _open_dataset(
            os.path.join(ms_store, sub_xds_name), load=load
        )

    optional_sub_xds = {
        "weather_xds": "WEATHER",
        "pointing_xds": "POINTING",
    }
    for sub_xds_key, sub_xds_name in optional_sub_xds.items():
        sub_xds_path = os.path.join(ms_store, sub_xds_name)
        if os.path.isdir(sub_xds_path):
            sub_xds_dict[sub_xds_key] = _open_dataset(sub_xds_path, load=load)

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
