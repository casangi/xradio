import os

import xarray as xr

from ._processing_set import processing_set


def read_processing_set(ps_name, intents=None, fields=None):
    items = os.listdir(ps_name)
    ms_xds = xr.Dataset()
    ps = processing_set()
    for i in items:
        if "ddi" in i:
            xds = xr.open_zarr(ps_name + "/" + i + "/MAIN")

            if (intents is None) or (xds.attrs["intent"] in intents):
                if (fields is None) or (xds.attrs["field_info"]["name"] in fields):
                    ps[i] = xds
                    sub_xds = {
                        "antenna_xds": "ANTENNA",
                        "pointing_xds": "POINTING",
                    }
                    for sub_xds_key, sub_xds_name in sub_xds.items():
                        ps[i].attrs[sub_xds_key] = xr.open_zarr(
                            ps_name + "/" + i + "/" + sub_xds_name
                        )

                    optional_sub_xds = {
                        "weather_xds": "WEATHER",
                    }
                    for sub_xds_key, sub_xds_name in optional_sub_xds.items():
                        sub_xds_path = ps_name + "/" + i + "/" + sub_xds_name
                        if os.path.isdir(sub_xds_path):
                            ps[i].attrs[sub_xds_key] = xr.open_zarr(sub_xds_path)

    return ps
