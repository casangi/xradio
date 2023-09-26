import os
import xarray as xr
import pandas as pd
from ._processing_set import _processing_set


def read_processing_set(ps_name, intents=None, fields=None):
    items = os.listdir(ps_name)
    ms_xds = xr.Dataset()
    ps = _processing_set()
    for i in items:
        if "ddi" in i:
            xds = xr.open_zarr(ps_name + "/" + i + "/MAIN")

            if (intents is None) or (xds.attrs["intent"] in intents):
                if (fields is None) or (xds.attrs["field_info"]["name"] in fields):
                    ps[i] = xds
                    ps[i].attrs["antenna_xds"] = xr.open_zarr(
                        ps_name + "/" + i + "/ANTENNA"
                    )

    return ps
