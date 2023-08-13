import os
import xarray as xr
import pandas as pd

def read_processing_set(ps_name):
    items = os.listdir(ps_name)

    ps = {}
    for i in items:
        if "ddi" in i:
            ps[i] = xr.open_zarr(ps_name + "/" +i+"/MAIN")
            ps[i].attrs["antenna_xds"] = xr.open_zarr(ps_name + "/" +i+"/ANTENNA")
    return ps

