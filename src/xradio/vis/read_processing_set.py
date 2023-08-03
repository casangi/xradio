import os
import xarray as xr
import pandas as pd

def read_processing_set(ps_name):
    items = os.listdir(ps_name)

    ps = {}
    for i in items:
        if 'MAIN' in i:
            ps[i[:-5]] = xr.open_zarr(ps_name+'/'+i)
            ps[i[:-5]].attrs['antenna_xds'] = xr.open_zarr(ps_name+'/'+i[:-5]+'_ANTENNA')
    return ps

