import os
import xarray as xr
import pandas as pd

def read_processing_set(ps_name):
    items = os.listdir(ps_name)

    ps = {}
    for i in items:
        if 'ANTENNA' not in i:
            ps[i] = xr.open_zarr(ps_name+'/'+i)
            ps[i].attrs['antenna_df'] = pd.read_parquet(ps_name+'/'+i+'_ANTENNA.pq')
    return ps

