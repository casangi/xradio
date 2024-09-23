import pandas as pd
from xradio._utils.list_and_array import to_list
import xarray as xr
import numbers
import os

cor_attrs_xds_names = {
    "ANTENNA": "antenna_xds",
    "POINTING": "pointing_xds",
    "SYSCAL": "system_calibration_xds",
    "GAIN_CURVE": "gain_curve_xds",
    "PHASE_CAL": "phase_calibration_xds",
    "WEATHER": "weather_xds",
}

class CorrelatedXds(xr.Dataset):
    __slots__ = ()
    
    def __init__(self, xds):
        super().__init__(xds.data_vars, xds.coords, xds.attrs)

    def to_store(self, store):        
        copy_cor_xds = self.copy() #No deep copy
        
        #Remove field_and_source_xds from all correlated_data (VISIBILITY/SPECTRUM) data variables
        #and save them as separate zarr files.
        for data_group_name, data_group in self.attrs["data_groups"].items():
            del copy_cor_xds[data_group["correlated_data"]].attrs["field_and_source_xds"]
            xr.Dataset.to_zarr(self[data_group["correlated_data"]].attrs["field_and_source_xds"], os.path.join(store, "field_and_source_xds_" + data_group_name))
   
        #Remove xds attributes from copy_cor_xds and save xds attributes as separate zarr files.        
        for attrs_name in self.attrs:
            if "xds" in attrs_name:
                del copy_cor_xds.attrs[attrs_name]
                xr.Dataset.to_zarr(self.attrs[attrs_name],os.path.join(store, attrs_name))

        #Save copy_cor_xds as zarr file.
        xr.Dataset.to_zarr(copy_cor_xds,os.path.join(store, "main_xds"))
