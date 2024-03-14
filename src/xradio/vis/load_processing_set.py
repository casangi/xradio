import xarray as xr
import zarr
import copy
import os
from ._processing_set import processing_set


def load_processing_set(
    ps_name, sel_parms, data_variables=None, load_sub_datasets=True
):
    """
    sel_parms
        A dictionary where the keys are the names of the ms_xds's and the values are slice_dicts.
        slice_dicts: A dictionary where the keys are the dimension names and the values are slices.
    """
    from xradio._utils.zarr.common import _open_dataset

    ps = processing_set()
    for ms_dir_name, ms_xds_isel in sel_parms.items():
        xds = _open_dataset(
            os.path.join(ps_name, ms_dir_name, "MAIN"),
            ms_xds_isel,
            data_variables,
            load=True,
        )

        if load_sub_datasets:
            from xradio.vis.read_processing_set import _read_sub_xds

            xds.attrs = {
                **xds.attrs,
                **_read_sub_xds(os.path.join(ps_name, ms_dir_name), load=True),
            }

        ps[ms_dir_name] = xds
    return ps


class processing_set_iterator:

    def __init__(self, data_selection, input_data_store, input_data=None):
        self.input_data = input_data
        self.input_data_store = input_data_store
        self.data_selection = data_selection
        self.xds_name_iter = iter(data_selection.keys())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            xds_name = next(self.xds_name_iter)
        except Exception as e:
            raise StopIteration

        if self.input_data is None:
            slice_description = self.data_selection[xds_name]
            ps = load_processing_set(
                ps_name=self.input_data_store,
                sel_parms={xds_name: slice_description},
            )
            xds = ps.get(0)
        else:
            xds = self.input_data[xds_name]  # In memory

        return xds


# def _load_ms_xds(
#     ps_name, ms_xds_name, slice_dict={}, cache_dir=None, chunk_id=None, date_time=""
# ):
#     # logger = _get_logger()
#     if cache_dir:
#         xds_cached_name = (
#             os.path.join(cache_dir, ms_xds_name) + "_" + str(chunk_id) + "_" + date_time
#         )

#         # Check if already chached:
#         try:
#             ms_xds = _load_ms_xds_core(
#                 ms_xds_name=xds_cached_name, slice_dict=slice_dict
#             )

#             # logger.debug(ms_xds_name + ' chunk ' + str(slice_dict) + ' was found in cache: ' + xds_cached)
#             found_in_cache = True
#             return xds, found_in_cache
#         except:
#             # logger.debug(xds_cached + ' chunk ' + str(slice_dict) + ' was not found in cache or failed to load. Retrieving chunk from ' + ms_xds_name + ' .')
#             ms_xds = _load_ms_xds_core(
#                 ms_xds_name=os.path.join(ps_name, ms_xds_name), slice_dict=slice_dict
#             )
#             write_ms_xds(ms_xds, xds_cached_name)

#             found_in_cache = False
#             return xds, found_in_cache
#     else:
#         found_in_cache = None
#         ms_xds = _load_ms_xds_core(
#             ms_xds_name=os.path.join(ps_name, ms_xds_name), slice_dict=slice_dict
#         )
#         return ms_xds, found_in_cache


# def _write_ms_xds(ms_xds, ms_xds_name):
#     ms_xds_temp = ms_xds
#     xr.Dataset.to_zarr(
#         ms_xds.attrs["ANTENNA"],
#         os.path.join(xds_cached_name, "ANTENNA"),
#         consolidated=True,
#     )
#     ms_xds_temp = ms_xds
#     ms_xds_temp.attrs["ANTENNA"] = {}
#     xr.Dataset.to_zarr(
#         ms_xds_temp, os.path.join(xds_cached_name, "MAIN"), consolidated=True
#     )


# def _load_ms_xds_core(ms_xds_name, slice_dict):
#     ms_xds = _load_no_dask_zarr(
#         zarr_name=os.path.join(ms_xds_name, "MAIN"), slice_dict=slice_dict
#     )
#     ms_xds.attrs["antenna_xds"] = _load_no_dask_zarr(
#         zarr_name=os.path.join(ms_xds_name, "ANTENNA")
#     )
#     sub_xds = {
#         "antenna_xds": "ANTENNA",
#     }
#     for sub_xds_key, sub_xds_name in sub_xds.items():
#         ms_xds.attrs[sub_xds_key] = _load_no_dask_zarr(
#             zarr_name=os.path.join(ms_xds_name, sub_xds_name)
#         )
#     optional_sub_xds = {
#         "weather_xds": "WEATHER",
#         "pointing_xds": "POINTING",
#     }
#     for sub_xds_key, sub_xds_name in sub_xds.items():
#         sub_xds_path = os.path.join(ms_xds_name, sub_xds_name)
#         if os.path.isdir(sub_xds_path):
#             ms_xds.attrs[sub_xds_key] = _load_no_dask_zarr(
#                 zarr_name=os.path.join(ms_xds_name, sub_xds_name)
#             )

#     return ms_xds
