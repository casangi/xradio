import os
from typing import Dict, Union
import dask
import xarray as xr
import s3fs


def load_processing_set(
    ps_store: str,
    sel_parms: dict = None,
    data_group_name: str = None,
    include_variables: Union[list, None] = None,
    drop_variables: Union[list, None] = None,
    load_sub_datasets: bool = True,
) -> xr.DataTree:
    """Loads a processing set into memory.

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set. For example '/users/user_1/uid___A002_Xf07bba_Xbe5c_target.lsrk.vis.zarr' for a file stored on a local file system, or 's3://viper-test-data/Antennae_North.cal.lsrk.split.vis.zarr/' for a file in AWS object storage.
    sel_parms : dict, optional
        A dictionary where the keys are the names of the ms_xdt's (measurement set xarray data trees) and the values are slice_dicts.
        slice_dicts: A dictionary where the keys are the dimension names and the values are slices.

        For example::

            {

                'ms_v4_name_1': {'frequency': slice(0, 160, None),'time':slice(0,100)},
                ...
                'ms_v4_name_n': {'frequency': slice(0, 160, None),'time':slice(0,100)},
            }

        By default None, which loads all ms_xdts.
    data_group_name : str, optional
        The name of the data group to select. By default None, which loads all data groups.
    include_variables : Union[list, None], optional
        The list of data variables to load into memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will load all data variables into memory.
    drop_variables : Union[list, None], optional
        The list of data variables to drop from memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will not drop any data variables from memory.
    load_sub_datasets : bool, optional
        If true sub-datasets (for example weather_xds, antenna_xds, pointing_xds, system_calibration_xds ...) will be loaded into memory, by default True.

    Returns
    -------
    xarray.DataTree
        In memory representation of processing set using xr.DataTree.
    """
    from xradio._utils.zarr.common import _get_file_system_and_items

    file_system, ms_store_list = _get_file_system_and_items(ps_store)

    with dask.config.set(
        scheduler="synchronous"
    ):  # serial scheduler, critical so that this can be used within delayed functions.
        ps_xdt = xr.DataTree()

        if sel_parms:
            for ms_name, ms_xds_isel in sel_parms.items():
                ms_store = os.path.join(ps_store, ms_name)

                if isinstance(file_system, s3fs.core.S3FileSystem):
                    ms_store = s3fs.S3Map(root=ps_store, s3=file_system, check=False)

                if ms_xds_isel:
                    ms_xdt = (
                        xr.open_datatree(
                            ms_store, engine="zarr", drop_variables=drop_variables
                        )
                        .isel(ms_xds_isel)
                        .xr_ms.sel(data_group_name=data_group_name)
                    )
                else:
                    ms_xdt = xr.open_datatree(
                        ms_store, engine="zarr", drop_variables=drop_variables
                    ).xr_ms.sel(data_group_name=data_group_name)

                if include_variables is not None:
                    for data_vars in ms_xdt.ds.data_vars:
                        if data_vars not in include_variables:
                            ms_xdt.ds = ms_xdt.ds.drop_vars(data_vars)

                ps_xdt[ms_name] = ms_xdt

            ps_xdt.attrs["type"] = "processing_set"
        else:
            ps_xdt = xr.open_datatree(
                ps_store, engine="zarr", drop_variables=drop_variables
            )

            if (include_variables is not None) or data_group_name:
                for ms_name, ms_xdt in ps_xdt.items():

                    ms_xdt = ms_xdt.xr_ms.sel(data_group_name=data_group_name)

                    if include_variables is not None:
                        for data_vars in ms_xdt.ds.data_vars:
                            if data_vars not in include_variables:
                                ms_xdt.ds = ms_xdt.ds.drop_vars(data_vars)
                    ps_xdt[ms_name] = ms_xdt

        if not load_sub_datasets:
            for ms_xdt in ps_xdt.children.values():
                ms_xdt_names = list(ms_xdt.keys())
                for sub_xds_name in ms_xdt_names:
                    if "xds" in sub_xds_name:
                        del ms_xdt[sub_xds_name]

        ps_xdt = ps_xdt.load()

    return ps_xdt


class ProcessingSetIterator:
    def __init__(
        self,
        sel_parms: dict,
        input_data_store: str,
        input_data: Union[Dict, xr.DataTree, None] = None,
        data_group_name: str = None,
        include_variables: Union[list, None] = None,
        drop_variables: Union[list, None] = None,
        load_sub_datasets: bool = True,
    ):
        """An iterator that will go through a processing set one MS v4 at a time.

        Parameters
        ----------
        sel_parms : dict
            A dictionary where the keys are the names of the ms_xds's and the values are slice_dicts.
            slice_dicts: A dictionary where the keys are the dimension names and the values are slices.
            For example::

                {
                    'ms_v4_name_1': {'frequency': slice(0, 160, None),'time':slice(0,100)},
                    ...
                    'ms_v4_name_n': {'frequency': slice(0, 160, None),'time':slice(0,100)},
                }
        input_data_store : str
            String of the path and name of the processing set. For example '/users/user_1/uid___A002_Xf07bba_Xbe5c_target.lsrk.vis.zarr'.
        input_data : Union[Dict, xr.DataTree, None], optional
            If the processing set is in memory already it can be supplied here. By default None which will make the iterator load data using the supplied input_data_store.
        data_group_name : str, optional
            The name of the data group to select. By default None, which loads all data groups.
        data_group_name : str, optional
            The name of the data group to select. By default None, which loads all data groups.
        include_variables : Union[list, None], optional
            The list of data variables to load into memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will load all data variables into memory.
        drop_variables : Union[list, None], optional
            The list of data variables to drop from memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will not drop any data variables from memory.
        load_sub_datasets : bool, optional
            If true sub-datasets (for example weather_xds, antenna_xds, pointing_xds, system_calibration_xds ...) will be loaded into memory, by default True.
        """

        self.input_data = input_data
        self.input_data_store = input_data_store
        self.sel_parms = sel_parms
        self.xds_name_iter = iter(sel_parms.keys())
        self.data_group_name = data_group_name
        self.include_variables = include_variables
        self.drop_variables = drop_variables
        self.load_sub_datasets = load_sub_datasets

    def __iter__(self):
        return self

    def __next__(self):
        try:
            sub_xds_name = next(self.xds_name_iter)
        except Exception as e:
            raise StopIteration

        if self.input_data is None:
            slice_description = self.sel_parms[sub_xds_name]
            ps_xdt = load_processing_set(
                ps_store=self.input_data_store,
                sel_parms={sub_xds_name: slice_description},
                data_group_name=self.data_group_name,
                include_variables=self.include_variables,
                drop_variables=self.drop_variables,
                load_sub_datasets=self.load_sub_datasets,
            )
            sub_xdt = ps_xdt.get(0)
        else:
            sub_xdt = self.input_data[sub_xds_name]  # In memory

        return sub_xdt
