import os
from xradio.measurement_set import ProcessingSet
from typing import Dict, Union


def load_processing_set(
    ps_store: str,
    sel_parms: dict,
    data_variables: Union[list, None] = None,
    load_sub_datasets: bool = True,
) -> ProcessingSet:
    """Loads a processing set into memory.

    Parameters
    ----------
    ps_store : str
        String of the path and name of the processing set. For example '/users/user_1/uid___A002_Xf07bba_Xbe5c_target.lsrk.vis.zarr' for a file stored on a local file system, or 's3://viper-test-data/Antennae_North.cal.lsrk.split.vis.zarr/' for a file in AWS object storage.
    sel_parms : dict
        A dictionary where the keys are the names of the ms_xds's and the values are slice_dicts.
        slice_dicts: A dictionary where the keys are the dimension names and the values are slices.
        For example::

            {
                'ms_v4_name_1': {'frequency': slice(0, 160, None),'time':slice(0,100)},
                ...
                'ms_v4_name_n': {'frequency': slice(0, 160, None),'time':slice(0,100)},
            }

    data_variables : Union[list, None], optional
        The list of data variables to load into memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will load all data variables into memory.
    load_sub_datasets : bool, optional
        If true sub-datasets (for example weather_xds, antenna_xds, pointing_xds, system_calibration_xds ...) will be loaded into memory, by default True.

    Returns
    -------
    ProcessingSet
        In memory representation of processing set (data is represented by Dask.arrays).
    """
    from xradio._utils.zarr.common import _open_dataset, _get_file_system_and_items

    file_system, ms_store_list = _get_file_system_and_items(ps_store)

    ps = ProcessingSet()
    for ms_name, ms_xds_isel in sel_parms.items():
        ms_store = os.path.join(ps_store, ms_name)
        correlated_store = os.path.join(ms_store, "correlated_xds")

        xds = _open_dataset(
            correlated_store,
            file_system,
            ms_xds_isel,
            data_variables,
            load=True,
        )
        data_groups = xds.attrs["data_groups"]

        if load_sub_datasets:
            from xradio.measurement_set.open_processing_set import _open_sub_xds

            sub_xds_dict, field_and_source_xds_dict = _open_sub_xds(
                ms_store, file_system=file_system, load=True, data_groups=data_groups
            )

            xds.attrs = {
                **xds.attrs,
                **sub_xds_dict,
            }
            for data_group_name, data_group_vals in data_groups.items():

                xds[data_group_vals["correlated_data"]].attrs[
                    "field_and_source_xds"
                ] = field_and_source_xds_dict[data_group_name]

        ps[ms_name] = xds

    return ps


class ProcessingSetIterator:
    def __init__(
        self,
        sel_parms: dict,
        input_data_store: str,
        input_data: Union[Dict, ProcessingSet, None] = None,
        data_variables: list = None,
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
        input_data : Union[Dict, processing_set, None], optional
            If the processing set is in memory already it can be supplied here. By default None which will make the iterator load data using the supplied input_data_store.
        data_variables : list, optional
            The list of data variables to load into memory for example ['VISIBILITY', 'WEIGHT, 'FLAGS']. By default None which will load all data variables into memory.
        load_sub_datasets : bool, optional
            If true sub-datasets (for example weather_xds, antenna_xds, pointing_xds, system_calibration_xds ...) will be loaded into memory, by default True.
        """

        self.input_data = input_data
        self.input_data_store = input_data_store
        self.sel_parms = sel_parms
        self.xds_name_iter = iter(sel_parms.keys())
        self.data_variables = data_variables
        self.load_sub_datasets = load_sub_datasets

    def __iter__(self):
        return self

    def __next__(self):
        try:
            xds_name = next(self.xds_name_iter)
        except Exception as e:
            raise StopIteration

        if self.input_data is None:
            slice_description = self.sel_parms[xds_name]
            ps = load_processing_set(
                ps_store=self.input_data_store,
                sel_parms={xds_name: slice_description},
                data_variables=self.data_variables,
                load_sub_datasets=self.load_sub_datasets,
            )
            xds = ps.get(0)
        else:
            xds = self.input_data[xds_name]  # In memory

        return xds
