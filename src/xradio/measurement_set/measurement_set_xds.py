import pandas as pd
from xradio._utils.list_and_array import to_list
import xarray as xr
import numbers
import os
from collections.abc import Mapping, Iterable
from typing import Any, Union


class MeasurementSetXds(xr.Dataset):
    __slots__ = ()

    def __init__(self, xds):
        super().__init__(xds.data_vars, xds.coords, xds.attrs)

    def to_store(self, store, **kwargs):
        """
        Write the MeasurementSetXds to a Zarr store.
        Does not write to cloud storage yet.

        Args:
            store (str): The path to the Zarr store.
            **kwargs: Additional keyword arguments to be passed to `xarray.Dataset.to_zarr`. See https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html for more information.

        Returns:
            None
        """

        copy_cor_xds = self.copy()  # No deep copy

        # Remove field_and_source_xds from all correlated_data (VISIBILITY/SPECTRUM) data variables
        # and save them as separate zarr files.
        for data_group_name, data_group in self.attrs["data_groups"].items():
            del copy_cor_xds[data_group["correlated_data"]].attrs[
                "field_and_source_xds"
            ]

            # print("data_group_name", data_group_name)
            xr.Dataset.to_zarr(
                self[data_group["correlated_data"]].attrs["field_and_source_xds"],
                os.path.join(store, "field_and_source_xds_" + data_group_name),
                **kwargs,
            )

        # Remove xds attributes from copy_cor_xds and save xds attributes as separate zarr files.
        for attrs_name in self.attrs:
            if "xds" in attrs_name:
                del copy_cor_xds.attrs[attrs_name]
                xr.Dataset.to_zarr(
                    self.attrs[attrs_name], os.path.join(store, attrs_name), **kwargs
                )

        # Save copy_cor_xds as zarr file.
        xr.Dataset.to_zarr(
            copy_cor_xds, os.path.join(store, "correlated_xds"), **kwargs
        )

    def sel(
        self,
        indexers: Union[Mapping[Any, Any], None] = None,
        method: Union[str, None] = None,
        tolerance: Union[int, float, Iterable[Union[int, float]], None] = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ):
        """
        Select data along dimension(s) by label. Overrides `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ so that a data group can be selected by name by using the `data_group_name` parameter.
        For more information on data groups see `Data Groups <https://xradio.readthedocs.io/en/latest/measurement_set_overview.html#Data-Groups>`__ section. See `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ for parameter descriptions.

        Returns:
            MeasurementSetXds

        Examples
        --------
        >>> # Select data group 'corrected' and polarization 'XX'.
        >>> selected_ms_xds = ms_xds.sel(data_group_name='corrected', polarization='XX')

        >>> # Select data group 'corrected' and polarization 'XX' using a dict.
        >>> selected_ms_xds = ms_xds.sel({'data_group_name':'corrected', 'polarization':'XX')
        """

        if "data_group_name" in indexers_kwargs:
            data_group_name = indexers_kwargs["data_group_name"]
            del indexers_kwargs["data_group_name"]
        elif (indexers is not None) and ("data_group_name" in indexers):
            data_group_name = indexers["data_group_name"]
            del indexers["data_group_name"]
        else:
            data_group_name = None

        if data_group_name is not None:
            sel_data_group_set = set(
                self.attrs["data_groups"][data_group_name].values()
            )

            data_variables_to_drop = []
            for dg in self.attrs["data_groups"].values():
                temp_set = set(dg.values()) - sel_data_group_set
                data_variables_to_drop.extend(list(temp_set))

            data_variables_to_drop = list(set(data_variables_to_drop))

            sel_ms_xds = MeasurementSetXds(
                super()
                .sel(indexers, method, tolerance, drop, **indexers_kwargs)
                .drop_vars(data_variables_to_drop)
            )

            sel_ms_xds.attrs["data_groups"] = {
                data_group_name: self.attrs["data_groups"][data_group_name]
            }

            return sel_ms_xds
        else:
            return MeasurementSetXds(
                super().sel(indexers, method, tolerance, drop, **indexers_kwargs)
            )
