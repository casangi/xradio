import pandas as pd
from xradio._utils.list_and_array import to_list
import xarray as xr
import numpy as np
import numbers
import os
from collections.abc import Mapping, Iterable
from typing import Any, Union

MS_DATASET_TYPES = {"visibility", "spectrum", "radiometer"}


class InvalidAccessorLocation(ValueError):
    """
    Raised by MeasurementSetXdt accessor functions called on a wrong DataTree node (not MSv4).
    """

    pass


@xr.register_datatree_accessor("xr_ms")
class MeasurementSetXdt:
    """Accessor to the Measurement Set DataTree node. Provides MSv4 specific functionality
    such as:

        - get_partition_info(): produce an info dict with a general MSv4 description including
          intents, SPW name, field and source names, etc.
        - get_field_and_source_xds() to retrieve the field_and_source_xds for a given data
          group.
        - sel(): select data by dimension labels, for example by data group and polaritzation

    """

    _xdt: xr.DataTree

    def __init__(self, datatree: xr.DataTree):
        """
        Initialize the MeasurementSetXdt instance.

        Parameters
        ----------
        datatree: xarray.DataTree
            The MSv4 DataTree node to construct a MeasurementSetXdt accessor.
        """

        self._xdt = datatree
        self.meta = {"summary": {}}

    def sel(
        self,
        indexers: Union[Mapping[Any, Any], None] = None,
        method: Union[str, None] = None,
        tolerance: Union[int, float, Iterable[Union[int, float]], None] = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xr.DataTree:
        """
        Select data along dimension(s) by label. Alternative to `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ so that a data group can be selected by name by using the `data_group_name` parameter.
        For more information on data groups see `Data Groups <https://xradio.readthedocs.io/en/latest/measurement_set_overview.html#Data-Groups>`__ section. See `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ for parameter descriptions.

        Returns
        -------
        xarray.DataTree
            xarray DataTree with MeasurementSetXdt accessors

        Examples
        --------
        >>> # Select data group 'corrected' and polarization 'XX'.
        >>> selected_ms_xdt = ms_xdt.xr_ms.sel(data_group_name='corrected', polarization='XX')

        >>> # Select data group 'corrected' and polarization 'XX' using a dict.
        >>> selected_ms_xdt = ms_xdt.xr_ms.sel({'data_group_name':'corrected', 'polarization':'XX')
        """

        if self._xdt.attrs.get("type") not in MS_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xdt.path} is not a MSv4 node.")

        assert self._xdt.attrs["type"] in [
            "visibility",
            "spectrum",
            "radiometer",
        ], "The type of the xdt must be 'visibility', 'spectrum' or 'radiometer'."

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
                self._xdt.attrs["data_groups"][data_group_name].values()
            )

            data_variables_to_drop = []
            for dg in self._xdt.attrs["data_groups"].values():
                temp_set = set(dg.values()) - sel_data_group_set
                data_variables_to_drop.extend(list(temp_set))

            data_variables_to_drop = list(set(data_variables_to_drop))

            sel_ms_xdt = self._xdt

            sel_corr_xds = self._xdt.ds.sel(
                indexers, method, tolerance, drop, **indexers_kwargs
            ).drop_vars(data_variables_to_drop)

            sel_ms_xdt.ds = sel_corr_xds

            sel_ms_xdt.attrs["data_groups"] = {
                data_group_name: self._xdt.attrs["data_groups"][data_group_name]
            }

            return sel_ms_xdt
        else:
            return self._xdt.sel(indexers, method, tolerance, drop, **indexers_kwargs)

    def get_field_and_source_xds(self, data_group_name: str = None) -> xr.Dataset:
        """Get the field_and_source_xds associated with data group `data_group_name`.

        Parameters
        ----------
        data_group_name : str, optional
            The data group to process. Default is "base" or if not found to first data group.

        Returns
        -------
        xarray.Dataset
            field_and_source_xds associated with the data group.
        """
        if self._xdt.attrs.get("type") not in MS_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xdt.path} is not a MSv4 node.")

        if data_group_name is None:
            if "base" in self._xdt.attrs["data_groups"].keys():
                data_group_name = "base"
            else:
                data_group_name = list(self._xdt.attrs["data_groups"].keys())[0]

        return self._xdt[f"field_and_source_xds_{data_group_name}"].ds

    def get_partition_info(self, data_group_name: str = None) -> dict:
        """
        Generate a partition info dict for an MSv4, with general MSv4 description including
        information such as field and source names, SPW name, scan name, the intents string,
        etc.

        The information is gathered from various coordinates, secondary datasets, and info
        dicts of the MSv4. For example, the SPW name comes from the attributes of the
        frequency coordinate, whereas field and source related information such as field and
        source names come from the field_and_source_xds (base) dataset of the MSv4.

        Parameters
        ----------
        data_group_name : str, optional
            The data group to process. Default is "base" or if not found to first data group.

        Returns
        -------
        dict
            Partition info dict for the MSv4
        """
        if self._xdt.attrs.get("type") not in MS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a MSv4 node (type {self._xdt.attrs.get('type')}."
            )

        if data_group_name is None:
            if "base" in self._xdt.attrs["data_groups"].keys():
                data_group_name = "base"
            else:
                data_group_name = list(self._xdt.attrs["data_groups"].keys())[0]

        field_and_source_xds = self._xdt.xr_ms.get_field_and_source_xds(data_group_name)

        if "line_name" in field_and_source_xds.coords:
            line_name = to_list(
                np.unique(np.ravel(field_and_source_xds.line_name.values))
            )
        else:
            line_name = []

        partition_info = {
            "spectral_window_name": self._xdt.frequency.attrs["spectral_window_name"],
            "field_name": to_list(np.unique(field_and_source_xds.field_name.values)),
            "polarization_setup": to_list(self._xdt.polarization.values),
            "scan_name": to_list(np.unique(self._xdt.scan_name.values)),
            "source_name": to_list(np.unique(field_and_source_xds.source_name.values)),
            "intents": self._xdt.observation_info["intents"],
            "line_name": line_name,
            "data_group_name": data_group_name,
        }

        return partition_info
