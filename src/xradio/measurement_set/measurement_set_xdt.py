from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from xradio._utils.list_and_array import to_python_type
from xradio._utils.xarray_helpers import (
    create_new_data_group,
    get_data_group_name,
    register_uncached_accessor,
    remove_variables_from_data_groups,
    replace_data_groups,
)

MS_DATASET_TYPES = {"visibility", "spectrum", "radiometer"}


class InvalidAccessorLocation(ValueError):
    """
    Raised by MeasurementSetXdt accessor functions called on a wrong DataTree node (not MSv4).
    """

    pass


class MeasurementSetXdt:
    """Accessor to the Measurement Set DataTree node. Provides MSv4 specific functionality
    such as:

        - get_partition_info(): produce an info dict with a general MSv4 description including
          intents, SPW name, field and source names, etc.
        - get_field_and_source_xds() to retrieve the field_and_source_xds for a given data
          group.
        - sel(): select data by dimension labels, for example by data group and polaritzation

    Registered as ``xr.DataTree.xr_ms`` without xarray's accessor cache: every
    ``xdt.xr_ms`` builds a new accessor that holds the node strongly and is
    never stored on it. Chained calls on temporaries
    (``xdt.isel(...).xr_ms.sel(...)``) therefore work, and there is no
    ``node -> accessor -> node`` reference cycle (the 2026-08 memory
    diagnosis).
    """

    def __init__(self, datatree: xr.DataTree):
        """
        Initialize the MeasurementSetXdt instance.

        Parameters
        ----------
        datatree: xarray.DataTree
            The MSv4 DataTree node to construct a MeasurementSetXdt accessor.
        """

        self._xdt: xr.DataTree = datatree
        self.meta = {"summary": {}}

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
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
            data_group_name = indexers_kwargs.pop("data_group_name")
        elif (indexers is not None) and ("data_group_name" in indexers):
            # Copy rather than edit the caller's indexers mapping
            indexers = dict(indexers)
            data_group_name = indexers.pop("data_group_name")
        else:
            data_group_name = None

        if data_group_name is not None:
            sel_data_group_set = set(
                self._xdt.attrs["data_groups"][data_group_name].values()
            ) - {"date", "description"}

            sel_field_and_source_xds = self._xdt.attrs["data_groups"][data_group_name][
                "field_and_source"
            ]

            data_variables_to_drop = []
            field_and_source_to_drop = []
            for dg in self._xdt.attrs["data_groups"].values():
                f_and_s = dg["field_and_source"]
                dg_copy = dg.copy()
                dg_copy.pop("date", None)
                dg_copy.pop("description", None)
                dg_copy.pop("field_and_source", None)
                temp_set = set(dg_copy.values()) - sel_data_group_set
                data_variables_to_drop.extend(list(temp_set))

                if f_and_s != sel_field_and_source_xds:
                    field_and_source_to_drop.append(f_and_s)

            data_variables_to_drop = list(set(data_variables_to_drop))

            # print("Data variables to drop: ", data_variables_to_drop)
            # print("Field and source to drop: ", field_and_source_to_drop)

            sel_corr_xds = self._xdt.ds.sel(
                indexers, method, tolerance, drop, **indexers_kwargs
            ).drop_vars(data_variables_to_drop)

            # Build the selection on a shallow copy of the node (and its
            # children): assigning .ds to self._xdt itself would replace the
            # data of the caller's tree.
            sel_ms_xdt = self._xdt.copy(deep=False)
            sel_ms_xdt.ds = sel_corr_xds

            # Replace the attrs mapping and copy the selected group, so that
            # nothing is shared with the caller's data_groups.
            replace_data_groups(
                sel_ms_xdt,
                {
                    data_group_name: dict(
                        self._xdt.attrs["data_groups"][data_group_name]
                    )
                },
            )

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

        data_group_name = get_data_group_name(self._xdt, data_group_name)

        field_and_source_xds_name = self._xdt.attrs["data_groups"][data_group_name][
            "field_and_source"
        ]
        return self._xdt[field_and_source_xds_name].ds

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

        data_group_name = get_data_group_name(self._xdt, data_group_name)

        field_and_source_xds = self._xdt.xr_ms.get_field_and_source_xds(data_group_name)

        if "line_name" in field_and_source_xds.coords:
            line_name = to_python_type(
                np.unique(np.ravel(field_and_source_xds.line_name.values))
            )
        else:
            line_name = []

        if "spectral_window_intents" not in self._xdt.frequency.attrs:
            spw_intent = "UNSPECIFIED"
        else:
            spw_intent = self._xdt.frequency.attrs["spectral_window_intents"]

        if "intents" in self._xdt.observation_info:
            scan_intents = self._xdt.observation_info["intents"]
        else:
            scan_intents = self._xdt.scan_name.attrs.get(
                "scan_intents", ["UNSPECIFIED"]
            )

        partition_info = {
            "spectral_window_name": self._xdt.frequency.attrs["spectral_window_name"],
            "spectral_window_intents": spw_intent,
            "field_name": to_python_type(
                np.unique(field_and_source_xds.field_name.values)
            ),
            "polarization_setup": to_python_type(self._xdt.polarization.values),
            "scan_name": to_python_type(np.unique(self._xdt.scan_name.values)),
            "source_name": to_python_type(
                np.unique(field_and_source_xds.source_name.values)
            ),
            "scan_intents": scan_intents,
            "line_name": line_name,
            "data_group_name": data_group_name,
        }

        return partition_info

    def delete_data_variables(self, variables: list[str]) -> xr.DataTree:
        """Delete data variables from the MSv4 dataset and all data groups.

        The variables are deleted from this DataTree node in place (names that
        are not data variables of the node are ignored), and every data group
        role that refers to one of them is removed. Trees that share data with
        this one (e.g. made with ``isel``, ``sel``, ``copy`` or :meth:`sel`)
        keep their variables and data groups.

        Parameters
        ----------
        variables : list of str
            List of data variable names to delete.

        Returns
        -------
        xarray.DataTree
            MSv4 DataTree with specified data variables deleted.
        """
        if self._xdt.attrs.get("type") not in MS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a MSv4 node (type {self._xdt.attrs.get('type')})."
            )

        if isinstance(variables, str):
            variables = [variables]
        variables_to_delete = [var for var in variables if var in self._xdt.data_vars]

        # Delete from the node itself: self._xdt.ds is a view built from a
        # copy of the node's variables, so deleting from it leaves the tree
        # unchanged.
        for var in variables_to_delete:
            del self._xdt[var]

        remove_variables_from_data_groups(self._xdt, variables_to_delete)

        return self._xdt

    def add_data_group(
        self,
        new_data_group_name: str,
        new_data_group: dict | None = None,
        data_group_dv_shared_with: str = None,
    ) -> xr.DataTree:
        """Adds a data group to the MSv4 DataTree, grouping the given data, weight, flag, etc. variables
        and field_and_source_xds.

        Parameters
        ----------
        new_data_group_name : str
            _description_
        new_data_group : dict
            _description_
        data_group_dv_shared_with : str, optional
            _description_, by default "base"

        Returns
        -------
        xr.DataTree
            MSv4 DataTree with the new group added
        """

        if self._xdt.attrs.get("type") not in MS_DATASET_TYPES:
            raise InvalidAccessorLocation(
                f"{self._xdt.path} is not a MSv4 node (type {self._xdt.attrs.get('type')}."
            )

        if new_data_group is None:
            new_data_group = {}

        new_data_group_name, new_data_group = create_new_data_group(
            self._xdt,
            "msv4",
            new_data_group_name,
            new_data_group,
            data_group_dv_shared_with=data_group_dv_shared_with,
        )

        # Replace the attrs mapping: the data_groups dict may be shared with
        # the trees this one was derived from (or derived into).
        replace_data_groups(
            self._xdt,
            {**self._xdt.attrs["data_groups"], new_data_group_name: new_data_group},
        )
        return self._xdt

        # data_group_dv_shared_with = get_data_group_name(
        #     self._xdt, data_group_dv_shared_with
        # )

        # default_data_group = self._xdt.attrs["data_groups"][data_group_dv_shared_with]

        # new_data_group = {}

        # if correlated_data is None:
        #     correlated_data = default_data_group["correlated_data"]
        # new_data_group["correlated_data"] = correlated_data
        # assert (
        #     correlated_data in self._xdt.ds.data_vars
        # ), f"Data variable {correlated_data} not found in dataset."

        # if weight is None:
        #     weight = default_data_group["weight"]
        # new_data_group["weight"] = weight
        # assert (
        #     weight in self._xdt.ds.data_vars
        # ), f"Data variable {weight} not found in dataset."

        # if flag is None:
        #     flag = default_data_group["flag"]
        # new_data_group["flag"] = flag
        # assert (
        #     flag in self._xdt.ds.data_vars
        # ), f"Data variable {flag} not found in dataset."

        # if self._xdt.attrs["type"] == "visibility":
        #     if uvw is None:
        #         uvw = default_data_group["uvw"]
        #     new_data_group["uvw"] = uvw
        #     assert (
        #         uvw in self._xdt.ds.data_vars
        #     ), f"Data variable {uvw} not found in dataset."

        # if field_and_source_xds is None:
        #     field_and_source_xds = default_data_group["field_and_source"]
        # new_data_group["field_and_source"] = field_and_source_xds
        # assert (
        #     field_and_source_xds in self._xdt.children
        # ), f"Data variable {field_and_source_xds} not found in dataset."

        # if date_time is None:
        #     date_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # new_data_group["date"] = date_time

        # if description is None:
        #     description = ""
        # new_data_group["description"] = description

        # self._xdt.attrs["data_groups"][new_data_group_name] = new_data_group

        # return self._xdt


register_uncached_accessor("xr_ms", xr.DataTree)(MeasurementSetXdt)
