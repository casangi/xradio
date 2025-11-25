from collections.abc import Mapping, Iterable
import datetime
from typing import Any, Union

import numpy as np
import xarray as xr

from xradio._utils.list_and_array import to_list

IMAGE_DATASET_TYPES = {"image"}

from xradio._utils.xarray_helpers import get_data_group_name, create_new_data_group


class InvalidAccessorLocation(ValueError):
    """
    Raised by ImageXds accessor functions called on a wrong Dataset (not image).
    """

    pass


@xr.register_dataset_accessor("xr_img")
class ImageXds:
    """Accessor to the Image Dataset."""

    _xds: xr.Dataset

    def __init__(self, dataset: xr.Dataset):
        """
        Initialize the ImageXds instance.

        Parameters
        ----------
        dataset: xarray.Dataset
            The image Dataset node to construct an ImageXds accessor.
        """

        self._xds = dataset
        self.meta = {"summary": {}}

    def test_func(self):
        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")

        return "Hallo"

    def add_data_group(
        self,
        new_data_group_name: str,
        new_data_group: dict = {},
        data_group_dv_shared_with: str = None,
    ) -> xr.Dataset:
        """Adds a data group to the image Dataset, grouping the given data, weight, flag, etc. variables
        and field_and_source_xds.

        Parameters
        ----------
        new_data_group_name : str
            _description_
        new_data_group : dict
            _description_, by default Non
        data_group_dv_shared_with : str, optional
            _description_, by default "base"

        Returns
        -------
        xr.Dataset
          Image Dataset with the new group added
        """

        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")

        new_data_group_name, new_data_group = create_new_data_group(
            self._xds,
            "image",
            new_data_group_name,
            new_data_group,
            data_group_dv_shared_with=data_group_dv_shared_with,
        )

        self._xds.attrs["data_groups"][new_data_group_name] = new_data_group
        return self._xds

    def sel(
        self,
        indexers: Union[Mapping[Any, Any], None] = None,
        method: Union[str, None] = None,
        tolerance: Union[int, float, Iterable[Union[int, float]], None] = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xr.Dataset:
        """
        Select data along dimension(s) by label. Alternative to `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ so that a data group can be selected by name by using the `data_group_name` parameter.
        For more information on data groups see `Data Groups <https://xradio.readthedocs.io/en/latest/measurement_set_overview.html#Data-Groups>`__ section. See `xarray.Dataset.sel <https://xarray.pydata.org/en/stable/generated/xarray.Dataset.sel.html>`__ for parameter descriptions.

        Returns
        -------
        xarray.Dataset
            xarray Dataset with ImageXds accessors

        Examples
        --------
        >>> # Select data group 'robust0.5' and polarization 'XX'.
        >>> selected_img_xds = img_xds.xr_img.sel(data_group_name='robust0.5', polarization='XX')
        """

        if self._xds.attrs.get("type") not in IMAGE_DATASET_TYPES:
            raise InvalidAccessorLocation(f"{self._xds.path} is not a image node.")

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
                self._xds.attrs["data_groups"][data_group_name].values()
            ) - set(["date", "description"])

            data_variables_to_drop = []
            for dg_name, dg in self._xds.attrs["data_groups"].items():
                # print(f"Data group: {dg_name}", dg)
                dg_copy = dg.copy()
                dg_copy.pop("date", None)
                dg_copy.pop("description", None)
                temp_set = set(dg_copy.values()) - sel_data_group_set
                data_variables_to_drop.extend(list(temp_set))

            data_variables_to_drop = list(set(data_variables_to_drop))

            sel_img_xds = self._xds

            sel_corr_xds = self._xds.ds.sel(
                indexers, method, tolerance, drop, **indexers_kwargs
            ).drop_vars(data_variables_to_drop)

            sel_img_xds.ds = sel_corr_xds

            sel_img_xds.attrs["data_groups"] = {
                data_group_name: self._xds.attrs["data_groups"][data_group_name]
            }

            return sel_img_xds
        else:
            return self._xds.sel(indexers, method, tolerance, drop, **indexers_kwargs)
