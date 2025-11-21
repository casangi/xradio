from typing import Union
import xarray as xr
from xradio._utils.schema import get_data_group_keys


def get_data_group_name(
    xdx: Union[xr.Dataset, xr.DataTree], data_group_name: str = None
) -> str:

    if data_group_name is None:
        if "base" in xdx.attrs["data_groups"]:
            data_group_name = "base"
        else:
            data_group_name = list(xdx.attrs["data_groups"].keys())[0]

    return data_group_name


def add_data_group(
    xdx: Union[xr.Dataset, xr.DataTree],
    schema_name: str,
    new_data_group_name: str,
    data_group: dict,
    data_group_dv_shared_with: str = None,
) -> xr.DataTree:
    """Adds a data group to Xarray Data Structure (Dataset or DataTree).

    Parameters
    ----------
    new_data_group_name : str
        The name of the new data group to add.
    data_group : dict
        A dictionary containing the data group variables and their attributes.
    data_group_dv_shared_with : str, optional
        The name of the data group to share data variables with, by default "base"

    Returns
    -------
    xr.DataTree
        MSv4 DataTree with the new group added
    """

    data_group_dv_shared_with = get_data_group_name(xdx, data_group_dv_shared_with)

    default_data_group = xdx.attrs["data_groups"][data_group_dv_shared_with]

    new_data_group = {}

    data_group_keys = get_data_group_keys(schema_name)

    for key in data_group_keys:
        if key in data_group:
            new_data_group[key] = data_group[key]
        else:
            new_data_group[key] = default_data_group[key]

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
