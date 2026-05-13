import xarray as xr
from xradio._utils.schema import get_data_group_keys
from collections.abc import Mapping, Iterable
from typing import Any, Union


def get_data_group_name(
    xdx: Union[xr.Dataset, xr.DataTree], data_group_name: str = None
) -> str:

    if data_group_name is None:
        if "base" in xdx.attrs["data_groups"]:
            data_group_name = "base"
        else:
            data_group_name = list(xdx.attrs["data_groups"].keys())[0]

    return data_group_name


def create_new_data_group(
    xdx: Union[xr.Dataset, xr.DataTree],
    schema_name: str,
    new_data_group_name: str,
    data_group: dict,
    data_group_dv_shared_with: str = None,
) -> tuple[str, dict]:
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
    tuple[str, dict]
        A tuple containing the name of the new data group and the dictionary with its variables and attributes.
    """

    data_group_dv_shared_with = get_data_group_name(xdx, data_group_dv_shared_with)

    default_data_group = xdx.attrs["data_groups"][data_group_dv_shared_with]

    new_data_group = {}

    data_group_keys = get_data_group_keys(schema_name)

    for key, optional in data_group_keys.items():
        if key in data_group:
            new_data_group[key] = data_group[key]
        else:
            if key not in default_data_group and not optional:
                raise ValueError(
                    f"Data group key '{key}' is required but not provided and not present in shared data group '{data_group_dv_shared_with}'."
                )
            elif key in default_data_group:
                new_data_group[key] = default_data_group[key]

    return new_data_group_name, new_data_group


def delete_data_variables(xdx: Union[xr.Dataset, xr.DataTree], variables: list):
    """Deletes data variables from an Xarray Dataset or DataTree.

    Parameters
    ----------
    xdx : Union[xr.Dataset, xr.DataTree]
        The Xarray Dataset or DataTree to delete data variables from.
    variables : list
        A list of variable names to delete.
    Returns
    -------
    """
    
    for var in variables:
        if var in xdx.data_vars:
            del xdx[var]
        else:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
    
    for data_group_name in xdx.attrs["data_groups"]:
        data_group = xdx.attrs["data_groups"][data_group_name]
        for var in variables:
            if var in data_group:
                del data_group[var]