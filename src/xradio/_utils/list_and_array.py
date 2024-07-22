"""Contains optimised functions to be used within other modules."""

import numpy as np
import xarray as xr


def to_list(x):
    if isinstance(x, (list, np.ndarray)):
        if x.ndim == 0:
            return [x.item()]
        return list(x)  # needed for json serialization
    return [x]


def to_np_array(x):
    if isinstance(x, (list, np.ndarray)):
        if x.ndim == 0:
            return np.array([x.item()])
        return np.array(x)  # needed for json serialization
    return np.array([x])


def check_if_consistent(array: np.ndarray, array_name: str) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    col : _type_
        _description_
    col_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if array.ndim == 0:
        return array.item()

    array_unique = unique_1d(array)
    assert len(array_unique) == 1, array_name + " is not consistent."
    return array_unique[0]


import numpy as np
import pandas as pd


def unique_1d(array: np.ndarray) -> np.ndarray:
    """
    Optimised version of np.unique for 1D arrays.

    Parameters
    ----------
    array : np.ndarray
        a 1D array of values.

    Returns
    -------
    np.ndarray
        a sorted array of unique values.

    """
    if isinstance(array, xr.core.dataarray.DataArray):
        array = array.values

    if array.ndim == 0:
        return np.array([array.item()])

    return np.sort(pd.unique(array))


def pairing_function(antenna_pairs: np.ndarray) -> np.ndarray:
    """
    Pairing function to convert each array pair to a single value.

    This custom pairing function will only work if the maximum value is less
    than 2**20 and less than 2,048 if using signed 32-bit integers.

    Parameters
    ----------
    antenna_pairs : np.ndarray
        a 2D array containing antenna 1 and antenna
        2 ids, which forms a baseline.

    Returns
    -------
    np.ndarray
        a 1D array of the paired values.

    """
    return antenna_pairs[:, 0] * 2**20 + antenna_pairs[:, 1]


def inverse_pairing_function(paired_array: np.ndarray) -> np.ndarray:
    """
    Inverse pairing function to convert each paired value to an antenna pair.

    This inverse pairing function is the inverse of the custom pairing function.

    Parameters
    ----------
    paired_array : np.ndarray
        a 1D array of the paired values.

    Returns
    -------
    np.ndarray
        a 2D array containing antenna 1 and antenna 2 ids.

    """
    return np.column_stack(np.divmod(paired_array, 2**20))
