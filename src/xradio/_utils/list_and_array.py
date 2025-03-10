"""Contains optimised functions to be used within other modules."""

import numpy as np
import xarray as xr
import pandas as pd


def cast_to_str(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def get_pad_value(col_dtype: np.dtype) -> object:
    """
    Produce a padding/missing/nan value appropriate for a casacore data column
    (for when we need to pad data vars coming from columns with rows of
    variable size array values)

    Parameters
    ----------
    col_dtype : dtype
        dtype of data being loaded from a table column

    Returns
    -------
    object
        pad value ("missing" / "fill") for the type given
    """
    # Fill values for missing/NaN data in integer variables, based on usual
    # numpy fill values. See https://github.com/numpy/numpy/issues/21166,
    # https://github.com/casangi/xradio/issues/219, https://github.com/casangi/xradio/pull/177
    fill_value_int32 = np.int32(-2147483648)
    fill_value_int64 = np.int64(-9223372036854775808)

    if col_dtype == np.int32:
        return fill_value_int32
    elif col_dtype == np.int64 or col_dtype == "int":
        return fill_value_int64
    elif np.issubdtype(col_dtype, np.floating):
        return np.nan
    elif np.issubdtype(col_dtype, np.complexfloating):
        return complex(np.nan, np.nan)
    elif np.issubdtype(col_dtype, np.bool_):
        return False
    elif np.issubdtype(col_dtype, str):
        return ""
    else:
        raise RuntimeError(
            "Padding / missing value not defined for the type requested: "
            f"{col_dtype} (of type: {type(col_dtype)})"
        )


def to_list(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return [x.item()]
        return list(x)  # needed for json serialization
    elif isinstance(x, list):
        return x
    return [x]


def to_np_array(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.array([x.item()])
        return np.array(x)  # needed for json serialization
    elif isinstance(x, list):
        return np.array(x)
    return np.array([x])


def check_if_consistent(array: np.ndarray, array_name: str) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    array : _type_
        _description_
    array_name : _type_
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

    return np.sort(
        pd.unique(array)
    )  # Don't remove the sort! It will cause errors that are very difficult to detect. Specifically create_field_info_and_check_ephemeris has a TaQL query that requires this.


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
