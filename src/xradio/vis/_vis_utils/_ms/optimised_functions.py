"""Contains optimised functions to be used within other modules."""

from typing import Union
import numpy as np
import pandas as pd


def unique_1d(array: Union[np.ndarray, list]) -> np.ndarray:
    """Optimised version of np.unique for 1D arrays.

    Args:
        array (np.ndarray/list): a 1D array or list of values.

    Returns:
        np.ndarray: a sorted array of unique values.
    """
    return np.sort(pd.unique(array))


def pairing_function(antenna_pairs):
    return antenna_pairs[:, 0] * 2**20 + antenna_pairs[:, 1]


def inverse_pairing_function(x):
    return np.column_stack(np.divmod(x, 2**20))
