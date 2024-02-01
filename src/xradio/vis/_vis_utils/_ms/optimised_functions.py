"""Contains optimised functions to be used within other modules."""

import numpy as np
import pandas as pd


def unique_1d(array: np.ndarray) -> np.ndarray:
    """Optimised version of np.unique for 1D arrays.

    Args:
        array (np.ndarray): a 1D array of values.

    Returns:
        np.ndarray: a sorted array of unique values.
    """
    return np.sort(pd.unique(array))


def pairing_function(antenna_pairs: np.ndarray) -> np.ndarray:
    """Pairing function to convert each array pair to a single value.

    This custom pairing function will only work if the maximum value is less
    than 2**20 and less than 2,048 if using signed 32-bit integers.

    Args:
        antenna_pairs (np.ndarray): a 2D array containing antenna 1 and antenna
        2 ids, which forms a baseline.

    Returns:
        np.ndarray: a 1D array of the paired values.
    """
    return antenna_pairs[:, 0] * 2**20 + antenna_pairs[:, 1]


def inverse_pairing_function(paired_array: np.ndarray) -> np.ndarray:
    """Inverse pairing function to convert each paired value to an antenna pair.

    This inverse pairing function is the inverse of the custom pairing function.

    Args:
        paired_array (np.ndarray): a 1D array of the paired values.

    Returns:
        np.ndarray: a 2D array containing antenna 1 and antenna 2 ids.
    """
    return np.column_stack(np.divmod(paired_array, 2**20))
