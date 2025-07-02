from typing import Tuple

import numpy as np
import pandas as pd

try:
    from casacore import tables
except ImportError:
    import xradio._utils._casacore.casacore_from_casatools as tables


from .table_query import open_query
from xradio._utils.list_and_array import (
    unique_1d,
    pairing_function,
    inverse_pairing_function,
)


def get_utimes_tol(mtable: tables.table, taql_where: str) -> Tuple[np.ndarray, float]:
    taql_utimes = f"select DISTINCT TIME from $mtable {taql_where}"
    with open_query(mtable, taql_utimes) as query_utimes:
        utimes = unique_1d(query_utimes.getcol("TIME", 0, -1))
        # add a tol around the time ranges returned by taql
        if len(utimes) < 2:
            tol = 1e-5
        else:
            tol = np.diff(utimes).min() / 4

    return utimes, tol


def get_baselines(tb_tool: tables.table) -> np.ndarray:
    """
    Gets the unique baselines from antenna 1 and antenna 2 ids.

    Uses a pairing function and inverse pairing function to decrease the
    computation time of finding unique values.

    Parameters
    ----------
    tb_tool : tables.table
        MeasurementSet table to get the antenna ids.

    Returns
    -------
    unique_baselines : np.ndarray
        a 2D array of unique antenna pairs
        (baselines) from the MeasurementSet table provided.
    """
    ant1, ant2 = tb_tool.getcol("ANTENNA1", 0, -1), tb_tool.getcol("ANTENNA2", 0, -1)

    baselines = np.column_stack((ant1, ant2))

    # Using pairing function to reduce the computation time of finding unique values.
    baselines_paired = pairing_function(baselines)
    unique_baselines_paired = pd.unique(baselines_paired)
    unique_baselines = inverse_pairing_function(unique_baselines_paired)

    # Sorting the unique baselines.
    unique_baselines = unique_baselines[unique_baselines[:, 1].argsort()]
    unique_baselines = unique_baselines[
        unique_baselines[:, 0].argsort(kind="mergesort")
    ]

    return unique_baselines


def get_baseline_indices(
    unique_baselines: np.ndarray, baseline_set: np.ndarray
) -> np.ndarray:
    """
    Finds the baseline indices of a set of baselines using the unique baselines.

    Uses a pairing function to reduce the number of values so it's more
    efficient to find the indices.

    Parameters
    ----------
    unique_baselines : np.ndarray
        a 2D array of unique antenna pairs (baselines).
    baseline_set : np.ndarray
        a 2D array of antenna pairs (baselines). This array may contain duplicates.

    Returns
    -------
    baseline_indices : np.ndarray
        the indices of the baseline set that
        correspond to the unique baselines.
    """
    unique_baselines_paired = pairing_function(unique_baselines)
    baseline_set_paired = pairing_function(baseline_set)

    # Pairing function doesn't preserve order so they need to be sorted.
    unique_baselines_sorted = np.argsort(unique_baselines_paired)
    sorted_indices = np.searchsorted(
        unique_baselines_paired[unique_baselines_sorted],
        baseline_set_paired,
    )
    baseline_indices = unique_baselines_sorted[sorted_indices]

    return baseline_indices
