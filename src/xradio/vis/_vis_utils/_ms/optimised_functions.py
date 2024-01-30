"""Contains optimised functions to be used within other modules."""


########################################################################
# list of affected files:
# - src/xradio/vis/_vis_utils/_ms/conversion.py
# - src/xradio/vis/_vis_utils/_ms/_tables/read_main_table.py
# - src/xradio/vis/_vis_utils/_ms/_tables/read.py
# - src/xradio/vis/_vis_utils/_ms/partition_queries.py


########################################################################
# function changes to be made:


# ts_bases = [
#         str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
#         for ll in np.hstack([ts_ant1[:, None], ts_ant2[:, None]])
#     ] ->
# ts_bases -> antennas_to_baselines(ts_ant1, ts_ant2)


# baseline_ant1_id, baseline_ant2_id = np.array(
#         [tuple(map(int, x.split("_"))) for x in baselines]
#     ).T ->
# baseline_ant1_id, baseline_ant2_id = baselines_to_antennas()


########################################################################

# NOTE: the function get_baselines() in read_main_table module is distinct
# from the above operation baselines_to_antennas() as it first gets the unique
# antennas in each column. The function get_baselines() can either be brought
# into this module, or optimised in place


########################################################################
# functions to test
from typing import Union
import numpy as np
import pandas as pd
from casacore import tables


def unique_1d(array: Union[np.ndarray, list]) -> np.ndarray:
    """Optimised version of np.unique for 1D arrays.

    Args:
        array (np.ndarray/list): a 1D array or list of values.

    Returns:
        np.ndarray: a sorted array of unique values.
    """
    return np.sort(pd.unique(array))


def searchsorted(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.searchsorted(arr1, arr2)


# Optimised function
# @nb.njit(parallel=False, fastmath=True)
# def searchsorted(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
#     res = np.empty(len(arr2), np.intp)
#     for i in nb.prange(len(arr2)):
#         res[i] = np.searchsorted(arr1, arr2[i])
#     return res


# returns a np.ndarray with dtype=str
def antennas_to_baselines(ant1: np.ndarray, ant2: np.ndarray) -> np.ndarray:
    baselines = [
        str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
        for ll in np.hstack([ant1[:, None], ant2[:, None]])
    ]
    return baselines


def get_baselines(tb_tool: tables.table) -> np.ndarray:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = tb_tool.getcol("ANTENNA1", 0, -1), tb_tool.getcol("ANTENNA2", 0, -1)

    baselines = np.unique(np.column_stack((ant1, ant2)), axis=0)

    return baselines
