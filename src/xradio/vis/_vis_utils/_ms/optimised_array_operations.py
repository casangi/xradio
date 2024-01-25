########################################################################
# module which contains optimised functions to be called by other utilities


########################################################################
# list of affected files:
# - src/xradio/vis/_vis_utils/_ms/conversion.py
# - src/xradio/vis/_vis_utils/_ms/_tables/read_main_table.py
# - src/xradio/vis/_vis_utils/_ms/_tables/read.py
# - src/xradio/vis/_vis_utils/_ms/partition_queries.py


########################################################################
# function changes to be made:


# np.unique() -> unique()


# ts_bases = [
#         str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
#         for ll in np.hstack([ts_ant1[:, None], ts_ant2[:, None]])
#     ] ->
# ts_bases -> antennas_to_baselines(ts_ant1, ts_ant2)


# np.searchsorted() -> searchsorted()


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
import numpy as np
import pandas as pd
from casacore import tables
# TODO check numba is a dependency
import numba as nb



def unique(arr):
    return np.unique(arr)

# Optimised function
# def unique(arr):
#     return np.sort(pd.unique(arr))




# returns a np.ndarray with dtype=str
def antennas_to_baselines(ant1: np.ndarray, ant2: np.ndarray) -> np.ndarray:
    baselines = [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.hstack([ant1[:, None], ant2[:, None]])
        ]
    return baselines

# Optimised function
# returns a np.ndarray with dtype=np.int32
# @nb.njit(parallel=False, fastmath=True)
# def antennas_to_baselines(ant1, ant2):
#     all_baselines = np.empty(ant1.size, dtype=np.int32)
    
#     for i in nb.prange(len(ant1)):
#         # This is a Cantor pairing funciton
#         # max_num_antenna_pairs required to give expected ordering (20000 may be too small for certain SKA-low observations)
#         # see https://math.stackexchange.com/questions/3212587/does-there-exist-a-pairing-function-which-preserves-ordering
#         # and https://math.stackexchange.com/questions/3969617/what-pairing-function-coincides-with-the-g%C3%B6del-pairing-on-the-natural-numbers
#         max_num_antenna_pairs = 20000
#         all_baselines[i] = ((ant1[i] + ant2[i]) * (ant1[i] + ant2[i] + 1)) // 2 + max_num_antenna_pairs*ant1[i]



def searchsorted(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.searchsorted(arr1, arr2)

# Optimised function
# @nb.njit(parallel=False, fastmath=True)
# def searchsorted(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
#     res = np.empty(len(arr2), np.intp)
#     for i in nb.prange(len(arr2)):
#         res[i] = np.searchsorted(arr1, arr2[i])
#     return res
    



########################################################################
# special case for get_baselines() in read_main_table
# will use a combination of functions

# current function (copied from read_main_table.py)
def get_baselines(tb_tool: tables.table) -> np.ndarray:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = tb_tool.getcol("ANTENNA1", 0, -1), tb_tool.getcol("ANTENNA2", 0, -1)
        
    # TODO
    # swap np.unique(arr) for np.sort(pd.unique(arr))
    # swap string baseline identifiers with integer based cantor pairing
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(np.hstack([ant1[:, None], ant2[:, None]]), axis=0)
        ]
    )

    return baselines


# Optimised function
# get_baselines(tb_tool: tables.table) -> np.ndarray:
#     # main table uses time x (antenna1,antenna2)
#     ant1, ant2 = tb_tool.getcol("ANTENNA1", 0, -1), tb_tool.getcol("ANTENNA2", 0, -1)
#     
#     baselines = np.sort(np.unique(antennas_to_baselines(ant1, ant2)))
#     
#     return baselines
# TODO check above function is correct. Only quickly scanned through msconverter and may be wrong