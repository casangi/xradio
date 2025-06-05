"""
Checks that an MSv4 converted from a generated MSv2 (gen_test_ms) has:
 - the expected sizes
 - input IDs (for antennas, fields, scans, etc.)
 - data column
 - etc. structure
of the generated MSv2 given as input to the MSv2=>MSv4 converter.
"""

import numpy as np
import xarray as xr


def check_msv4_matches_msv2_descr(proc_set_xdt: xr.DataTree, msv2_descr: dict):
    """
    Parameters
    ----------
    proc_set_xdt : xr.DataTree
        A processing set DataTree containing MSv4s to be checked

    msv2_descr : dict
        MSv2 description that was used to generate a test input MSv2 in gen_test_ms

    Returns
    -------

    """
    for _msv4_name, msv4_xdt in proc_set_xdt.items():
        check_msv4_matches_descr(msv4_xdt, msv2_descr)


def check_msv4_matches_descr(msv4_xdt, msv2_descr):
    """Checks a single partition / MSv4

    Parameters
    ----------
    msv4_xdt : xr.DataTree
        An MSv4 DataTree containing one MSv4 partition

    msv2_descr : dict
        MSv2 description that was used to generate a test input MSv2 in gen_test_ms

    """
    msv2_to_msv4_data_colname = {
        "DATA": "VISIBILITY",
        "MODEL_DATA": "VISIBILITY_MODEL",
        "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    }

    msv4_xds = msv4_xdt.ds
    assert (msv2_descr["nrows_per_ddi"],) == msv4_xds.time.shape
    nantennas = len(msv2_descr["ANTENNA"])
    nbaselines = int(nantennas * (nantennas - 1) / 2)
    assert (nbaselines,) == msv4_xds.baseline_id.shape
    assert (msv2_descr["nchans"],) == msv4_xds.frequency.shape
    assert (msv2_descr["nchans"],) == msv4_xds.frequency.shape
    assert (msv2_descr["npols"],) == msv4_xds.polarization.shape

    assert "scan_name" in msv4_xds
    assert len(np.unique(msv4_xds.coords["scan_name"])) == 1

    for msv2_data_colname in msv2_descr["data_cols"]:
        msv4_data_colname = msv2_to_msv4_data_colname[msv2_data_colname]
        assert msv4_data_colname in msv4_xds.data_vars

        data_shape = msv4_xds.data_vars[msv4_data_colname].shape
        assert msv2_descr["nrows_per_ddi"] == data_shape[0]
        assert nbaselines == data_shape[1]
        assert msv2_descr["nchans"] == data_shape[2]
        assert msv2_descr["npols"] == data_shape[3]

    ant_xds = msv4_xdt["antenna_xds"].ds
    assert "antenna_name" in ant_xds
    assert len(ant_xds.coords["antenna_name"]) == nantennas
