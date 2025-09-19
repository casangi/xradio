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


def check_msv4_matches_descr(msv4_xdt, msv2_descr):
    """
    Checks a single partition / MSv4

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

    if msv2_descr["params"]["misbehave"]:
        expected_type = "field_and_source"
    else:
        expected_type = "field_and_source_ephemeris"

    field_and_source_name = "field_and_source_base_xds"
    assert (
        field_and_source_name
        == msv4_xdt.ds.attrs["data_groups"]["base"]["field_and_source"]
    )
    assert field_and_source_name in msv4_xdt
    assert msv4_xdt[field_and_source_name].ds.attrs["type"] == expected_type

    if msv2_descr["params"]["opt_tables"]:
        assert "system_calibration_xds" in msv4_xdt
        assert "weather_xds" in msv4_xdt
        if not msv2_descr["params"]["misbehave"]:
            assert "SOURCE_DIRECTION" in msv4_xdt["field_and_source_base_xds"].ds
            assert (
                msv4_xdt["field_and_source_base_xds"].attrs["type"]
                == "field_and_source_ephemeris"
            )

    if msv2_descr["params"]["vlbi_tables"]:
        assert "gain_curve_xds" in msv4_xdt
        assert "phase_calibration_xds" in msv4_xdt

    partition_info = msv4_xdt.xr_ms.get_partition_info()
    processor_info = msv4_xdt.ds.attrs["processor_info"]
    if msv2_descr["params"]["misbehave"]:
        # SPW names should be empty string in MSv2
        assert partition_info["spectral_window_name"] == "spw_0"
        assert not processor_info["type"]
        assert not processor_info["sub_type"]
    else:
        assert partition_info["spectral_window_name"]
        assert processor_info["type"]
        assert processor_info["sub_type"]

    observation_info = msv4_xdt.ds.attrs["observation_info"]
    if msv2_descr["params"]["opt_tables"] and not msv2_descr["params"]["misbehave"]:
        assert "session_reference_UID" in observation_info
        assert "observing_script" in observation_info
    else:
        assert "session_reference_UID" not in observation_info
        assert "observing_script" not in observation_info

    if (
        (msv2_descr["params"]["opt_tables"] and not msv2_descr["params"]["misbehave"])
    ) or "OBSERVATION" in msv2_descr:
        assert "execution_block_UID" in observation_info
        assert "scheduling_block_UID" in observation_info
    else:
        assert "execution_block_UID" not in observation_info
        assert "scheduling_block_UID" not in observation_info


def check_processing_set_matches_msv2_descr(
    proc_set_xdt: xr.DataTree, msv2_descr: dict
):
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
