import numpy as np
import os
import pathlib
import pytest
import time
import warnings

# pytest.skip(
#     "Skipping because we are debugging another set of tests and these take a while to run.",
#     allow_module_level=True,
# )

import pandas as pd
import xarray as xr

from toolviper.utils.data import download
import toolviper.utils.logger as logger
from xradio.measurement_set import (
    open_processing_set,
    load_processing_set,
    convert_msv2_to_processing_set,
    estimate_conversion_memory_and_cores,
    MeasurementSetXdt,
    ProcessingSetXdt,
)
from xradio.schema.check import check_datatree

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


# Uncomment to not clean up files between test (i.e. skip downloading them again)
@pytest.fixture
def tmp_path():
    return pathlib.Path("/tmp/test")


def download_and_convert_msv2_to_processing_set(
    msv2_name, folder, partition_scheme, parallel_mode: str = "partition"
):

    _logger_name = "xradio"
    if os.getenv("VIPER_LOGGER_NAME") != _logger_name:
        os.environ["VIPER_LOGGER_NAME"] = _logger_name
        logger.setup_logger(
            logger_name="xradio",
            log_to_term=True,
            log_to_file=False,  # True
            log_file="xradio-logfile",
            log_level="DEBUG",
            # log_level="INFO",
        )

    download(file=msv2_name, folder=str(folder))
    ps_name = folder / (msv2_name[:-3] + ".ps.zarr")

    if os.path.isdir(ps_name):
        os.system("rm -rf " + str(ps_name))  # Remove ps folder.

    estimates = estimate_conversion_memory_and_cores(
        str(folder / msv2_name), partition_scheme=partition_scheme
    )
    mem_estimate = estimates[0]
    assert mem_estimate < 0.1, f"Too high estimate: {mem_estimate}"
    # test_sd_A002_X1015532_X1926f is the smallest so far
    assert mem_estimate > 6.5e-5, f"Too low estimate: {mem_estimate}"

    convert_msv2_to_processing_set(
        in_file=str(folder / msv2_name),
        out_file=ps_name,
        partition_scheme=partition_scheme,
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        # phase_cal_interpolate=True,
        # sys_cal_interpolate=True,
        use_table_iter=False,
        overwrite=True,
        parallel_mode=parallel_mode,
    )
    return ps_name


def check_expected_datasets_presence(ps_xdt, expected_secondary_xds: set[str]):
    """
    expected_secondary_xds should be for example {"antenna", "weather"}, {"antenna", "pointing", "weather"}, or
    {"antenna", "gain_curve", "system_calibration"}.
    The "_xds" suffix is not needed.
    """

    def check_xds_in_datatree(xds, msv4_xdt):
        assert xds in msv4_xdt
        assert isinstance(msv4_xdt[xds], xr.DataTree)
        assert isinstance(msv4_xdt[xds].ds, xr.Dataset)

    if not expected_secondary_xds:
        return

    for _msv4_xds_name, msv4_xdt in ps_xdt.items():
        for xds in expected_secondary_xds:
            if not xds.endswith("_xds"):
                xds = xds + "_xds"

            # system_calibration is only present for proper visibility data (not for RADIOMETER, wvr and the like)
            if xds != "system_calibration_xds":
                check_xds_in_datatree(xds, msv4_xdt)
            else:
                if msv4_xdt.ds.processor_info["type"] == "CORRELATOR":
                    check_xds_in_datatree(xds, msv4_xdt)


def base_check_ps_accessor(ps_lazy_xdt: xr.DataTree, ps_xdt: xr.DataTree):
    """
    Basic checks on the `ps` accessor of processing sets ps_xdt/ps_lazy_xdt

    Parameters
    ----------
    ps_lazy_xdt: lazy processing set (created with open_processing_set or equivalent)
    ps_xdt: in memory processing set (created with load_processing_set or equivalent)

    """

    for top_xdt in [ps_lazy_xdt, ps_xdt]:
        assert hasattr(top_xdt, "xr_ps") and isinstance(top_xdt.xr_ps, ProcessingSetXdt)
        assert "type" in top_xdt.attrs and top_xdt.attrs["type"] == "processing_set"

    expected_summary_keys = [
        "name",
        "intents",
        "shape",
        "polarization",
        "scan_name",
        "spw_name",
        "spw_intent",
        "field_name",
        "source_name",
        "line_name",
        "field_coords",
        "start_frequency",
        "end_frequency",
    ]
    ps_lazy_xdt_df = ps_lazy_xdt.xr_ps.summary()
    assert all([key in ps_lazy_xdt_df for key in expected_summary_keys])
    ps_xdt_df = ps_xdt.xr_ps.summary()
    assert all([key in ps_xdt_df for key in expected_summary_keys])
    pd.testing.assert_frame_equal(ps_lazy_xdt_df, ps_xdt_df)

    expected_dims = [
        "time",
        "frequency",
        "polarization",
    ]
    is_single_dish = "spectrum" in [
        ps_lazy_xdt[name].attrs["type"] for name in ps_lazy_xdt
    ]
    if not is_single_dish:
        expected_dims += {"baseline_id", "uvw_label"}
    else:
        expected_dims += {"antenna_name"}
    max_dims = ps_xdt.xr_ps.get_max_dims()
    assert isinstance(max_dims, dict)
    assert all([dim in max_dims for dim in expected_dims])

    empty_query_result = ps_xdt.xr_ps.query()
    assert isinstance(empty_query_result, xr.DataTree)
    empty_query_df = empty_query_result.xr_ps.summary()
    pd.testing.assert_frame_equal(ps_xdt_df, empty_query_df)

    field_query_result = ps_xdt.xr_ps.query(field_name=ps_xdt_df.field_name.values[0])
    assert isinstance(field_query_result, xr.DataTree)
    data_group_query_result = ps_xdt.xr_ps.query(data_group_name="base")
    data_group_query_df = data_group_query_result.xr_ps.summary()
    pd.testing.assert_frame_equal(ps_xdt_df, data_group_query_df)

    freq_axis = ps_xdt.xr_ps.get_freq_axis()
    assert isinstance(freq_axis, xr.DataArray)

    combined_field_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    assert type(combined_field_xds) == xr.Dataset
    combined_antenna = ps_xdt.xr_ps.get_combined_antenna_xds()
    assert type(combined_antenna) == xr.Dataset
    base_field_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds("base")
    assert type(base_field_xds) == xr.Dataset

    try:
        import matplotlib
        from matplotlib import pyplot as plt

        matplotlib.use("Agg")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="FigureCanvasAgg is non-interactive",
            )
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="No artists",
            )

            with plt.ioff():
                label_all_fields = label_all_antennas = len(ps_xdt_df) > 1
                ps_xdt.xr_ps.plot_phase_centers(label_all_fields=label_all_fields)
                plt.close()
                ps_xdt.xr_ps.plot_antenna_positions(
                    label_all_antennas=label_all_antennas
                )
                plt.close()

    except Exception as exc:
        logger.warning(
            f"Could not run processing set plot functions, exception details: {exc}"
        )


def base_check_ms_accessor(ps_xdt: xr.DataTree):
    """
    Basic checks on the children of ps_xdt (ms_xdt trees) and their `.ds` and `.ms` accessor
    """
    for ms_xds_name in ps_xdt.keys():
        ms_xdt = ps_xdt[ms_xds_name]
        assert "type" in ms_xdt.attrs and ms_xdt.attrs["type"] in [
            "visibility",
            "radiometer",
            "spectrum",
        ]

        # dt produces a DatasetView
        assert hasattr(ms_xdt, "ds") and isinstance(ms_xdt.ds, xr.Dataset)
        assert ms_xdt["antenna_xds"]
        assert ms_xdt["field_and_source_base_xds"]
        # Should check depending on availability of metadata in input MSv2:
        # assert ms_xdt["gain_curve_xds"]
        # assert ms_xdt["phase_calibration_xds"]
        # assert ms_xdt["pointing_xds"]
        # assert ms_xdt["system_calibration_xds"]
        # assert ms_xdt["weather_xds"]
        # assert ms_xdt.ds  # DatasetView

        assert hasattr(ms_xdt, "xr_ms") and isinstance(ms_xdt.xr_ms, MeasurementSetXdt)
        assert (
            hasattr(ms_xdt.xr_ms, "get_field_and_source_xds")
            and callable(ms_xdt.xr_ms.get_field_and_source_xds)
            and isinstance(ms_xdt.xr_ms.get_field_and_source_xds(), xr.Dataset)
        )
        assert (
            hasattr(ms_xdt.xr_ms, "get_partition_info")
            and callable(ms_xdt.xr_ms.get_partition_info)
            and isinstance(ms_xdt.xr_ms.get_partition_info(), dict)
        )
        assert (
            hasattr(ms_xdt.xr_ms, "sel")
            and callable(ms_xdt.xr_ms.sel)
            and isinstance(ms_xdt.xr_ms.sel(), xr.DataTree)
        )


def base_check_time_centroid(ms_xds: xr.Dataset):
    """
    Basic consistency check of TIME_CENTROID values against time. Checks that the TIME_CENTROID
    values are not too off the time centers (within the integration time).

    Note: the maximum allowed diff between TIME_CENTROID and time ('mid-point of the nominal
    sampling interval') can/should? be `<= int_time / 2.0`, except for the test_vlass
    dataset which for some reason has time diffs close to the int_time
    (~7.64s diff, int_time=~8.1s).
    """
    if "TIME_CENTROID" in ms_xds:
        time_centroid = ms_xds["TIME_CENTROID"]
        int_time = ms_xds.time.attrs["integration_time"]["data"]
        time_diff = time_centroid - time_centroid.time
        assert np.max(time_diff) <= int_time


def base_check_time_secondary_datasets(ps_xdt: xr.DataTree):
    """
    Assert basic consistency checks of the time_* coordinates of subdatasets:
    - pointing_xds, weather_xds, system_calibration_xds, and phase_calibration_xds.

    Ensures that the secondary xds specific time values are not too off the (main)
    time values of the correlated dataset (+/- a simple buffer).
    """

    def get_ps_time_min_max(ps_xdt: xr.DataTree) -> tuple[np.float64, np.float64]:
        """Finds min/max of the time coord across all the MSv4 of the PS"""
        ps_time_min, ps_time_max = 1.5e308, -1.5e308
        for ms_name, ms_xdt in ps_xdt.items():
            ms_time_min, ms_time_max = ms_xdt.time.min(), ms_xdt.time.max()
            if ms_time_min < ps_time_min:
                ps_time_min = ms_time_min
            if ms_time_max > ps_time_max:
                ps_time_max = ms_time_max

        return ps_time_min, ps_time_max

    time_coords = {
        "phase_calibration_xds": "time_phase_cal",
        "pointing_xds": "time_pointing",
        "system_calibration_xds": "time_sys_cal",
        "weather_xds": "time_weather",
    }

    # Some datasets need bigger buffers (example: weather table with entries a
    # few hours or even days before/after the correlated data time range of a particular MSv4).
    time_buffer = {
        "time_phase_cal": 45,
        "time_pointing_xds": 1,
        "time_system_cal": 1,
        # For test_vlba (VLBA_TL016B_split) for example:
        "time_weather": 169000,
    }
    ps_time_min, ps_time_max = get_ps_time_min_max(ps_xdt)
    for _ms_name, ms_xdt in ps_xdt.items():
        for xds_name, time_name in time_coords.items():
            if xds_name in ms_xdt:
                if time_name in ms_xdt[xds_name].coords:
                    xds_time = ms_xdt[xds_name].coords[time_name]
                    assert xds_time.min() >= ps_time_min - time_buffer[time_name]
                    assert xds_time.max() <= ps_time_max + time_buffer[time_name]


def base_test(
    file_name: str,
    folder: pathlib.Path,
    expected_sum_value: float,
    is_s3: bool = False,
    partition_schemes: list = [[], ["FIELD_ID"]],
    parallel_mode: str = "none",
    preconverted: bool = False,
    do_schema_check: bool = True,
    expected_secondary_xds: set = None,
):
    start = time.time()

    from toolviper.dask.client import local_client

    viper_client = local_client(
        cores=2, memory_limit="3GB"
    )  ##Do not increase size otherwise GitHub MacOS runner will hang.

    viper_client

    ps_list = (
        []
    )  # Create a list of PS for each partition scheme. This will be returned.
    for partition_scheme in partition_schemes:
        if is_s3:
            ps_name = file_name
        elif preconverted:
            download(file=file_name)
            ps_name = file_name
        else:
            ps_name = download_and_convert_msv2_to_processing_set(
                file_name, folder, partition_scheme, parallel_mode=parallel_mode
            )

        print(f"Opening Processing Set, {ps_name}")
        ps_lazy_xdt = open_processing_set(str(ps_name))

        if is_s3:
            ps_copy_name = str(ps_name).split("/")[-1] + "_copy"
        else:
            ps_copy_name = str(ps_name) + "_copy"

        ps_lazy_xdt.to_zarr(ps_copy_name)  # Test writing to disk.

        # sel_parms = {key: {} for key in ps_lazy_xdt.keys()}
        ps_xdt = load_processing_set(str(ps_copy_name))

        if os.path.isdir(ps_copy_name):
            os.system("rm -rf " + str(ps_copy_name))  # Remove ps_xdt copy folder.

        base_check_ps_accessor(ps_lazy_xdt, ps_xdt)

        base_check_ms_accessor(ps_xdt)

        base_check_time_secondary_datasets(ps_xdt)

        sum = 0.0
        sum_lazy = 0.0
        for ms_xds_name in ps_xdt.keys():
            ms_xds = ps_xdt[ms_xds_name]
            if "VISIBILITY" in ms_xds:
                data_name = "VISIBILITY"
            else:
                data_name = "SPECTRUM"
            sum = sum + np.nansum(np.abs(ms_xds[data_name] * ms_xds.WEIGHT))
            sum_lazy = sum_lazy + np.nansum(
                np.abs(
                    ps_lazy_xdt[ms_xds_name][data_name]
                    * ps_lazy_xdt[ms_xds_name].WEIGHT
                )
            )

            base_check_time_centroid(ms_xds)

        print("sum", sum, sum_lazy)
        assert (
            sum == sum_lazy
        ), "open_processing_set and load_processing_set VISIBILITY and WEIGHT values differ."
        assert sum == pytest.approx(
            expected_sum_value, rel=relative_tolerance
        ), "VISIBILITY and WEIGHT values have changed."

        if do_schema_check:
            # print("*******************")
            # print(ps_xdt.xr_ps.summary())
            # #print(ps_xdt["ALMA_uid___A002_X1003af4_X75a3.split.avg_00"].field_and_source_base_xds)
            # print("*******************")

            start_check = time.time()
            issues = check_datatree(ps_xdt)

            print("***** Number of issues found:", len(issues))
            assert len(issues) == 0, "Schema check failed, issues found: " + str(issues)
            print(
                f"Time to check datasets (all MSv4s) against schema: {time.time() - start_check}"
            )

        check_expected_datasets_presence(ps_xdt, expected_secondary_xds)

        ps_list.append(ps_xdt)

    print("Time taken in test:", time.time() - start)
    return ps_list


# def test_s3(tmp_path):
#     # Similar to 'test_preconverted_alma' if this test fails on its own that
#     # probably is because the schema, the converter or the schema cheker have
#     # changed since the dataset was uploaded.
#     expected_subtables = {
#         "antenna"
#     }  # TODO: add "weather" once #365 is fixed and dataset re-converted
#     base_test(
#         "s3://viper-test-data/Antennae_North.cal.lsrk.split.py39.v7.vis.zarr",
#         tmp_path,
#         190.0405216217041,
#         is_s3=True,
#         partition_schemes=[[]],
#         expected_secondary_xds=expected_subtables,
#     )


def test_alma(tmp_path):
    expected_subtables = {"antenna", "weather"}

    print("the temp path is", tmp_path)
    base_test(
        "Antennae_North.cal.lsrk.split.ms",
        tmp_path,
        190.0405216217041,
        expected_secondary_xds=expected_subtables,
    )


# def test_preconverted_alma(tmp_path):
#     # If this test has failed on its own it most probably means the schema has changed.
#     # Create a fresh version using "Antennae_North.cal.lsrk.split.ms" and reconvert it using generate_zarr.py (in dropbox folder).
#     # Zip this folder and add it to the dropbox folder and update the file.download.json file.
#     # If you not sure how to do any of this contact jsteeb@nrao.edu
#     expected_subtables = {
#         "antenna"
#     }  # TODO: add "weather" once #365 is fixed and dataset re-converted
#     base_test(
#         "Antennae_North.cal.lsrk.split.py39.vis.zarr",
#         tmp_path,
#         190.0405216217041,
#         preconverted=True,
#         partition_schemes=[[]],
#         do_schema_check=False,
#         expected_secondary_xds=expected_subtables,
#     )


@pytest.mark.skipif(
    os.getenv("SKIP_TESTS_CASATOOLS") == "1",
    reason="Skip tests that require casatasks. getcolnp not available in casatools.",
)
def test_ska_low(tmp_path):
    expected_subtables = {"antenna", "phased_array"}
    base_test(
        "ska_low_sim_18s.ms",
        tmp_path,
        119802044416.0,
        expected_secondary_xds=expected_subtables,
        parallel_mode="time",
    )


@pytest.mark.skipif(
    os.getenv("SKIP_TESTS_CASATOOLS") == "1",
    reason=" Skip tests that require casatasks. getcolnp not available in casatools.",
)
def test_ska_mid(tmp_path):
    expected_subtables = {"antenna"}
    base_test(
        "AA2-Mid-sim_00000.ms",
        tmp_path,
        551412.3125,
        expected_secondary_xds=expected_subtables,
        parallel_mode="time",
    )


def test_lofar(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "small_lofar.ms",
        tmp_path,
        10345086189568.0,
        expected_secondary_xds=expected_subtables,
    )


def test_meerkat(tmp_path):
    expected_subtables = {"antenna"}
    base_test(
        "small_meerkat.ms",
        tmp_path,
        333866268.0,
        expected_secondary_xds=expected_subtables,
    )


def test_global_vlbi(tmp_path):
    expected_subtables = {"antenna", "gain_curve", "system_calibration"}
    base_test(
        "global_vlbi_gg084b_reduced.ms",
        tmp_path,
        161588975616.0,
        expected_secondary_xds=expected_subtables,
    )


def test_vlba(tmp_path):
    expected_subtables = {
        "antenna",
        "gain_curve",
        "system_calibration",
        "phase_calibration",
        "weather_xds",
    }
    base_test(
        "VLBA_TL016B_split.ms",
        tmp_path,
        94965412864.0,
        expected_secondary_xds=expected_subtables,
    )


def test_ngeht(tmp_path):
    expected_subtables = {"antenna"}
    base_test(
        "ngEHT_E17A10.0.bin0000.source0000_split.ms",
        tmp_path,
        64306946048.0,
        expected_secondary_xds=expected_subtables,
    )


def test_ephemeris(tmp_path):
    expected_subtables = {"antenna", "weather"}
    base_test(
        "venus_ephem_test.ms",
        tmp_path,
        81741343621120.0,
        expected_secondary_xds=expected_subtables,
    )


partition_schemes_sd = [[], ["FIELD_ID"], ["FIELD_ID", "ANTENNA1"]]


def test_single_dish(tmp_path):
    expected_subtables = {"antenna", "pointing", "system_calibration", "weather"}
    base_test(
        "sdimaging.ms",
        tmp_path,
        5487446.5,
        partition_schemes=partition_schemes_sd,
        expected_secondary_xds=expected_subtables,
    )


def test_alma_ephemeris_mosaic(tmp_path):
    expected_subtables = {"antenna", "system_calibration", "weather"}
    ps_list = base_test(
        "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms",
        tmp_path,
        8.11051993222426e17,
        expected_secondary_xds=expected_subtables,
    )
    # Here we test if the field_and_source_xds structure is correct.
    check_source_and_field_xds(
        ps_list[0], "ALMA_uid___A002_X1003af4_X75a3.split.avg_17", 1.09056423
    )
    check_source_and_field_xds(
        ps_list[1], "ALMA_uid___A002_X1003af4_X75a3.split.avg_81", 0.04194478
    )

    # Test PS sel
    check_ps_query(ps_list[0])
    check_ps_query(ps_list[1])


def check_ps_query(ps_xdt):
    ps_xdt.xr_ps.query(
        query="start_frequency > 2.46e11",
        field_coords="Ephemeris",
        field_name=["Sun_10_10", "Sun_10_11"],
    ).xr_ps.summary()
    min_freq = min(ps_xdt.xr_ps.summary()["start_frequency"])
    ps_xdt.xr_ps.query(start_frequency=min_freq).xr_ps.summary()
    ps_xdt.xr_ps.query(
        name="ALMA_uid___A002_X1003af4_X75a3.split.avg_01", string_exact_match=True
    ).xr_ps.summary()
    ps_xdt.xr_ps.query(field_name="Sun_10", string_exact_match=False).xr_ps.summary()
    ps_xdt.xr_ps.query(
        name="ALMA_uid___A002_X1003af4_X75a3.split.avg", string_exact_match=False
    ).xr_ps.summary()


def check_source_and_field_xds(ps_xdt, msv4_name, expected_NP_sum):
    field_and_source_xds = ps_xdt[msv4_name].xr_ms.get_field_and_source_xds()

    field_and_source_data_variable_names = [
        "FIELD_PHASE_CENTER_DIRECTION",
        "FIELD_PHASE_CENTER_DISTANCE",
        "HELIOCENTRIC_RADIAL_VELOCITY",
        "LINE_REST_FREQUENCY",
        "LINE_SYSTEMIC_VELOCITY",
        "NORTH_POLE_ANGULAR_DISTANCE",
        "NORTH_POLE_POSITION_ANGLE",
        "OBSERVER_POSITION",
        "OBSERVER_PHASE_ANGLE",
        "SOURCE_DIRECTION",
        "SOURCE_DISTANCE",
        "SOURCE_RADIAL_VELOCITY",
        "SUB_OBSERVER_DIRECTION",
    ]
    assert are_all_variables_in_dataset(
        field_and_source_xds, field_and_source_data_variable_names
    ), "field_and_source_xds is missing data variables."

    assert np.sum(field_and_source_xds.NORTH_POLE_ANGULAR_DISTANCE) == pytest.approx(
        expected_NP_sum, rel=relative_tolerance
    ), "The sum of the NORTH_POLE_ANGULAR_DISTANCE has changed."


def are_all_variables_in_dataset(dataset, variable_list):
    return all(var in dataset.data_vars for var in variable_list)


def test_vlass(tmp_path):
    # Don't do partition_scheme ['FIELD_ID'], will try and create >800 partitions.
    expected_subtables = {"antenna", "pointing", "weather"}
    base_test(
        "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms",
        tmp_path,
        173858574208.0,
        partition_schemes=[[]],
        expected_secondary_xds=expected_subtables,
    )


def test_sd_A002_X1015532_X1926f(tmp_path):
    expected_subtables = {"antenna", "pointing", "system_calibration_xds", "weather"}
    base_test(
        "uid___A002_X1015532_X1926f.small.ms",
        tmp_path,
        5.964230735563984e21,
        partition_schemes=partition_schemes_sd,
        expected_secondary_xds=expected_subtables,
    )


def test_sd_A002_Xae00c5_X2e6b(tmp_path):
    expected_subtables = {"antenna", "pointing", "system_calibration_xds", "weather"}
    base_test(
        "uid___A002_Xae00c5_X2e6b.small.ms",
        tmp_path,
        2451894476.0,
        partition_schemes=partition_schemes_sd,
        expected_secondary_xds=expected_subtables,
    )


def test_sd_A002_Xced5df_Xf9d9(tmp_path):
    expected_subtables = {"antenna", "pointing", "system_calibration_xds", "weather"}
    base_test(
        "uid___A002_Xced5df_Xf9d9.small.ms",
        tmp_path,
        9.892002713707104e21,
        partition_schemes=partition_schemes_sd,
        expected_secondary_xds=expected_subtables,
    )


def test_sd_A002_Xe3a5fd_Xe38e(tmp_path):
    expected_subtables = {"antenna", "pointing", "system_calibration_xds", "weather"}
    base_test(
        "uid___A002_Xe3a5fd_Xe38e.small.ms",
        tmp_path,
        246949088254189.5,
        partition_schemes=partition_schemes_sd,
        expected_secondary_xds=expected_subtables,
    )


def test_VLA(tmp_path):
    expected_subtables = {"antenna", "weather"}
    base_test(
        "SNR_G55_10s.split.ms",
        tmp_path,
        195110762496.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59749_bp_8beams_pattern(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59749_bp_8beams_pattern.ms",
        tmp_path,
        5652688384.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59750_altaz_2settings(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59750_altaz_2settings.ms",
        tmp_path,
        1878356864.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59754_altaz_2weights_0(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59754_altaz_2weights_0.ms",
        tmp_path,
        1504652800.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59754_altaz_2weights_15(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59754_altaz_2weights_15.ms",
        tmp_path,
        1334662656.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59755_eq_interleave_0(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59755_eq_interleave_0.ms",
        tmp_path,
        3052425984.0,
        expected_secondary_xds=expected_subtables,
    )


def test_askap_59755_eq_interleave_15(tmp_path):
    expected_subtables = {"antenna", "pointing"}
    base_test(
        "59755_eq_interleave_15.ms",
        tmp_path,
        2949046016.0,
        expected_secondary_xds=expected_subtables,
    )


def test_gmrt(tmp_path):
    expected_subtables = {"antenna"}
    base_test(
        "gmrt.ms", tmp_path, 541752852480.0, expected_secondary_xds=expected_subtables
    )


if __name__ == "__main__":
    a = 42
    from pathlib import Path

    # test_askap_59749_bp_8beams_pattern(tmp_path=Path("."))
    # test_askap_59750_altaz_2settings(tmp_path=Path("."))
    # test_askap_59754_altaz_2weights_0(tmp_path=Path("."))
    # test_askap_59754_altaz_2weights_15(tmp_path=Path("."))
    # test_askap_59755_eq_interleave_0(tmp_path=Path("."))
    # test_askap_59755_eq_interleave_15(tmp_path=Path("."))

    # test_sd_A002_X1015532_X1926f(tmp_path=Path("."))
    # test_sd_A002_Xae00c5_X2e6b(tmp_path=Path("."))
    # test_sd_A002_Xced5df_Xf9d9(tmp_path=Path("."))
    test_sd_A002_Xe3a5fd_Xe38e(tmp_path=Path("."))
    # test_s3(tmp_path=Path("."))
    # test_vlass(tmp_path=Path("."))
    # test_alma(tmp_path=Path("."))
    # #test_preconverted_alma(tmp_path=Path("."))
    # test_ska_mid(tmp_path=Path("."))
    # test_ska_low(tmp_path=Path("."))
    # test_lofar(tmp_path=Path("."))
    # test_meerkat(tmp_path=Path("."))
    # test_global_vlbi(tmp_path=Path("."))
    # test_vlba(tmp_path=Path("."))
    # test_ngeht(tmp_path=Path("."))
    # test_ephemeris(tmp_path=Path("."))
    # test_single_dish(tmp_path=Path("."))
    # test_alma_ephemeris_mosaic(tmp_path=Path("."))
    # test_VLA(tmp_path=Path("."))

# All test preformed on MAC with M3 and 16 GB Ram.
# pytest --durations=0 .
# Timing. Using Data Tree. Parallel False + get_col
# 78.41s call     tests/stakeholder/test_measure_set_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e
# 36.39s call     tests/stakeholder/test_measure_set_stakeholder.py::test_sd_A002_X1015532_X1926f

# 36.26s call     tests/stakeholder/test_measure_set_stakeholder.py::test_alma_ephemeris_mosaic
# 34.75s call     tests/stakeholder/test_measure_set_stakeholder.py::test_sd_A002_Xced5df_Xf9d9
# 22.63s call     tests/stakeholder/test_measure_set_stakeholder.py::test_vlass
# 12.57s call     tests/stakeholder/test_measure_set_stakeholder.py::test_sd_A002_Xae00c5_X2e6b
# 6.47s call     tests/stakeholder/test_measure_set_stakeholder.py::test_ephemeris
# 6.36s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59754_altaz_2weights_0
# 6.12s call     tests/stakeholder/test_measure_set_stakeholder.py::test_global_vlbi
# 6.07s call     tests/stakeholder/test_measure_set_stakeholder.py::test_alma
# 6.04s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59754_altaz_2weights_15
# 5.81s call     tests/stakeholder/test_measure_set_stakeholder.py::test_single_dish
# 5.43s call     tests/stakeholder/test_measure_set_stakeholder.py::test_VLA
# 5.35s call     tests/stakeholder/test_measure_set_stakeholder.py::test_vlba
# 4.95s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59749_bp_8beams_pattern
# 4.28s call     tests/stakeholder/test_measure_set_stakeholder.py::test_meerkat
# 4.28s call     tests/stakeholder/test_measure_set_stakeholder.py::test_ngeht
# 4.23s call     tests/stakeholder/test_measure_set_stakeholder.py::test_ska_mid
# 3.65s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59750_altaz_2settings
# 3.43s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59755_eq_interleave_15
# 3.41s call     tests/stakeholder/test_measure_set_stakeholder.py::test_askap_59755_eq_interleave_0
# 3.03s call     tests/stakeholder/test_measure_set_stakeholder.py::test_lofar


# All test preformed on MAC with M3 and 16 GB Ram.
# pytest --durations=0 .
# Timing. Parallel False + get_col
# 33.33s call     tests/stakeholder/test_cor_stakeholder.py::test_alma_ephemeris_mosaic
# 27.98s call     tests/stakeholder/test_cor_stakeholder.py::test_vlass
# 25.89s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e
# 25.29s call     tests/stakeholder/test_cor_stakeholder.py::test_s3
# 22.36s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xced5df_Xf9d9
# 14.06s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_X1015532_X1926f
# 13.03s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xae00c5_X2e6b
# 4.42s call     tests/stakeholder/test_cor_stakeholder.py::test_ephemeris
# 3.85s call     tests/stakeholder/test_cor_stakeholder.py::test_single_dish
# 3.62s call     tests/stakeholder/test_cor_stakeholder.py::test_alma
# 3.17s call     tests/stakeholder/test_cor_stakeholder.py::test_global_vlbi
# 3.01s call     tests/stakeholder/test_cor_stakeholder.py::test_vlba
# 2.97s call     tests/stakeholder/test_cor_stakeholder.py::test_meerkat
# 2.75s call     tests/stakeholder/test_cor_stakeholder.py::test_ska_mid
# 2.68s call     tests/stakeholder/test_cor_stakeholder.py::test_ngeht
# 2.33s call     tests/stakeholder/test_cor_stakeholder.py::test_lofar

# Timing. Parallel True + get_col
# 25.49s call     tests/stakeholder/test_cor_stakeholder.py::test_s3
# 19.05s call     tests/stakeholder/test_cor_stakeholder.py::test_alma_ephemeris_mosaic
# 17.01s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e
# 12.86s call     tests/stakeholder/test_cor_stakeholder.py::test_vlass
# 10.26s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xced5df_Xf9d9
# 7.68s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_X1015532_X1926f
# 7.00s call     tests/stakeholder/test_cor_stakeholder.py::test_sd_A002_Xae00c5_X2e6b
# 4.61s call     tests/stakeholder/test_cor_stakeholder.py::test_alma
# 4.58s call     tests/stakeholder/test_cor_stakeholder.py::test_global_vlbi
# 3.40s call     tests/stakeholder/test_cor_stakeholder.py::test_ephemeris
# 3.20s call     tests/stakeholder/test_cor_stakeholder.py::test_ngeht
# 3.01s call     tests/stakeholder/test_cor_stakeholder.py::test_single_dish
# 2.98s call     tests/stakeholder/test_cor_stakeholder.py::test_vlba
# 2.89s call     tests/stakeholder/test_cor_stakeholder.py::test_ska_mid
# 2.40s call     tests/stakeholder/test_cor_stakeholder.py::test_meerkat
# 2.39s call     tests/stakeholder/test_cor_stakeholder.py::test_lofar

# Timing. Parallel False + iter col
# 87.50s call     tests/stakeholder/test_cor_stakeholder.py::test_vlass
# 42.37s call     tests/stakeholder/test_cor_stakeholder.py::test_alma_ephemeris_mosaic
# 23.41s call     tests/stakeholder/test_cor_stakeholder.py::test_s3
# 14.96s call     tests/stakeholder/test_cor_stakeholder.py::test_ngeht
# 12.69s call     tests/stakeholder/test_cor_stakeholder.py::test_single_dish
# 6.80s call     tests/stakeholder/test_cor_stakeholder.py::test_vlba
# 4.05s call     tests/stakeholder/test_cor_stakeholder.py::test_ephemeris
# 3.66s call     tests/stakeholder/test_cor_stakeholder.py::test_alma
# 3.41s call     tests/stakeholder/test_cor_stakeholder.py::test_meerkat
# 3.00s call     tests/stakeholder/test_cor_stakeholder.py::test_global_vlbi
# 2.99s call     tests/stakeholder/test_cor_stakeholder.py::test_ska_mid
# 2.62s call     tests/stakeholder/test_cor_stakeholder.py::test_lofar


# How data was created:
# ALMA Example
"""
ALMA_uid___A002_X1003af4_X75a3.split.avg.ms: An ephemeris mosaic observation of the sun.

ALMA archive file downloaded: https://almascience.nrao.edu/dataPortal/2022.A.00001.S_uid___A002_X1003af4_X75a3.asdm.sdm.tar

- Project: 2022.A.00001.S
- Member ous id (MOUS): uid://A001/X3571/X130
- Group ous id (GOUS): uid://A001/X3571/X131

CASA commands used to create the dataset:
```python
importasdm(asdm='uid___A002_X1003af4_X75a3.asdm.sdm',vis='uid___A002_X1003af4_X75a3.ms',asis='Ephemeris Antenna Station Receiver Source CalAtmosphere CalWVR',bdfflags=True,with_pointing_correction=True,convert_ephem2geo=True)

mstransform(vis='ALMA_uid___A002_X1003af4_X75a3.split.ms',outputvis='ALMA_uid___A002_X1003af4_X75a3.split.avg.ms',createmms=False,timeaverage=True,timebin='2s',timespan='scan',scan='6~8', spw='3:60~66,4:60~66,5:60~66', reindex=True,datacolumn='all')

import numpy as np

for subtable in ['FLAG_CMD', 'POINTING', 'CALDEVICE', 'ASDM_CALATMOSPHERE']:
    tb.open('ALMA_uid___A002_X1003af4_X75a3.split.avg.ms::'+subtable,nomodify=False)
    tb.removerows(np.arange(tb.nrows()))
    tb.flush()
    tb.done()
```
"""

"""
SNR_G55_10s.ms : VLA

ALMA archive file downloaded: http://casa.nrao.edu/Data/EVLA/SNRG55/SNR_G55_10s.tar.gz



```python

mstransform(vis='SNR_G55_10s.ms',outputvis='SNR_G55_10s.split.ms',createmms=False,spw='1:30~34,2:30~34',antenna='ea01,ea05,ea07,ea09',scan='15~50',reindex=True,datacolumn='all')


```
"""
