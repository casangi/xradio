import importlib.resources
import numpy as np
import os
import pathlib
import pytest
import time

from toolviper.utils.data import download
from toolviper.utils.logger import setup_logger
from xradio.correlated_data import (
    open_processing_set,
    load_processing_set,
    convert_msv2_to_processing_set,
    VisibilityXds,
    SpectrumXds,
)
from xradio.schema.check import check_dataset

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


def download_and_convert_msv2_to_processing_set(msv2_name, folder, partition_scheme):

    # We can remove this once there is a new release of casacore
    # if os.environ["USER"] == "runner":
    #     casa_data_dir = (importlib.resources.files("casadata") / "__data__").as_posix()
    #     rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
    #     rc_file.write("\nmeasures.directory: " + casa_data_dir)
    #     rc_file.close()

    _logger_name = "xradio"
    if os.getenv("VIPER_LOGGER_NAME") != _logger_name:
        os.environ["VIPER_LOGGER_NAME"] = _logger_name
        setup_logger(
            logger_name="xradio",
            log_to_term=True,
            log_to_file=False,  # True
            log_file="xradio-logfile",
            # log_level="DEBUG",
            log_level="INFO",
        )

    download(file=msv2_name, folder=folder)
    ps_name = folder / (msv2_name[:-3] + ".ps")
    if os.path.isdir(ps_name):
        os.system("rm -rf " + str(ps_name))  # Remove ps folder.
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
        parallel=False,
    )
    return ps_name


def base_test(
    file_name: str,
    folder: pathlib.Path,
    expected_sum_value: float,
    is_s3: bool = False,
    partition_schemes: list = [[], ["FIELD_ID"]],
    preconverted: bool = False,
    do_schema_check: bool = True,
):
    start = time.time()
    from toolviper.dask.client import local_client

    # Strange bug when running test in paralell (the unrelated image tests fail).
    # viper_client = local_client(cores=4, memory_limit="4GB")
    # viper_client

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
                file_name, folder, partition_scheme
            )

        print(f"Opening Processing Set, {ps_name}")
        ps_lazy = open_processing_set(str(ps_name))

        ps_copy_name = str(ps_name) + "_copy"
        ps_lazy.to_store(ps_copy_name)  # Test writing yo disk.

        sel_parms = {key: {} for key in ps_lazy.keys()}
        ps = load_processing_set(str(ps_copy_name), sel_parms=sel_parms)

        if os.path.isdir(ps_copy_name):
            os.system("rm -rf " + str(ps_copy_name))  # Remove ps copy folder.

        ps_lazy_df = ps_lazy.summary()
        ps_df = ps.summary()
        
        ps.get_combined_field_and_source_xds()
        ps.get_combined_antenna_xds()

        sum = 0.0
        sum_lazy = 0.0

        for ms_xds_name in ps.keys():
            if "VISIBILITY" in ps[ms_xds_name]:
                data_name = "VISIBILITY"
            else:
                data_name = "SPECTRUM"
            sum = sum + np.nansum(
                np.abs(ps[ms_xds_name][data_name] * ps[ms_xds_name].WEIGHT)
            )
            sum_lazy = sum_lazy + np.nansum(
                np.abs(ps_lazy[ms_xds_name][data_name] * ps_lazy[ms_xds_name].WEIGHT)
            )

        print("sum", sum, sum_lazy)
        assert (
            sum == sum_lazy
        ), "open_processing_set and load_processing_set VISIBILITY and WEIGHT values differ."
        assert sum == pytest.approx(
            expected_sum_value, rel=relative_tolerance
        ), "VISIBILITY and WEIGHT values have changed."

        if do_schema_check:
            start_check = time.time()
            for xds_name in ps.keys():
                if ps[xds_name].attrs["type"] == "visibility":
                    check_dataset(ps[xds_name], VisibilityXds).expect()
                elif ps[xds_name].attrs["type"] == "spectrum":
                    check_dataset(ps[xds_name], SpectrumXds).expect()
                else:
                    raise RuntimeError(
                        "Cannot find visibility or spectrum type data in MSv4 {xds_name}!"
                    )

            print(
                f"Time to check datasets (all MSv4s) against schema: {time.time() - start_check}"
            )

        ps_list.append(ps)

    print("Time taken in test:", time.time() - start)
    return ps_list


def test_s3(tmp_path):
    # Similar to 'test_preconverted_alma' if this test fails on its own that
    # probably is because the schema, the converter or the schema cheker have
    # changed since the dataset was uploaded.
    base_test(
        "s3://viper-test-data/Antennae_North.cal.lsrk.split.py39.v2.vis.zarr",
        tmp_path,
        190.0405216217041,
        is_s3=True,
        partition_schemes=[[]],
        do_schema_check=False,
    )


def test_alma(tmp_path):
    base_test("Antennae_North.cal.lsrk.split.ms", tmp_path, 190.0405216217041)


def test_preconverted_alma(tmp_path):
    # If this test has failed on its own it most probably means the schema has changed.
    # Create a fresh version using "Antennae_North.cal.lsrk.split.ms" and reconvert it using generate_zarr.py (in dropbox folder).
    # Zip this folder and add it to the dropbox folder and update the file.download.json file.
    # If you not sure how to do any of this contact jsteeb@nrao.edu
    base_test(
        "Antennae_North.cal.lsrk.split.py39.vis.zarr",
        tmp_path,
        190.0405216217041,
        preconverted=True,
        partition_schemes=[[]],
        do_schema_check=False,
    )


def test_ska_mid(tmp_path):
    base_test("AA2-Mid-sim_00000.ms", tmp_path, 551412.3125)


def test_lofar(tmp_path):
    base_test("small_lofar.ms", tmp_path, 10345086189568.0)


def test_meerkat(tmp_path):
    base_test("small_meerkat.ms", tmp_path, 333866268.0)


def test_global_vlbi(tmp_path):
    base_test("global_vlbi_gg084b_reduced.ms", tmp_path, 161588975616.0)


def test_vlba(tmp_path):
    base_test("VLBA_TL016B_split.ms", tmp_path, 94965412864.0)


def test_ngeht(tmp_path):
    base_test("ngEHT_E17A10.0.bin0000.source0000_split.ms", tmp_path, 64306946048.0)


def test_ephemeris(tmp_path):
    base_test("venus_ephem_test.ms", tmp_path, 81741343621120.0)


def test_single_dish(tmp_path):
    base_test("sdimaging.ms", tmp_path, 5487446.5)


def test_alma_ephemeris_mosaic(tmp_path):
    ps_list = base_test(
        "ALMA_uid___A002_X1003af4_X75a3.split.avg.ms", tmp_path, 8.11051993222426e17
    )
    # Here we test if the field_and_source_xds structure is correct.
    check_source_and_field_xds(
        ps_list[0], "ALMA_uid___A002_X1003af4_X75a3.split.avg_17", 127796.84837227
    )
    check_source_and_field_xds(
        ps_list[1], "ALMA_uid___A002_X1003af4_X75a3.split.avg_81", 4915.66000546
    )

    # Test PS sel
    check_ps_sel(ps_list[0])
    check_ps_sel(ps_list[1])


def check_ps_sel(ps):
    ps.sel(
        query="start_frequency > 2.46e11",
        field_coords="Ephemeris",
        field_name=["Sun_10_10", "Sun_10_11"],
    ).summary()
    min_freq = min(ps.summary()["start_frequency"])
    ps.sel(start_frequency=min_freq).summary()
    ps.sel(
        name="ALMA_uid___A002_X1003af4_X75a3.split.avg_01", string_exact_match=True
    ).summary()
    ps.sel(field_name="Sun_10", string_exact_match=False).summary()
    ps.sel(
        name="ALMA_uid___A002_X1003af4_X75a3.split.avg", string_exact_match=False
    ).summary()


def check_source_and_field_xds(ps, msv4_name, expected_NP_sum):
    field_and_source_xds = ps[msv4_name].VISIBILITY.attrs["field_and_source_xds"]
    field_and_source_data_variable_names = [
        "FIELD_PHASE_CENTER",
        "HELIOCENTRIC_RADIAL_VELOCITY",
        "LINE_REST_FREQUENCY",
        "LINE_SYSTEMIC_VELOCITY",
        "NORTH_POLE_ANGULAR_DISTANCE",
        "NORTH_POLE_POSITION_ANGLE",
        "OBSERVER_POSITION",
        "OBSERVER_PHASE_ANGLE",
        "SOURCE_LOCATION",
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
    base_test(
        "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms",
        tmp_path,
        173858574208.0,
        partition_schemes=[[]],
    )


def test_sd_A002_X1015532_X1926f(tmp_path):
    base_test("uid___A002_X1015532_X1926f.small.ms", tmp_path, 5.964230735563984e21)


def test_sd_A002_Xae00c5_X2e6b(tmp_path):
    base_test("uid___A002_Xae00c5_X2e6b.small.ms", tmp_path, 2451894476.0)


def test_sd_A002_Xced5df_Xf9d9(tmp_path):
    base_test("uid___A002_Xced5df_Xf9d9.small.ms", tmp_path, 9.892002713707104e21)


def test_sd_A002_Xe3a5fd_Xe38e(tmp_path):
    base_test("uid___A002_Xe3a5fd_Xe38e.small.ms", tmp_path, 246949088254189.5)


def test_VLA(tmp_path):
    base_test("SNR_G55_10s.split.ms", tmp_path, 195110762496.0)


def test_askap_59749_bp_8beams_pattern(tmp_path):
    base_test("59749_bp_8beams_pattern.ms", tmp_path, 5652688384.0)


def test_askap_59750_altaz_2settings(tmp_path):
    base_test("59750_altaz_2settings.ms", tmp_path, 1878356864.0)


def test_askap_59754_altaz_2weights_0(tmp_path):
    base_test("59754_altaz_2weights_0.ms", tmp_path, 1504652800.0)


def test_askap_59754_altaz_2weights_15(tmp_path):
    base_test("59754_altaz_2weights_15.ms", tmp_path, 1334662656.0)


def test_askap_59755_eq_interleave_0(tmp_path):
    base_test("59755_eq_interleave_0.ms", tmp_path, 3052425984.0)


def test_askap_59755_eq_interleave_15(tmp_path):
    base_test("59755_eq_interleave_15.ms", tmp_path, 2949046016.0)


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
    # test_sd_A002_Xe3a5fd_Xe38e(tmp_path=Path("."))
    # test_s3(tmp_path=Path("."))
    # test_vlass(tmp_path=Path("."))
    # test_alma(tmp_path=Path("."))
    # test_preconverted_alma(tmp_path=Path("."))
    test_ska_mid(tmp_path=Path("."))
    # test_lofar(tmp_path=Path("."))
    # test_meerkat(tmp_path=Path("."))
    # test_global_vlbi(tmp_path=Path("."))
    # test_vlba(tmp_path=Path("."))
    # test_ngeht(tmp_path=Path("."))
    # test_ephemeris(tmp_path=Path("."))
    # test_single_dish(tmp_path=Path("."))
    # test_alma_ephemeris_mosaic(tmp_path=Path("."))
    # test_VLA(tmp_path=Path("."))

    # FAILED test_cor_stakeholder.py::test_ephemeris - ValueError: Buffer has wrong number of dimensions (expected 1, got 2)
    # FAILED test_cor_stakeholder.py::test_alma_ephemeris_mosaic - ValueError: Buffer has wrong number of dimensions (expected 1, got 2)
    # FAILED test_cor_stakeholder.py::test_sd_A002_X1015532_X1926f - ValueError: Buffer has wrong number of dimensions (expected 1, got 2)
    # FAILED test_cor_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e - ValueError: Buffer has wrong number of dimensions (expected 1, got 2)

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
