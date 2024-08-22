from xradio.vis import (
    read_processing_set,
    load_processing_set,
    convert_msv2_to_processing_set,
    VisibilityXds,
)
from xradio.schema.check import check_dataset
from graphviper.utils.data import download
import numpy as np
import pytest
import os
import importlib.resources
from graphviper.utils.logger import setup_logger
import time

# relative_tolerance = 10 ** (-12)
relative_tolerance = 10 ** (-6)


def download_and_convert_msv2_to_processing_set(msv2_name, partition_scheme):
    # We can remove this once there is a new release of casacore
    if os.environ["USER"] == "runner":
        casa_data_dir = (importlib.resources.files("casadata") / "__data__").as_posix()
        rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
        rc_file.write("\nmeasures.directory: " + casa_data_dir)
        rc_file.close()

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

    download(file=msv2_name)
    ps_name = msv2_name[:-3] + ".vis.zarr"
    convert_msv2_to_processing_set(
        in_file=msv2_name,
        out_file=ps_name,
        partition_scheme=partition_scheme,
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        overwrite=True,
        parallel=False,
    )
    return ps_name


def base_test(
    file_name,
    expected_sum_value,
    is_s3=False,
    partition_schemes=[[], ["FIELD_ID"]],
    preconverted=False,
):
    start = time.time()
    from graphviper.dask.client import local_client

    # Strange bug when running test in paralell (the unrelated image tests fail).
    # viper_client = local_client(cores=4, memory_limit="4GB")
    # viper_client

    for partition_scheme in partition_schemes:
        if is_s3:
            ps_name = file_name
        elif preconverted:
            download(file=file_name)
            ps_name = file_name
        else:
            ps_name = download_and_convert_msv2_to_processing_set(
                file_name, partition_scheme
            )

        print("ps_name", ps_name)
        ps_lazy = read_processing_set(ps_name)

        sel_parms = {key: {} for key in ps_lazy.keys()}
        ps = load_processing_set(ps_name, sel_parms=sel_parms)

        ps_lazy_df = ps_lazy.summary()
        ps_df = ps.summary()

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

        os.system("rm -rf " + ps_name)  # Remove vis.zarr folder.
        os.system("rm -rf " + file_name)  # Remove downloaded MSv2 folder.

        print("sum", sum, sum_lazy)
        assert (
            sum == sum_lazy
        ), "read_processing_set and load_processing_set VISIBILITY and WEIGHT values differ."
        assert sum == pytest.approx(
            expected_sum_value, rel=relative_tolerance
        ), "VISIBILITY and WEIGHT values have changed."

        msv4 = ps[ms_xds_name]
        start_check = time.time()
        check_dataset(ps[ms_xds_name], VisibilityXds).expect()
        print(f"Time to check dataset: {time.time() - start_check}")

    print("Time taken in test:", time.time() - start)


def test_s3():
    base_test(
        "s3://viper-test-data/Antennae_North.cal.lsrk.split.v5.vis.zarr",
        190.0405216217041,
        is_s3=True,
        partition_schemes=[[]],
    )


def test_alma():
    base_test("Antennae_North.cal.lsrk.split.ms", 190.0405216217041)


def DISABLED223test_preconverted_alma():
    # If this test has failed on its own it most probably means the schema has changed.
    # Create a fresh version using "Antennae_North.cal.lsrk.split.ms" and reconvert it using generate_zarr.py (in dropbox folder).
    # Zip this folder and add it to the dropbox folder and update the file.download.json file.
    # If you not sure how to do any of this contact jsteeb@nrao.edu
    base_test(
        "Antennae_North.cal.lsrk.split.vis.zarr",
        190.0405216217041,
        preconverted=True,
        partition_schemes=[[]],
    )


def test_ska_mid():
    base_test("AA2-Mid-sim_00000.ms", 551412.3125)


def test_lofar():
    base_test("small_lofar.ms", 10345086189568.0)


def test_meerkat():
    base_test("small_meerkat.ms", 333866268.0)


def test_global_vlbi():
    base_test("global_vlbi_gg084b_reduced.ms", 161588975616.0)


def test_vlba():
    base_test("VLBA_TL016B_split.ms", 94965412864.0)


def test_ngeht():
    base_test("ngEHT_E17A10.0.bin0000.source0000_split.ms", 64306946048.0)


def test_ephemeris():
    base_test("venus_ephem_test.ms", 81741343621120.0)


def test_single_dish():
    base_test("sdimaging.ms", 5487446.5)


def test_alma_ephemeris_mosaic():
    base_test("ALMA_uid___A002_X1003af4_X75a3.split.avg.ms", 8.11051993222426e17)


def test_vlass():
    # Don't do partition_scheme ['FIELD_ID'], will try and create >800 partitions.
    base_test(
        "VLASS3.2.sb45755730.eb46170641.60480.16266136574.split.v6.ms",
        173858574208.0,
        partition_schemes=[[]],
    )


def test_sd_A002_X1015532_X1926f():
    base_test("uid___A002_X1015532_X1926f.small.ms", 5.964230735563984e21)


def test_sd_A002_Xae00c5_X2e6b():
    base_test("uid___A002_Xae00c5_X2e6b.small.ms", 2451894476.0)


def test_sd_A002_Xced5df_Xf9d9():
    base_test("uid___A002_Xced5df_Xf9d9.small.ms", 9.892002713707104e21)


def test_sd_A002_Xe3a5fd_Xe38e():
    base_test("uid___A002_Xe3a5fd_Xe38e.small.ms", 246949088254189.5)


def test_VLA():
    base_test("SNR_G55_10s.split.ms", 195110762496.0)


if __name__ == "__main__":
    a = 42
    # test_sd_A002_X1015532_X1926f()
    # test_sd_A002_Xae00c5_X2e6b()
    # test_sd_A002_Xced5df_Xf9d9()
    # test_sd_A002_Xe3a5fd_Xe38e()
    # test_s3()
    # test_vlass()
    test_alma()
    # test_preconverted_alma()
    # test_ska_mid()
    # test_lofar()
    # test_meerkat()
    # test_global_vlbi()
    # test_vlba()
    # test_ngeht()
    # test_ephemeris()
    # test_single_dish()
    # test_alma_ephemeris_mosaic()
    # test_VLA()

# All test preformed on MAC with M3 and 16 GB Ram.
# pytest --durations=0 .
# Timing. Parallel False + get_col
# 33.33s call     tests/stakeholder/test_vis_stakeholder.py::test_alma_ephemeris_mosaic
# 27.98s call     tests/stakeholder/test_vis_stakeholder.py::test_vlass
# 25.89s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e
# 25.29s call     tests/stakeholder/test_vis_stakeholder.py::test_s3
# 22.36s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xced5df_Xf9d9
# 14.06s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_X1015532_X1926f
# 13.03s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xae00c5_X2e6b
# 4.42s call     tests/stakeholder/test_vis_stakeholder.py::test_ephemeris
# 3.85s call     tests/stakeholder/test_vis_stakeholder.py::test_single_dish
# 3.62s call     tests/stakeholder/test_vis_stakeholder.py::test_alma
# 3.17s call     tests/stakeholder/test_vis_stakeholder.py::test_global_vlbi
# 3.01s call     tests/stakeholder/test_vis_stakeholder.py::test_vlba
# 2.97s call     tests/stakeholder/test_vis_stakeholder.py::test_meerkat
# 2.75s call     tests/stakeholder/test_vis_stakeholder.py::test_ska_mid
# 2.68s call     tests/stakeholder/test_vis_stakeholder.py::test_ngeht
# 2.33s call     tests/stakeholder/test_vis_stakeholder.py::test_lofar

# Timing. Parallel True + get_col
# 25.49s call     tests/stakeholder/test_vis_stakeholder.py::test_s3
# 19.05s call     tests/stakeholder/test_vis_stakeholder.py::test_alma_ephemeris_mosaic
# 17.01s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xe3a5fd_Xe38e
# 12.86s call     tests/stakeholder/test_vis_stakeholder.py::test_vlass
# 10.26s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xced5df_Xf9d9
# 7.68s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_X1015532_X1926f
# 7.00s call     tests/stakeholder/test_vis_stakeholder.py::test_sd_A002_Xae00c5_X2e6b
# 4.61s call     tests/stakeholder/test_vis_stakeholder.py::test_alma
# 4.58s call     tests/stakeholder/test_vis_stakeholder.py::test_global_vlbi
# 3.40s call     tests/stakeholder/test_vis_stakeholder.py::test_ephemeris
# 3.20s call     tests/stakeholder/test_vis_stakeholder.py::test_ngeht
# 3.01s call     tests/stakeholder/test_vis_stakeholder.py::test_single_dish
# 2.98s call     tests/stakeholder/test_vis_stakeholder.py::test_vlba
# 2.89s call     tests/stakeholder/test_vis_stakeholder.py::test_ska_mid
# 2.40s call     tests/stakeholder/test_vis_stakeholder.py::test_meerkat
# 2.39s call     tests/stakeholder/test_vis_stakeholder.py::test_lofar

# Timing. Parallel False + iter col
# 87.50s call     tests/stakeholder/test_vis_stakeholder.py::test_vlass
# 42.37s call     tests/stakeholder/test_vis_stakeholder.py::test_alma_ephemeris_mosaic
# 23.41s call     tests/stakeholder/test_vis_stakeholder.py::test_s3
# 14.96s call     tests/stakeholder/test_vis_stakeholder.py::test_ngeht
# 12.69s call     tests/stakeholder/test_vis_stakeholder.py::test_single_dish
# 6.80s call     tests/stakeholder/test_vis_stakeholder.py::test_vlba
# 4.05s call     tests/stakeholder/test_vis_stakeholder.py::test_ephemeris
# 3.66s call     tests/stakeholder/test_vis_stakeholder.py::test_alma
# 3.41s call     tests/stakeholder/test_vis_stakeholder.py::test_meerkat
# 3.00s call     tests/stakeholder/test_vis_stakeholder.py::test_global_vlbi
# 2.99s call     tests/stakeholder/test_vis_stakeholder.py::test_ska_mid
# 2.62s call     tests/stakeholder/test_vis_stakeholder.py::test_lofar


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
