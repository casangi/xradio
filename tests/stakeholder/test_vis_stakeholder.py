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

relative_tolerance = 10 ** (-12)


def download_and_convert_msv2_to_processing_set(msv2_name):
    # We can remove this once there is a new release of casacore
    if os.environ["USER"] == "runner":
        casa_data_dir = (importlib.resources.files("casadata") / "__data__").as_posix()
        rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
        rc_file.write("\nmeasures.directory: " + casa_data_dir)
        rc_file.close()

    download(file=msv2_name)
    ps_name = msv2_name[:-3] + ".vis.zarr"
    convert_msv2_to_processing_set(
        in_file=msv2_name,
        out_file=ps_name,
        partition_scheme="ddi_intent_field",
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        overwrite=True,
    )
    return ps_name


def base_test(file_name, expected_sum_value, is_s3=False):

    if is_s3:
        ps_name = file_name
    else:
        ps_name = download_and_convert_msv2_to_processing_set(file_name)

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

    if not is_s3:
        os.system("rm -rf " + file_name)
        os.system("rm -rf " + ps_name)

    assert (
        sum == sum_lazy
    ), "read_processing_set and load_processing_set VISIBILITY and WEIGHT values differ."
    assert sum == pytest.approx(
        expected_sum_value, rel=relative_tolerance
    ), "VISIBILITY and WEIGHT values have changed."

    for xds_name in ps.keys():
        issues = check_dataset(ps[xds_name], VisibilityXds)
        if not issues:
            print(f"{xds_name}: okay\n")
        else:
            print(f"{xds_name}: {issues}\n")


def test_s3():
    base_test(
        "s3://viper-test-data/Antennae_North.cal.lsrk.split.v3.vis.zarr",
        190.0405216217041,
        is_s3=True,
    )


def test_alma():
    base_test("Antennae_North.cal.lsrk.split.ms", 190.0405216217041)


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


# test_s3()
# test_alma()
# test_ska_mid()
# test_lofar()
# test_meerkat()
# test_global_vlbi()
# test_vlba()
# test_ngeht()
# test_ephemeris()
# test_single_dish()
