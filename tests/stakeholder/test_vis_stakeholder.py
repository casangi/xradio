from xradio.vis import (
    read_processing_set,
    load_processing_set,
    convert_msv2_to_processing_set,
)
from graphviper.utils.data import download
import numpy as np
import pytest
import os
import pkg_resources

relative_tolerance = 10 ** (-12)


def base_test(msv2_name, expected_sum_value):
    if os.environ["USER"] == "runner":
        casa_data_dir = pkg_resources.resource_filename("casadata", "__data__")
        rc_file = open(os.path.expanduser("~/.casarc"), "a+")  # append mode
        rc_file.write("\nmeasures.directory: " + casa_data_dir)
        rc_file.close()

    download(file=msv2_name)
    ps_name = msv2_name[:-3] + ".vis.zarr"
    convert_msv2_to_processing_set(
        in_file=msv2_name,
        out_file=ps_name,
        partition_scheme="ddi_intent_field",
        overwrite=True,
    )

    ps_lazy = read_processing_set(ps_name)

    sel_parms = {key: {} for key in ps_lazy.keys()}
    ps = load_processing_set(ps_name, sel_parms=sel_parms)

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

    print(sum)

    os.system("rm -rf " + msv2_name)
    os.system("rm -rf " + ps_name)

    assert (
        sum == sum_lazy
    ), "read_processing_set and load_processing_set VISIBILITY and WEIGHT values differ."
    assert sum == pytest.approx(
        expected_sum_value, rel=relative_tolerance
    ), "VISIBILITY and WEIGHT values have changed."


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


# test_alma()
# test_ska_mid()
# test_lofar()
# test_meerkat()
# test_global_vlbi()
# test_vlba()
# test_ngeht()
# test_ephemeris()
# test_single_dish()
