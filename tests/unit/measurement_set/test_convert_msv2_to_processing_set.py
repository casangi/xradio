import shutil

import xarray as xr

import pytest

mem_estimate_min_ms = 0.011131428182125092


@pytest.mark.parametrize(
    "input_name, partition_scheme, expected_estimation",
    [
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            [],
            (mem_estimate_min_ms, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID"],
            (mem_estimate_min_ms, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID", "SCAN_NUMBER"],
            (mem_estimate_min_ms, 4, 1),
        ),
    ],
)
def test_estimate_conversion_memory_and_cores_minimal(
    input_name, partition_scheme, expected_estimation, request
):
    from xradio.measurement_set import estimate_conversion_memory_and_cores

    fixture = request.getfixturevalue(input_name)
    input_path = fixture.fname if isinstance(fixture, tuple) else fixture

    res = estimate_conversion_memory_and_cores(
        input_path, partition_scheme=partition_scheme
    )
    assert res[0] == pytest.approx(expected_estimation[0], rel=1e-6)
    assert res[1:] == expected_estimation[1:]


@pytest.mark.parametrize(
    "input_path, expected_error",
    [
        (
            "inexistent_foo_path_.mms",
            pytest.raises(RuntimeError, match="does not exist"),
        ),
    ],
)
def test_estimate_conversion_memory_and_cores_with_errors(input_path, expected_error):
    from xradio.measurement_set import estimate_conversion_memory_and_cores

    # if not input_path:
    #    input_path = "ms_minimal_required.ms"
    with expected_error:
        res = estimate_conversion_memory_and_cores(input_path, [])
        assert res


def test_convert_msv2_to_processing_set_with_other_opts(ms_minimal_misbehaved):
    """Uses a few options that are not exercised in other tests"""
    from xradio.measurement_set import (
        convert_msv2_to_processing_set,
        open_processing_set,
    )
    from xradio.schema.check import check_datatree

    out_path = "test_convert_msv2_to_proc_set_without_ps_zarr_ending"
    out_path_with_ending = out_path + ".ps.zarr"
    try:
        convert_msv2_to_processing_set(
            ms_minimal_misbehaved.fname,
            out_file=out_path,
            partition_scheme=["FIELD_ID"],
            overwrite=False,
            parallel_mode="bogus_mode",
        )
        ps_xdt = xr.open_datatree(out_path_with_ending,engine="zarr")
        check_datatree(ps_xdt)

        # TODO: break this out to a proper test_open_processing_set:
        open_xdt = open_processing_set(out_path_with_ending, intents="faulty")
        check_datatree(open_xdt)

    finally:
        shutil.rmtree(out_path_with_ending)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
