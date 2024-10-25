import pytest


@pytest.mark.parametrize(
    "input_name, partition_scheme, expected_estimation",
    [
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            [],
            (0.002607484348118305, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID"],
            (0.002607484348118305, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID", "SCAN_NUMBER"],
            (0.002607484348118305, 4, 1),
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
