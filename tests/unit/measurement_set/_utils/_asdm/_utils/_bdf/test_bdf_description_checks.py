from contextlib import nullcontext as no_raises

import pytest

import pyasdm


@pytest.mark.parametrize(
    "input_correlation_mode, expected_error",
    [
        (
            pyasdm.enumerations.CorrelationMode.CROSS_ONLY,
            pytest.raises(RuntimeError, match="Unexpected"),
        ),
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, no_raises()),
        (pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO, no_raises()),
    ],
)
def test_check_correlation_mode(input_correlation_mode, expected_error):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        check_correlation_mode,
    )

    with expected_error:
        check_correlation_mode(input_correlation_mode)


@pytest.mark.parametrize(
    "data_array_names, binary_types, bdf_path, expected_error",
    [
        ([], [], "foo_non_existent.nope", no_raises()),
        (
            ["flags"],
            [],
            "foo_non_existent.nope",
            pytest.raises(RuntimeError, match="does not have"),
        ),
        (
            ["flags"],
            ["actualTimes", "actualDurations", "flags", "autoData", "crossData"],
            "foo_non_existent.nope",
            no_raises(),
        ),
        (
            ["crossData", "autoData"],
            [],
            "foo_non_existent.nope",
            pytest.raises(RuntimeError, match="does not have"),
        ),
    ],
)
def test_ensure_presence_binary_components(
    data_array_names, binary_types, bdf_path, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        ensure_presence_binary_components,
    )

    with expected_error:
        ensure_presence_binary_components(data_array_names, binary_types, bdf_path)


@pytest.mark.parametrize(
    "input_names, exclude_also_for_flags, expected_error",
    [
        ([], False, no_raises()),
        (
            ["TIM", "BAL", "ANT", "BAB", "SPW", "BIN", "APC", "SPP", "POL", "ANY"],
            False,
            no_raises(),
        ),
        (
            ["TIM", "BAL", "ANT", "BAB", "SPW", "BIN", "APC", "SPP", "POL", "ANY"],
            True,
            pytest.raises(RuntimeError, match="Unsupported dimension"),
        ),
        (["DIM", "STT"], False, no_raises()),
        (["DIM", "STT"], True, no_raises()),
        (["STO"], False, pytest.raises(RuntimeError, match="STO")),
        (["STO"], True, pytest.raises(RuntimeError, match="STO")),
        (["HOL"], False, pytest.raises(RuntimeError, match="HOL")),
        (["HOL"], True, pytest.raises(RuntimeError, match="HOL")),
        (["STO", "DIM", "HOL"], False, pytest.raises(RuntimeError, match="STO")),
        (["STO", "APC", "SPP"], True, pytest.raises(RuntimeError, match="STO")),
        (["APC", "DIM"], False, no_raises()),
        (["APC", "DIM"], True, pytest.raises(RuntimeError, match="APC")),
        (["DIM", "SPP"], False, no_raises()),
        (["DIM", "SPP"], True, pytest.raises(RuntimeError, match="SPP")),
    ],
)
def test_exclude_unsupported_axis_names(
    input_names, exclude_also_for_flags, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        exclude_unsupported_axis_names,
    )

    with expected_error:
        exclude_unsupported_axis_names(input_names, exclude_also_for_flags)
