from contextlib import nullcontext as no_raises
import pandas as pd
import pytest


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


def test_find_spw_in_basebands_list_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_spw_in_basebands_list,
    )

    baseband_idx, spw_idx = find_spw_in_basebands_list(
        "bogus_path_non_existant.nope", 0, []
    )
    assert baseband_idx == 0
    assert spw_idx == 0


def test_find_different_basebands_spws_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_different_basebands_spws,
    )

    result = find_different_basebands_spws([])
    assert result is False


def test_find_different_basebands_pols_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_different_basebands_pols,
    )

    result = find_different_basebands_pols([])
    assert result is False
