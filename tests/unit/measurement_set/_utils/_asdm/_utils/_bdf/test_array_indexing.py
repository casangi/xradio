import pytest


@pytest.mark.parametrize(
    "input_slice, default_min, default_max, expected_result",
    [
        (slice(None, None), 0, 33, (0, 33)),
        (slice(0, None), 0, 33, (0, 33)),
        (slice(10, None), 0, 133, (10, 133)),
        (slice(None, 20), 0, 233, (0, 20)),
        (slice(None, 34), 0, 333, (0, 34)),
        (slice(30, 40), 0, 40, (30, 40)),
        (0, 0, 33, (0, 1)),
        (3, 0, 4, (3, 4)),
        (4, 3, 1112, (4, 5)),
    ],
)
def test_min_max_from_dimension_slice(
    input_slice, default_min, default_max, expected_result
):
    from xradio.measurement_set._utils._asdm._utils._bdf.array_indexing import (
        min_max_from_dimension_slice,
    )

    result = min_max_from_dimension_slice(input_slice, default_min, default_max)
    assert result == expected_result
