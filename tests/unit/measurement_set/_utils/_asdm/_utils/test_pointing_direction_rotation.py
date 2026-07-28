from contextlib import nullcontext as no_raises

import numpy as np

import pytest

from xradio.measurement_set._utils._asdm._utils.pointing_direction_rotation import (
    rotate_offset_to_target,
)


@pytest.mark.parametrize(
    "input_target, input_offset, expected_output, expected_error",
    [
        (None, None, None, pytest.raises(TypeError, match="NoneType")),
        (
            np.array([[[np.pi / 2, np.pi / 2]]]),
            np.array([[[0.001, 0.0]]]),
            [[[3.14259, 0.0]]],
            no_raises(),
        ),
        (
            np.array([[[np.pi / 2, np.pi / 2]]]),
            np.array([[[0.001, 0.001]]]),
            [[[3.14259, 1e-3]]],
            no_raises(),
        ),
        (
            np.array([[[np.pi / 2, np.pi / 2]]]),
            np.array([[[0.0, 0.0]]]),
            [[[np.pi, 0.0]]],
            no_raises(),
        ),
        (
            np.array([[[0.90, 0.35]]]),
            np.array([[[0.01, 0.001]]]),
            [[[2.4733, 0.0097366]]],
            no_raises(),
        ),
        (
            np.array([[[0.90, 0.35]]]),
            np.array([[[0.0, 0.0]]]),
            [[[2.4708, 0.0]]],
            no_raises(),
        ),
    ],
)
def test_rotate_offset_to_target(
    input_target, input_offset, expected_output, expected_error
):
    with expected_error:
        result = rotate_offset_to_target(input_target, input_offset)
        assert np.allclose(result, expected_output, rtol=1e-5)
