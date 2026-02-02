from contextlib import nullcontext as no_raises
import numpy as np

import pytest

import pyasdm


@pytest.mark.parametrize(
    "times_asdm, expected_output, expected_error",
    [
        (0, None, pytest.raises(TypeError, match="object is not subscriptable")),
        (np.array([0, 1, 2]), [-3.5067168e09] * 3, no_raises()),
        (np.array([0, 1, 2]), [-3.5067168e09] * 3, no_raises()),
        (np.array([pyasdm.types.ArrayTime(0)]), [-3.5067168e09] * 3, no_raises()),
        (
            np.array([pyasdm.types.ArrayTime(0), pyasdm.types.ArrayTime(1e9)]),
            np.array([-3506716800, 86396493283200]),
            no_raises(),
        ),
        (
            np.array(
                [
                    pyasdm.types.ArrayTime(1e9),
                    pyasdm.types.ArrayTime(2e9),
                    pyasdm.types.ArrayTime(3e9),
                ]
            ),
            np.array([86396493283200, 172796493283200, 259196493283200]),
            no_raises(),
        ),
    ],
)
def test_convert_time_asdm_to_unix(times_asdm, expected_output, expected_error):
    from xradio.measurement_set._utils._asdm._utils.time import (
        convert_time_asdm_to_unix,
    )

    with expected_error:
        converted = convert_time_asdm_to_unix(times_asdm)
        assert (converted == expected_output).all()
