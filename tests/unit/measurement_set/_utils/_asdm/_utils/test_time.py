from contextlib import nullcontext as no_raises

import numpy as np
import pyasdm
import pytest

MJD_TO_UNIX_TIME_DELTA = 3_506_716_800


@pytest.mark.parametrize(
    "times_asdm, expected_output, expected_error",
    [
        (0, None, pytest.raises(TypeError, match="has no len")),
        (
            np.array([0, 1, 5]),
            np.array([0, 1, 5]) - MJD_TO_UNIX_TIME_DELTA,
            no_raises(),
        ),
        (
            np.array([1e9, 3e9, 5e9]),
            np.array([1e9, 3e9, 5e9]) - MJD_TO_UNIX_TIME_DELTA,
            no_raises(),
        ),
        (
            np.array([pyasdm.types.ArrayTime(0)]),
            np.array([-MJD_TO_UNIX_TIME_DELTA], dtype="float"),
            no_raises(),
        ),
        (
            np.array([pyasdm.types.ArrayTime(1), pyasdm.types.ArrayTime(int(1.5e18))]),
            np.array([-MJD_TO_UNIX_TIME_DELTA, -2006716800.0], dtype="float"),
            no_raises(),
        ),
        (
            np.array(
                [
                    pyasdm.types.ArrayTime(),
                    pyasdm.types.ArrayTime("1858-11-17T00:00:00.0"),
                    pyasdm.types.ArrayTime("2015-05-11T02:30:15.1"),
                ]
            ),
            np.array(
                [-MJD_TO_UNIX_TIME_DELTA, -MJD_TO_UNIX_TIME_DELTA, 1431311415.1000001],
                dtype="float",
            ),
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
        assert converted.dtype == expected_output.dtype
        # assert (converted == expected_output).all()
        np.testing.assert_allclose(converted, expected_output, rtol=1e-15)
