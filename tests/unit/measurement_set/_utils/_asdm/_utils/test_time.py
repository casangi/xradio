import datetime

import numpy as np
import pyasdm
import pytest
from astropy.time import Time

#: 2024-01-01T12:00:00 as ASDM ArrayTime (ns since 1858-11-17)
NS_2024 = (60310 * 86400 + 12 * 3600) * 10**9
#: same instant, seconds since 1970-01-01
UNIX_2024 = 1704110400.0


def test_time_measure_labels():
    from xradio.measurement_set._utils._asdm._utils.time import (
        ASDM_TIME_FORMAT,
        ASDM_TIME_SCALE,
        MJD_TO_UNIX_TIME_DELTA,
        MJD_TO_UNIX_TIME_DELTA_NS,
    )

    assert ASDM_TIME_FORMAT == "unix"
    assert ASDM_TIME_SCALE == "utc"
    # MJD 40587 is 1970-01-01
    assert MJD_TO_UNIX_TIME_DELTA == 40587 * 86400
    assert MJD_TO_UNIX_TIME_DELTA_NS == MJD_TO_UNIX_TIME_DELTA * 10**9
    assert isinstance(MJD_TO_UNIX_TIME_DELTA_NS, int)


@pytest.mark.parametrize(
    "times_asdm, expected_output",
    [
        (np.array([NS_2024]), np.array([UNIX_2024])),
        ([NS_2024, NS_2024 + 1_500_000_000], np.array([UNIX_2024, UNIX_2024 + 1.5])),
        (np.array([pyasdm.types.ArrayTime(NS_2024)]), np.array([UNIX_2024])),
        (
            np.array(
                [pyasdm.types.ArrayTime(NS_2024), pyasdm.types.ArrayTime(NS_2024 + 1)]
            ),
            np.array([UNIX_2024, UNIX_2024 + 1e-9]),
        ),
        (np.array([float(NS_2024)]), np.array([UNIX_2024])),
        # the epochs
        (np.array([0]), np.array([-3_506_716_800.0])),
        (np.array([3_506_716_800 * 10**9]), np.array([0.0])),
    ],
)
def test_convert_time_asdm_to_unix(times_asdm, expected_output):
    from xradio.measurement_set._utils._asdm._utils.time import (
        convert_time_asdm_to_unix,
    )

    converted = convert_time_asdm_to_unix(times_asdm)
    assert converted.dtype == np.float64
    np.testing.assert_allclose(converted, expected_output, rtol=0, atol=1e-6)


def test_convert_time_asdm_to_unix_integer_precision():
    """The epoch is subtracted in integer ns: 1 ns after the unix epoch is 1e-9 s."""
    from xradio.measurement_set._utils._asdm._utils.time import (
        MJD_TO_UNIX_TIME_DELTA_NS,
        convert_time_asdm_to_unix,
    )

    ns = np.array([MJD_TO_UNIX_TIME_DELTA_NS + 1, MJD_TO_UNIX_TIME_DELTA_NS + 7])
    assert convert_time_asdm_to_unix(ns).tolist() == [1e-9, 7e-9]


def test_convert_time_asdm_to_unix_decodes_to_true_utc():
    """Decoded with astropy using the labels (format/scale) the backend gives these
    values, the instants are the true calendar instants of the ArrayTime values."""
    from xradio.measurement_set._utils._asdm._utils.time import (
        ASDM_TIME_FORMAT,
        ASDM_TIME_SCALE,
        convert_time_asdm_to_unix,
    )

    ns = np.array([NS_2024, NS_2024 + 123_456_789_000])
    decoded = Time(
        convert_time_asdm_to_unix(ns), format=ASDM_TIME_FORMAT, scale=ASDM_TIME_SCALE
    )
    epoch = datetime.datetime(1858, 11, 17)
    expected = Time(
        [epoch + datetime.timedelta(microseconds=int(val) // 1000) for val in ns],
        scale="utc",
    )
    assert np.max(np.abs((decoded - expected).to_value("s"))) < 1e-6
    assert decoded[0].isot == "2024-01-01T12:00:00.000"


def test_convert_time_asdm_to_unix_shapes():
    from xradio.measurement_set._utils._asdm._utils.time import (
        convert_time_asdm_to_unix,
    )

    scalar = convert_time_asdm_to_unix(NS_2024)
    assert scalar.shape == ()
    assert float(scalar) == UNIX_2024

    empty = convert_time_asdm_to_unix([])
    assert empty.shape == (0,)
    assert empty.dtype == np.float64

    two_d = convert_time_asdm_to_unix(np.full((2, 3), NS_2024))
    assert two_d.shape == (2, 3)
    np.testing.assert_array_equal(two_d, np.full((2, 3), UNIX_2024))


@pytest.mark.parametrize(
    "times_asdm, match",
    [
        (np.array(["2024-01-01"]), "dtype"),
        # ArrayTime(float) is MJD days: 1e9 days does not fit int64 ns
        (np.array([pyasdm.types.ArrayTime(1e9)]), "int64"),
        (np.array([None, 3]), "int64"),
    ],
)
def test_convert_time_asdm_to_unix_errors(times_asdm, match):
    from xradio.measurement_set._utils._asdm._utils.time import (
        convert_time_asdm_to_unix,
    )

    with pytest.raises(TypeError, match=match):
        convert_time_asdm_to_unix(times_asdm)


def test_convert_time_asdm_to_datetime64():
    from xradio.measurement_set._utils._asdm._utils.time import (
        convert_time_asdm_to_datetime64,
    )

    expected = np.datetime64("2024-01-01T12:00:00", "ns")
    scalar = convert_time_asdm_to_datetime64(NS_2024)
    assert scalar.shape == ()
    assert scalar == expected
    assert str(scalar) == "2024-01-01T12:00:00.000000000"

    # integer arithmetic: no precision lost at ns level
    values = convert_time_asdm_to_datetime64(
        [pyasdm.types.ArrayTime(NS_2024), pyasdm.types.ArrayTime(NS_2024 + 1)]
    )
    assert values.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(
        values, [expected, expected + np.timedelta64(1, "ns")]
    )

    empty = convert_time_asdm_to_datetime64(np.array([], dtype=np.int64))
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype("datetime64[ns]")

    with pytest.raises(TypeError, match="datetime64"):
        convert_time_asdm_to_datetime64(np.array([1.5e18]))
