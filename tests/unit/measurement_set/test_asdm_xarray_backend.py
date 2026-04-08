from unittest import mock

import pytest

import pyasdm


def test_open_datatree_drop_variables_not_supported():
    from xradio.measurement_set.asdm_xarray_backend import ASDMBackendEntryPoint

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(RuntimeError, match="not supported"):
        asdm_backend.open_datatree(bogus_path, drop_variables=["a", "b", "c"])


def test_open_datatree_fails():
    from xradio.measurement_set.asdm_xarray_backend import ASDMBackendEntryPoint

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(pyasdm.exceptions.ConversionException, match="Cannot convert"):
        asdm_backend.open_datatree(bogus_path)


def test_open_datatree_mocked_open_asdm():
    from xradio.measurement_set.asdm_xarray_backend import ASDMBackendEntryPoint

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with mock.patch(
        "xradio.measurement_set.asdm_xarray_backend.open_asdm"
    ) as mock_open_asdm:
        dummy_asdm_ps = 44
        mock_open_asdm.side_effect = [dummy_asdm_ps]

        asdm_ps = asdm_backend.open_datatree(bogus_path)
        assert asdm_ps == dummy_asdm_ps
        mock_open_asdm.assert_called_once()


def test_guess_can_open_except():
    from xradio.measurement_set.asdm_xarray_backend import ASDMBackendEntryPoint

    asdm_backend = ASDMBackendEntryPoint()
    # something that does not cast to Path cleanly
    bogus_path = 33
    guess = asdm_backend.guess_can_open(bogus_path)
    assert guess == False


def test_guess_can_open_false():
    from xradio.measurement_set.asdm_xarray_backend import ASDMBackendEntryPoint

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    guess = asdm_backend.guess_can_open(bogus_path)
    assert guess == False
