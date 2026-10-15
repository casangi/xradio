from unittest import mock

import pyasdm
import pytest
import xarray as xr


def test_ASDMBackendEntryPoint_in_xarray_backends_list():
    """This test is actually for the entry-point defined in the pyproject.toml"""

    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    expected_name = "xradio_asdm"
    engines = xr.backends.list_engines()
    assert expected_name in engines
    assert isinstance(engines[expected_name], ASDMBackendEntryPoint)


def test_open_datatree_drop_variables_not_supported():
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(RuntimeError, match="not supported"):
        asdm_backend.open_datatree(bogus_path, drop_variables=["a", "b", "c"])


def test_open_datatree_fails():
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(pyasdm.exceptions.ConversionException, match="Cannot convert"):
        asdm_backend.open_datatree(bogus_path)


def test_open_datatree_mocked_open_asdm():
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    assert asdm_backend.supports_groups
    assert asdm_backend.url and isinstance(asdm_backend.url, str)

    bogus_path = "/bogus/inexistent/test_path"
    with mock.patch(
        "xradio.measurement_set._utils._asdm.open_asdm.open_asdm"
    ) as mock_open_asdm:
        dummy_asdm_ps = 44
        mock_open_asdm.side_effect = [dummy_asdm_ps]

        asdm_ps = asdm_backend.open_datatree(bogus_path)
        assert asdm_ps == dummy_asdm_ps
        mock_open_asdm.assert_called_once()
        assert mock_open_asdm.call_args[0] == (bogus_path, None, None, None, None, None)


def test_xarray_open_groups():
    """open_groups should give the Processing Set nodes (/partitions)"""

    group_0_name = "/uid___A002_X...._X...._0"
    group_1_name = "/uid___A002_X...._X...._0"
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"type": "fake_processing_set"}),
            group_0_name: xr.Dataset({"VISIBILITY": ("time", [1.0, 2.0])}),
            group_1_name: xr.Dataset({"VISIBILITY": ("time", [1.0, 2.0])}),
        }
    )
    with mock.patch(
        "xradio.measurement_set._utils._asdm.open_asdm.open_asdm"
    ) as mock_open_asdm:
        mock_open_asdm.side_effect = [tree]
        groups = xr.open_groups("/bogus/inexistent/test_path", engine="xradio_asdm")

    assert sorted(groups) == ["/", group_0_name]
    assert groups["/"].attrs["type"] == "fake_processing_set"
    assert groups[group_0_name].VISIBILITY.values.tolist() == [1.0, 2.0]


def test_guess_can_open_except():
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    # something that does not cast to Path cleanly
    bogus_path = 33
    guess = asdm_backend.guess_can_open(bogus_path)
    assert not guess


def test_guess_can_open_inexistent():
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    guess = asdm_backend.guess_can_open(bogus_path)
    assert not guess


def test_guess_can_open_simple(empty_asdm_on_disk):
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint,
    )

    asdm_backend = ASDMBackendEntryPoint()
    guess = asdm_backend.guess_can_open(empty_asdm_on_disk)
    assert guess
    guess_str = asdm_backend.guess_can_open(str(empty_asdm_on_disk))
    assert guess_str
