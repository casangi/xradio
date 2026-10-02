import io
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from unittest import mock

import pyasdm
import pytest
import xarray as xr

from xradio._asdm_backend_entrypoint import ASDMBackendEntryPoint

OPEN_ASDM = "xradio.measurement_set._utils._asdm.open_asdm.open_asdm"


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter (same environment), to check imports."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_ASDMBackendEntryPoint_in_xarray_backends_list():
    """The entry point defined in pyproject.toml is registered with xarray, and
    the old module path is an alias of the same class."""
    from xradio.measurement_set._utils._asdm.asdm_backend_entrypoint import (
        ASDMBackendEntryPoint as OldPathEntryPoint,
    )

    assert OldPathEntryPoint is ASDMBackendEntryPoint
    engines = xr.backends.list_engines()
    assert "xradio_asdm" in engines
    assert isinstance(engines["xradio_asdm"], ASDMBackendEntryPoint)


def test_entrypoint_module_is_import_light():
    """Loading the entry point (as xarray does for every engine listing) imports
    neither pyasdm nor xradio.measurement_set (F26)."""
    result = run_python(
        """
        import sys
        from xradio._asdm_backend_entrypoint import ASDMBackendEntryPoint
        heavy = [name for name in ("pyasdm", "xradio.measurement_set", "astropy")
                 if name in sys.modules]
        assert not heavy, heavy
        """
    )
    assert result.returncode == 0, result.stderr


def test_list_engines_without_pyasdm():
    """Without pyasdm, listing the xarray engines does not warn (the entry point
    still loads), and opening an ASDM raises an ImportError with the install hint
    (F26)."""
    result = run_python(
        """
        import sys, warnings
        sys.modules["pyasdm"] = None  # makes "import pyasdm" fail
        warnings.simplefilter("error", RuntimeWarning)
        import xarray as xr
        engines = xr.backends.list_engines()
        assert "xradio_asdm" in engines, list(engines)
        try:
            engines["xradio_asdm"].open_datatree("/some/asdm")
        except ImportError as exc:
            assert "pip install 'xradio[asdm]'" in str(exc), str(exc)
            assert "pyasdm" in str(exc), str(exc)
        else:
            raise AssertionError("ImportError not raised")
        """
    )
    assert result.returncode == 0, result.stderr


def test_open_datatree_drop_variables_not_supported():
    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(RuntimeError, match="not supported"):
        asdm_backend.open_datatree(bogus_path, drop_variables=["a", "b", "c"])


def test_open_datatree_fails():
    asdm_backend = ASDMBackendEntryPoint()
    bogus_path = "/bogus/inexistent/test_path"
    with pytest.raises(pyasdm.exceptions.ConversionException, match="Cannot convert"):
        asdm_backend.open_datatree(bogus_path)


def test_open_dataset_not_supported():
    with pytest.raises(NotImplementedError, match="open_datatree"):
        ASDMBackendEntryPoint().open_dataset("/bogus/inexistent/test_path")


def test_open_datatree_mocked_open_asdm():
    """The backend options are forwarded unchanged to open_asdm."""
    asdm_backend = ASDMBackendEntryPoint()
    assert asdm_backend.supports_groups
    assert asdm_backend.url and isinstance(asdm_backend.url, str)

    bogus_path = "/bogus/inexistent/test_path"
    with mock.patch(OPEN_ASDM) as mock_open_asdm:
        dummy_asdm_ps = 44
        mock_open_asdm.side_effect = [dummy_asdm_ps]
        asdm_ps = asdm_backend.open_datatree(bogus_path)
        assert asdm_ps == dummy_asdm_ps
        mock_open_asdm.assert_called_once_with(
            bogus_path,
            partition_scheme=None,
            include_processor_types=None,
            include_spectral_resolution_types=None,
            with_pointing=False,
            pointing_for_only_spectral_resolution_types=None,
        )

    with mock.patch(OPEN_ASDM) as mock_open_asdm:
        asdm_backend.open_datatree(
            bogus_path,
            partition_scheme=[],
            include_processor_types=["RADIOMETER"],
            include_spectral_resolution_types=["CHANNEL_AVERAGE"],
            with_pointing=True,
            pointing_for_only_spectral_resolution_types=["FULL_RESOLUTION"],
        )
        mock_open_asdm.assert_called_once_with(
            bogus_path,
            partition_scheme=[],
            include_processor_types=["RADIOMETER"],
            include_spectral_resolution_types=["CHANNEL_AVERAGE"],
            with_pointing=True,
            pointing_for_only_spectral_resolution_types=["FULL_RESOLUTION"],
        )


def test_xr_open_datatree_forwards_backend_kwargs():
    with mock.patch(OPEN_ASDM) as mock_open_asdm:
        mock_open_asdm.side_effect = [xr.DataTree()]
        xr.open_datatree(
            "/bogus/inexistent/test_path",
            engine="xradio_asdm",
            partition_scheme=["scanNumber"],
            with_pointing=True,
        )
        kwargs = mock_open_asdm.call_args.kwargs
        assert kwargs["partition_scheme"] == ["scanNumber"]
        assert kwargs["with_pointing"] is True


def test_xr_open_groups():
    """supports_groups is True: xr.open_groups gives the processing set nodes."""
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"type": "processing_set"}),
            "/asdm_0": xr.Dataset({"VISIBILITY": ("time", [1.0, 2.0])}),
        }
    )
    with mock.patch(OPEN_ASDM) as mock_open_asdm:
        mock_open_asdm.side_effect = [tree]
        groups = xr.open_groups(
            "/bogus/inexistent/test_path", engine="xradio_asdm", partition_scheme=[]
        )
        assert mock_open_asdm.call_args.kwargs["partition_scheme"] == []
    assert sorted(groups) == ["/", "/asdm_0"]
    assert groups["/"].attrs["type"] == "processing_set"
    assert groups["/asdm_0"].VISIBILITY.values.tolist() == [1.0, 2.0]


@pytest.fixture
def fake_asdm_dir(tmp_path) -> Path:
    asdm_dir = tmp_path / "uid___A002_X1_X2"
    (asdm_dir / "ASDMBinary").mkdir(parents=True)
    (asdm_dir / "ASDM.xml").write_text("<ASDM/>")
    return asdm_dir


def test_guess_can_open_asdm_directory(fake_asdm_dir):
    """True for an ASDM directory given as str or Path (F27)."""
    asdm_backend = ASDMBackendEntryPoint()
    assert asdm_backend.guess_can_open(str(fake_asdm_dir)) is True
    assert asdm_backend.guess_can_open(fake_asdm_dir) is True


def test_guess_can_open_false(fake_asdm_dir, tmp_path):
    asdm_backend = ASDMBackendEntryPoint()
    assert asdm_backend.guess_can_open("/bogus/inexistent/test_path") is False
    assert asdm_backend.guess_can_open(str(fake_asdm_dir / "ASDM.xml")) is False
    assert asdm_backend.guess_can_open(str(tmp_path / "missing")) is False
    no_binary = tmp_path / "no_binary"
    no_binary.mkdir()
    (no_binary / "ASDM.xml").write_text("<ASDM/>")
    assert asdm_backend.guess_can_open(no_binary) is False
    no_xml = tmp_path / "no_xml"
    (no_xml / "ASDMBinary").mkdir(parents=True)
    assert asdm_backend.guess_can_open(no_xml) is False


@pytest.mark.parametrize("obj", [33, None, io.BytesIO(b"abc"), b"/bytes/path"])
def test_guess_can_open_non_path_objects(obj):
    assert ASDMBackendEntryPoint().guess_can_open(obj) is False


def test_engine_auto_detection(fake_asdm_dir, tmp_path):
    """xr.open_datatree / open_dataset pick xradio_asdm for an ASDM directory, and
    guessing on other directories does not warn (F27)."""
    assert (
        xr.backends.plugins.guess_engine(str(fake_asdm_dir), must_support_groups=True)
        == "xradio_asdm"
    )
    with mock.patch(OPEN_ASDM) as mock_open_asdm:
        mock_open_asdm.side_effect = [xr.DataTree()]
        xr.open_datatree(str(fake_asdm_dir))
        assert mock_open_asdm.call_args.args == (str(fake_asdm_dir),)
    other = tmp_path / "other"
    other.mkdir()
    entrypoint = xr.backends.list_engines()["xradio_asdm"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert entrypoint.guess_can_open(str(other)) is False
