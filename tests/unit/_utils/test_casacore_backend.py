"""
The CASA table backend of xradio (xradio._utils._casacore.backend): arcae,
python-casacore or casatools only where XRADIO_CASACORE_BACKEND selects them,
and no fallback. The backend is chosen when the module is imported: most
tests run in a new python process.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import types

import pytest

_CASA_PACKAGES = ("casacore", "casatools", "casaconfig")


def _run_python(code: str, cwd, backend: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("XRADIO_CASACORE_BACKEND", None)
    if backend is not None:
        env["XRADIO_CASACORE_BACKEND"] = backend
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_arcae_imports_neither_casacore_nor_casatools(tmp_path):
    """By default an MSv2 conversion and a CASA image write and read never
    try to import python-casacore or casatools, installed or not."""
    pytest.importorskip("arcae", reason="the default backend is arcae")
    result = _run_python(
        f"""
        import sys

        attempted = set()


        class Recorder:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {_CASA_PACKAGES!r}:
                    attempted.add(name)
                return None


        sys.meta_path.insert(0, Recorder())

        import numpy as np

        from xradio.image import make_empty_sky_image, open_image, write_image
        from xradio.measurement_set import (
            convert_msv2_to_processing_set,
            open_processing_set,
        )
        from xradio.testing.measurement_set import gen_minimal_ms

        msname, _ = gen_minimal_ms("tiny.ms")
        convert_msv2_to_processing_set(
            msname,
            out_file="tiny.ps.zarr",
            partition_scheme=[],
            partition_filter=lambda p: p["DATA_DESC_ID"][0] == 0,
            persistence_mode="w",
        )
        assert len(open_processing_set("tiny.ps.zarr"))

        xds = make_empty_sky_image(
            phase_center=[0.2, -0.5],
            image_size=[4, 3],
            cell_size=[1e-4, 1e-4],
            frequency_coords=[1.4e9],
            pol_coords=["I"],
            time_coords=[59000.5],
        )
        sky = np.arange(12, dtype=np.float32).reshape(1, 1, 1, 4, 3)
        xds["SKY"] = (
            ("time", "frequency", "polarization", "l", "m"),
            sky,
            {{"type": "sky", "units": "Jy/beam"}},
        )
        xds.attrs["data_groups"] = {{"base": {{"sky": "SKY"}}}}
        write_image(xds, "tiny.im", out_format="casa")
        np.testing.assert_array_equal(open_image("tiny.im").SKY.values, sky)

        loaded = [m for m in sys.modules if m.split(".")[0] in {_CASA_PACKAGES!r}]
        assert not attempted and not loaded, (attempted, loaded)
        """,
        tmp_path,
        None,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", ["arcae", "pyarrow"])
def test_without_arcae_there_is_no_fallback(tmp_path, missing):
    """Without arcae (or pyarrow) the conversion functions are absent and the
    backend names arcae, even where python-casacore is installed."""
    result = _run_python(
        f"""
        import sys

        sys.modules[{missing!r}] = None

        import pytest

        import xradio.measurement_set as ms

        assert not hasattr(ms, "convert_msv2_to_processing_set")
        with pytest.raises(ModuleNotFoundError, match="arcae"):
            import xradio._utils._casacore.backend
        loaded = [m for m in sys.modules if m.split(".")[0] in {_CASA_PACKAGES!r}]
        assert not loaded, loaded
        """,
        tmp_path,
        None,
    )
    assert result.returncode == 0, result.stderr


def test_a_broken_arcae_is_not_reported_as_missing(tmp_path):
    """An ImportError other than a missing arcae or pyarrow (here a broken
    arcae build) propagates, so the conversion functions do not silently
    go missing."""
    result = _run_python(
        """
        import importlib
        import importlib.abc
        import importlib.util
        import sys

        import pytest


        class BrokenArcae(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path=None, target=None):
                if name == "arcae":
                    return importlib.util.spec_from_loader(name, self)
                return None

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ImportError("undefined symbol: casacore")


        sys.meta_path.insert(0, BrokenArcae())

        for name in ("xradio.measurement_set", "xradio._utils._casacore.backend"):
            with pytest.raises(ImportError, match="undefined symbol") as info:
                importlib.import_module(name)
            assert not isinstance(info.value, ModuleNotFoundError), info.value
        """,
        tmp_path,
        None,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["casacore", "python-casacore"])
def test_python_casacore_selected(tmp_path, backend):
    pytest.importorskip("casacore.tables", reason="python-casacore is not installed")
    result = _run_python(
        """
        import sys

        from casacore import tables

        from xradio._utils._casacore import backend

        assert backend.BACKEND == "casacore"
        assert backend.tables is tables
        assert "casatools" not in sys.modules
        """,
        tmp_path,
        backend,
    )
    assert result.returncode == 0, result.stderr


def test_casatools_selected(tmp_path):
    """casatools is the backend where it is installed; elsewhere the
    conversion functions are absent."""
    result = _run_python(
        """
        import importlib.util
        import sys

        import xradio.measurement_set as ms

        try:
            installed = importlib.util.find_spec("casatools") is not None
        except ImportError:
            installed = False
        if installed:
            from xradio._utils._casacore import backend, tables

            assert backend.BACKEND == "casatools"
            assert tables.uses_casatools()
        else:
            assert not hasattr(ms, "convert_msv2_to_processing_set")
        assert "casacore" not in sys.modules
        """,
        tmp_path,
        "casatools",
    )
    assert result.returncode == 0, result.stderr


def test_unknown_backend_is_an_error(tmp_path):
    result = _run_python(
        """
        import pytest

        with pytest.raises(ValueError, match="XRADIO_CASACORE_BACKEND='arcea'"):
            import xradio._utils._casacore.backend
        """,
        tmp_path,
        "arcea",
    )
    assert result.returncode == 0, result.stderr


# The same selection in this process: a new module object is executed from the
# backend's source with XRADIO_CASACORE_BACKEND set and stub modules in
# sys.modules, so the backend that xradio has imported is left unchanged.

_PACKAGE = "xradio._utils._casacore"
_TABLEUTIL = ("makearrcoldesc", "makedminfo", "makescacoldesc", "maketabdesc")
_MS_HELPERS = (
    "default_ms",
    "default_ms_subtable",
    "complete_ms_desc",
    "required_ms_desc",
) + _TABLEUTIL


def _select(monkeypatch, backend: str | None):
    if backend is None:
        monkeypatch.delenv("XRADIO_CASACORE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("XRADIO_CASACORE_BACKEND", backend)
    spec = importlib.util.find_spec(f"{_PACKAGE}.backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reimport_arcae_shim(monkeypatch):
    """Make the selection import the arcae shim again instead of reusing it."""
    package = importlib.import_module(_PACKAGE)
    monkeypatch.delitem(sys.modules, f"{_PACKAGE}.casacore_from_arcae", raising=False)
    monkeypatch.delattr(package, "casacore_from_arcae", raising=False)


def test_arcae_selected_by_default(monkeypatch):
    pytest.importorskip("arcae", reason="the default backend is arcae")
    from xradio._utils._casacore import casacore_from_arcae, images_from_arcae

    backend = _select(monkeypatch, None)
    assert backend.BACKEND == "arcae"
    assert backend.tables is casacore_from_arcae
    assert backend.images is backend.coordinates is images_from_arcae
    for name in _MS_HELPERS:
        assert getattr(backend, name) is getattr(casacore_from_arcae, name), name


@pytest.mark.parametrize("missing", ["arcae", "pyarrow"])
def test_missing_arcae_is_reported_in_process(monkeypatch, missing):
    pytest.importorskip("arcae")
    _reimport_arcae_shim(monkeypatch)
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ModuleNotFoundError, match="arcae cannot be imported") as info:
        _select(monkeypatch, "arcae")
    assert info.value.__cause__.name == missing


def test_other_missing_module_of_the_arcae_shim_propagates(monkeypatch):
    pytest.importorskip("arcae")
    _reimport_arcae_shim(monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ModuleNotFoundError) as info:
        _select(monkeypatch, None)
    assert info.value.name == "numpy"
    assert "arcae cannot be imported" not in str(info.value)


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.mark.parametrize("value", ["casacore", "python-casacore"])
def test_python_casacore_selected_in_process(monkeypatch, value):
    coordinates = _module("casacore.images.coordinates")
    images = _module("casacore.images", coordinates=coordinates)
    msutil = _module(
        "casacore.tables.msutil", complete_ms_desc=object(), required_ms_desc=object()
    )
    tableutil = _module(
        "casacore.tables.tableutil", **{name: object() for name in _TABLEUTIL}
    )
    tables = _module(
        "casacore.tables",
        default_ms=object(),
        default_ms_subtable=object(),
        msutil=msutil,
        tableutil=tableutil,
    )
    for module in (coordinates, images, msutil, tableutil, tables):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules, "casacore", _module("casacore", tables=tables, images=images)
    )

    backend = _select(monkeypatch, value)
    assert backend.BACKEND == "casacore"
    assert backend.tables is tables
    assert backend.images is images
    assert backend.coordinates is coordinates
    assert backend.default_ms is tables.default_ms
    assert backend.default_ms_subtable is tables.default_ms_subtable
    assert backend.complete_ms_desc is msutil.complete_ms_desc
    assert backend.required_ms_desc is msutil.required_ms_desc
    for name in _TABLEUTIL:
        assert getattr(backend, name) is getattr(tableutil, name), name


def test_missing_python_casacore_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "casacore", None)
    monkeypatch.setitem(sys.modules, "casacore.tables", None)
    with pytest.raises(ModuleNotFoundError, match="Backend 'casacore' is not"):
        _select(monkeypatch, "python-casacore")


def test_casatools_selected_in_process(monkeypatch):
    shim = _module(f"{_PACKAGE}.casacore_from_casatools")
    monkeypatch.setitem(sys.modules, shim.__name__, shim)
    monkeypatch.delattr(
        importlib.import_module(_PACKAGE), "casacore_from_casatools", raising=False
    )
    backend = _select(monkeypatch, "casatools")
    assert backend.BACKEND == "casatools"
    assert backend.tables is backend.images is backend.coordinates is shim


def test_unknown_backend_is_an_error_in_process(monkeypatch):
    with pytest.raises(ValueError, match="XRADIO_CASACORE_BACKEND='arcea'"):
        _select(monkeypatch, "arcea")
