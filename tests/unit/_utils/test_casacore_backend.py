"""
The CASA table backend of xradio (xradio._utils._casacore.backend): arcae,
python-casacore or casatools only where XRADIO_CASACORE_BACKEND selects them,
and no fallback. Each test runs in a new python process, as the backend is
chosen when the module is imported.
"""

import os
import subprocess
import sys
import textwrap

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


def test_without_arcae_there_is_no_fallback(tmp_path):
    """Without arcae the conversion functions are absent and the backend
    names arcae, even where python-casacore is installed."""
    result = _run_python(
        f"""
        import sys

        sys.modules["arcae"] = None

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
