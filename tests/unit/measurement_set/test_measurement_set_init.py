"""Tests of the public xradio.measurement_set namespace (lazy open_asdm export)."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import xradio

# Directory containing the xradio package imported by this test session (for example
# src/ of a source checkout). It is put first on the PYTHONPATH of the subprocesses,
# so that they test the same xradio as this process, even if the sys.path of this
# process was set up by pytest (rootdir, pythonpath option) or another xradio
# (for example an older release) is installed in the environment.
XRADIO_PARENT_DIR = str(pathlib.Path(xradio.__file__).resolve().parents[1])


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter (sys.modules state must be pristine)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (XRADIO_PARENT_DIR, env.get("PYTHONPATH")) if path
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_subprocesses_import_the_xradio_under_test():
    result = _run_python(
        """
        import xradio
        print(xradio.__file__)
        """
    )
    assert result.returncode == 0, result.stderr
    imported = pathlib.Path(result.stdout.strip().splitlines()[-1]).resolve()
    assert imported == pathlib.Path(xradio.__file__).resolve()


def test_import_measurement_set_does_not_import_pyasdm():
    result = _run_python(
        """
        import sys
        import xradio.measurement_set
        print("pyasdm" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_open_asdm_public_export_is_the_backend_function():
    pytest.importorskip("pyasdm")
    import xradio.measurement_set as ms
    from xradio.measurement_set import open_asdm
    from xradio.measurement_set._utils._asdm.open_asdm import (
        open_asdm as private_open_asdm,
    )

    assert open_asdm is private_open_asdm
    assert ms.open_asdm is private_open_asdm
    assert "open_asdm" in ms.__all__
    assert "open_asdm" in dir(ms)


def test_unknown_attribute_still_raises_attribute_error():
    import xradio.measurement_set as ms

    with pytest.raises(AttributeError, match="has no attribute 'not_a_function'"):
        _ = ms.not_a_function

    assert not hasattr(ms, "not_a_function")


def test_open_asdm_without_pyasdm_gives_install_hint():
    """With pyasdm unavailable the package imports and open_asdm explains the extra."""
    result = _run_python(
        """
        import sys
        sys.modules["pyasdm"] = None  # make "import pyasdm" fail

        import xradio.measurement_set as ms
        print("in_all", "open_asdm" in ms.__all__)
        print("in_dir", "open_asdm" in dir(ms))
        try:
            from xradio.measurement_set import open_asdm
        except ModuleNotFoundError as exc:
            print("error_name", exc.name)
            print("error_msg", exc)
        else:
            print("no error")
        """
    )
    assert result.returncode == 0, result.stderr
    lines = dict(line.split(" ", 1) for line in result.stdout.strip().splitlines())
    assert lines["in_all"] == "False"
    assert lines["in_dir"] == "False"
    assert lines["error_name"] == "pyasdm"
    assert 'pip install "xradio[asdm]"' in lines["error_msg"]
    assert "'pyasdm'" in lines["error_msg"]
