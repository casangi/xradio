"""
convert_msv2_to_processing_set of downloaded test MSs with casatools, compared
with the processing sets that python-casacore gives
(reference_python_casacore.json, make_reference.py): every MSv4 and
sub-dataset, every variable's dims, shape, values (NaN payloads included) and
attributes, the coordinates and the attributes of every dataset, bit for
bit, but for the values of the variables computed with libm or interpolated
(OBSERVER_POSITION, and the ephemeris variables interpolated to the MSv4
times, e.g. FIELD_PHASE_CENTER_DIRECTION of the ALMA MS), whose last bits may
differ on other platforms (macOS arm64): to 1e-12 of their largest magnitude
(reference.approx_variables).

The cases (reference.CONVERSION_CASES) cover TiledShapeStMan, TiledColumnStMan,
IncrementalStMan and StandardStMan MAIN columns, cells of several shapes in one
column, decreasing frequencies, ephemeris fields, WEIGHT_SPECTRUM, undefined
WEIGHT (WEIGHT=1), POINTING (with_pointing=True), partition schemes [] and
["FIELD_ID"]; every case with the default main_chunksize and some with
explicit chunks (time steps or GiB), streamed writes of several batches and
the lazy reads of parallel_mode="time". With casatools no sub-table cache is
created, and every partition is converted with the same attempts as with
python-casacore (no column found unreadable only by the read).

Known difference with casatools (pre-existing, also on main): casatools
returns Float and Complex cells as float64 and complex128, so the data
variables read from them (VISIBILITY*, SPECTRUM*, WEIGHT), whose dtype is
that of the first cell, are float64 / complex128 instead of float32 /
complex64, with the same values.

Skipped where casatools is not installed or python-casacore is (the Linux and
macOS workflows), see reference.skip_unless_casatools_backend.
"""

import shutil

import numpy as np
import pytest

from tests.casatools import reference as ref

ref.skip_unless_casatools_backend()

# The main dataset variables casatools gives wider dtypes (see the docstring)
WIDENED_VARIABLES = ("VISIBILITY", "SPECTRUM", "WEIGHT")


@pytest.fixture(scope="module")
def reference() -> dict:
    return ref.load_reference()


def _widened_as_known(line: str) -> bool:
    """Whether a widened variable ("<msv4>/<path>/<var>: <dtype> -> <dtype>",
    fingerprint_differences) is a main dataset variable read from a Float or
    Complex MAIN column."""
    where = line.split(":", 1)[0]
    _, path, var = where.split("/", 2)
    return path == "." and var.startswith(WIDENED_VARIABLES)


@pytest.mark.parametrize(
    "case, variant",
    [
        (case, variant)
        for case, variants in ref.CASE_VARIANTS.items()
        for variant in variants
    ],
)
def test_convert_msv2_to_processing_set(case, variant, reference, tmp_path):
    out_file = tmp_path / f"{case}_{variant}.ps.zarr"
    try:
        with ref.ConversionSpy() as spy:
            ref.convert(case, variant, out_file)
        actual = ref.processing_set_fingerprint(out_file)
    finally:
        shutil.rmtree(out_file, ignore_errors=True)

    expected = dict(reference["conversions"][case], nodes=reference["nodes"])
    widened = []
    diffs = ref.fingerprint_differences(expected, actual, widened)
    assert not diffs, "\n".join(diffs)
    unexpected = [line for line in widened if not _widened_as_known(line)]
    assert not unexpected, "\n".join(unexpected)

    assert spy.subtable_caches == 0
    assert spy.attempts == expected["attempts"]
    options = ref.CONVERSION_VARIANTS[variant]
    if options.get("parallel_mode") == "time":
        # lazy (dask) reads of the MAIN columns, written by to_zarr
        assert not spy.streamed and not spy.read_whole
    else:
        assert len(spy.streamed) + len(spy.read_whole) == len(spy.attempts)
    if "stream_batch_bytes" in options:
        # streamed writes, some variables in several batches
        assert spy.streamed
        assert any(
            stats["summary"]["batches"] > stats["summary"]["variables"]
            for stats in spy.streamed
        )


def test_approx_values_compared_with_a_tolerance(reference):
    """
    The values of the variables computed with libm or interpolated
    (reference.approx_variables: OBSERVER_POSITION and the ephemeris
    variables of the ALMA MS) are compared with a tolerance relative to
    their largest magnitude: differences in the last bits (another platform's
    libm or compiler) pass, larger ones do not.
    """
    expected = dict(reference["conversions"]["alma"], nodes=reference["nodes"])
    kept = {
        name for node in reference["nodes"].values() for name in node.get("approx", {})
    }
    assert kept == {
        "FIELD_PHASE_CENTER_DIRECTION",
        "FIELD_PHASE_CENTER_DISTANCE",
        "OBSERVER_POSITION",
    }

    def scaled(factor: float) -> dict:
        """The expected fingerprint with the kept values scaled by factor."""
        nodes, msv4 = dict(reference["nodes"]), {}
        for name, paths in expected["msv4"].items():
            msv4[name] = dict(paths)
            for path, node_id in paths.items():
                node = reference["nodes"][node_id]
                if "approx" in node:
                    approx = {
                        var: (np.asarray(values, dtype=np.float64) * factor).tolist()
                        for var, values in node["approx"].items()
                    }
                    msv4[name][path] = f"{node_id}-scaled"
                    nodes[f"{node_id}-scaled"] = dict(node, approx=approx)
        return {"msv4": msv4, "nodes": nodes, "attrs": expected["attrs"]}

    assert not ref.fingerprint_differences(expected, scaled(1 + 4e-16))
    diffs = ref.fingerprint_differences(expected, scaled(1 + 1e-9))
    assert diffs and all(": values differ: " in line for line in diffs)
