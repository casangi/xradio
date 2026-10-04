"""
convert_msv2_to_processing_set of downloaded test MSs with casatools, compared
with the processing sets that python-casacore gives
(reference_python_casacore.json, make_reference.py): every MSv4 and
sub-dataset, every variable's dims, shape, values (NaN payloads included) and
attributes, the coordinates and the attributes of every dataset (the values
of OBSERVER_POSITION, computed with libm trigonometry, to a relative 1e-12:
see reference.APPROX_VARIABLES).

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

Skipped where python-casacore is installed (the Linux and macOS workflows).
"""

import shutil

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
