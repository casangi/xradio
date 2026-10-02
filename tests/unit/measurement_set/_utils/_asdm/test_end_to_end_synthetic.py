"""
End-to-end ORACLE tests of the ``xradio_asdm`` backend on synthetic ASDMs.

The ASDMs are written to disk by ``synthetic_asdm.py`` (real pyasdm tables and
real MIME BDFs whose values encode their position), see the ``synth_*``
fixtures in ``conftest.py``. Every expectation is computed from the writer's
own parameters (``synthetic_asdm.expected_*``), independently of the xradio
code, and follows the shared contracts (K1..K13) of the review fixes for
casangi/xradio PR #614. Finding IDs (F01..F99) refer to that review.
"""

import importlib.util
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.time import Time

from xradio.schema.check import check_datatree


def _load_synthetic_asdm():
    """Load synthetic_asdm.py (same module object as the conftest fixture)."""
    name = "xradio_tests_asdm_synthetic_asdm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("synthetic_asdm.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


synth = _load_synthetic_asdm()

ENGINE = "xradio_asdm"
MSV4_TYPES = {"visibility", "spectrum", "radiometer"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def open_ps(path, **kwargs) -> xr.DataTree:
    """
    Open with the public xarray path. ``cache=False`` so that every access reaches
    the backend arrays (with the default in-memory cache, a full load done by
    one test would make later lazy selections read the cached array).
    """
    kwargs.setdefault("cache", False)
    return xr.open_datatree(path, engine=ENGINE, **kwargs)


def msv4_nodes(ps: xr.DataTree) -> list[xr.DataTree]:
    return list(ps.children.values())


def correlated_data_name(ds: xr.Dataset) -> str:
    return ds.attrs["data_groups"]["base"]["correlated_data"]


def node_for_spw(ps: xr.DataTree, truth, spw_id: int) -> xr.DataTree:
    """The single MSv4 node of the processing set holding this SPW."""
    found = []
    for node in msv4_nodes(ps):
        try:
            if truth.spw_id_from_frequency(node.ds.frequency.values) == spw_id:
                found.append(node)
        except AssertionError:
            continue
    assert len(found) == 1, f"Expected one node for SPW {spw_id}, found {len(found)}"
    return found[0]


def matched_nodes(ps: xr.DataTree, truth, partitions) -> list:
    """[(node, expected_partition)] for every node, checking the node count."""
    nodes = msv4_nodes(ps)
    assert len(nodes) == len(partitions), (
        f"{len(nodes)} MSv4 nodes, {len(partitions)} expected partitions: "
        + str([(p.spw_id, p.obs_modes, p.field_ids) for p in partitions])
    )
    pairs = [
        (node, synth.match_partition(truth, node.ds, partitions)) for node in nodes
    ]
    assert len({id(part) for _node, part in pairs}) == len(pairs), (
        "Several nodes match the same expected partition"
    )
    return pairs


def decode_time(coord: xr.DataArray) -> Time:
    """astropy Time from a time measure, using its own format/scale attrs (K1)."""
    return Time(
        np.asarray(coord.values, dtype=float),
        format=coord.attrs["format"],
        scale=coord.attrs["scale"],
    )


def assert_true_utc(coord: xr.DataArray, integrations, atol_s: float = 2e-6):
    decoded = decode_time(coord)
    expected = Time(synth.expected_datetimes(integrations), scale="utc")
    assert decoded.shape == expected.shape
    diff = (decoded - expected).to_value("s")
    assert np.max(np.abs(diff)) < atol_s, (
        f"max |time - truth| = {np.max(np.abs(diff))} s"
    )


def assert_true_utc_unix(coord: xr.DataArray, unix_utc: np.ndarray, atol_s=1e-5):
    """The time measure decodes (own attrs) to the given UTC unix instants."""
    decoded = decode_time(coord)
    expected = Time(np.asarray(unix_utc, dtype=float), format="unix", scale="utc")
    diff = (decoded - expected).to_value("s")
    assert np.max(np.abs(diff)) < atol_s, (
        f"max |time - truth| = {np.max(np.abs(diff))} s"
    )


def numpy_key(dims, isel: dict) -> tuple:
    return tuple(isel.get(dim, slice(None)) for dim in dims)


def assert_values(actual, expected, what=""):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, (
        f"{what}: shape {actual.shape} != {expected.shape}"
    )
    if expected.dtype == bool:
        assert actual.dtype == bool, f"{what}: dtype {actual.dtype} is not bool"
        np.testing.assert_array_equal(actual, expected, err_msg=what)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, err_msg=what)


# ---------------------------------------------------------------------------
# opened processing sets (module scope: each ASDM is opened once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ps_interferometric(synth_interferometric):
    return open_ps(synth_interferometric.path)


@pytest.fixture(scope="module")
def ps_full_pol(synth_full_pol):
    return open_ps(synth_full_pol.path)


@pytest.fixture(scope="module")
def ps_single_dish(synth_single_dish):
    return open_ps(synth_single_dish.path)


@pytest.fixture(scope="module")
def ps_single_dish_simple(synth_single_dish_simple):
    return open_ps(synth_single_dish_simple.path)


@pytest.fixture(scope="module")
def ps_mosaic(synth_mosaic):
    return open_ps(synth_mosaic.path)


@pytest.fixture(scope="module")
def ps_mosaic_no_field_axis(synth_mosaic):
    return open_ps(synth_mosaic.path, partition_scheme=[])


@pytest.fixture(scope="module")
def ps_interleaved(synth_interleaved):
    return open_ps(synth_interleaved.path)


@pytest.fixture(scope="module")
def ps_interleaved_ch_avg(synth_interleaved):
    return open_ps(
        synth_interleaved.path,
        include_spectral_resolution_types=["CHANNEL_AVERAGE"],
    )


@pytest.fixture(scope="module")
def ps_interleaved_radiometer(synth_interleaved):
    return open_ps(
        synth_interleaved.path,
        include_processor_types=["RADIOMETER"],
        include_spectral_resolution_types=["FULL_RESOLUTION", "BASEBAND_WIDE"],
    )


@pytest.fixture(scope="module")
def dtype_variants(make_synthetic_asdm):
    """Tiny ASDMs with FLOAT32 / INT32 / INT16 crossData, 2 two-APC layouts and
    a single APC that is not AP_UNCORRECTED."""
    variants = {
        "float32": synth.small_dtype_spec(
            "FLOAT32_TYPE", 1.0, name="uid___A002_X77_X1"
        ),
        "int32_scaled": synth.small_dtype_spec(
            "INT32_TYPE", 0.5, name="uid___A002_X77_X2"
        ),
        "int16_scaled": synth.small_dtype_spec(
            "INT16_TYPE", 4.0, name="uid___A002_X77_X3"
        ),
        "two_apc": synth.small_dtype_spec(
            "INT32_TYPE",
            2.0,
            apc=("AP_UNCORRECTED", "AP_CORRECTED"),
            name="uid___A002_X77_X4",
        ),
        "two_apc_reversed": synth.small_dtype_spec(
            "FLOAT32_TYPE",
            1.0,
            apc=("AP_CORRECTED", "AP_UNCORRECTED"),
            name="uid___A002_X77_X5",
        ),
        "only_ap_corrected": synth.small_dtype_spec(
            "FLOAT32_TYPE",
            1.0,
            apc=("AP_CORRECTED",),
            name="uid___A002_X77_X6",
        ),
    }
    return {key: make_synthetic_asdm(spec) for key, spec in variants.items()}


# (fixture with the truth, fixture with the opened ps, open kwargs for expected_partitions)
VALUE_CASES = {
    "interferometric": ("synth_interferometric", "ps_interferometric", {}),
    "full_pol": ("synth_full_pol", "ps_full_pol", {}),
    "mosaic": ("synth_mosaic", "ps_mosaic", {}),
    "mosaic_no_field_axis": (
        "synth_mosaic",
        "ps_mosaic_no_field_axis",
        {"partition_scheme": []},
    ),
    "interleaved_full_res": ("synth_interleaved", "ps_interleaved", {}),
    "interleaved_ch_avg": (
        "synth_interleaved",
        "ps_interleaved_ch_avg",
        {"include_spectral_resolution_types": ["CHANNEL_AVERAGE"]},
    ),
}


def _case(request, case):
    truth_fixture, ps_fixture, kwargs = VALUE_CASES[case]
    truth = request.getfixturevalue(truth_fixture)
    ps = request.getfixturevalue(ps_fixture)
    return truth, ps, truth.expected_partitions(**kwargs)


# ---------------------------------------------------------------------------
# The synthetic writer itself (these pass independently of the backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "truth_fixture",
    [
        "synth_interferometric",
        "synth_interferometric_pointing",
        "synth_full_pol",
        "synth_single_dish",
        "synth_single_dish_simple",
        "synth_mosaic",
        "synth_interleaved",
    ],
)
def test_synthetic_asdm_roundtrips_with_pyasdm(request, truth_fixture):
    """The synthetic tables and BDFs are read back by pyasdm (ASDM.setFromFile,
    BDFReader.getSubset) with exactly the values written (F47: real BDF I/O)."""
    truth = request.getfixturevalue(truth_fixture)
    synth.verify_bdf_roundtrip(truth)


def test_synthetic_dtype_variants_roundtrip(dtype_variants):
    """INT16/INT32 (scaled) and 2-APC BDFs round-trip through pyasdm (F47)."""
    for truth in dtype_variants.values():
        synth.verify_bdf_roundtrip(truth)


def _pointing_row_xml(use_polynomials: str, pointing_tracking: str = "true") -> str:
    angles = "2 1 2 -1.46 1.14"
    return f"""<row><antennaId> Antenna_0 </antennaId>
<timeInterval> 5137194870768000000 24240000000 </timeInterval><numSample> 1 </numSample>
<encoder> {angles} </encoder><pointingTracking> {pointing_tracking} </pointingTracking>
<usePolynomials> {use_polynomials} </usePolynomials>
<timeOrigin> 5137194858648000000 </timeOrigin><numTerm> 1 </numTerm>
<pointingDirection> {angles} </pointingDirection><target> {angles} </target>
<offset> {angles} </offset><pointingModelId> 0 </pointingModelId></row>"""


@pytest.mark.parametrize(
    "use_polynomials, pointing_tracking, expected",
    [
        ("false", "true", (False, True)),
        (" False ", "TRUE", (False, True)),
        ("true", "false", (True, False)),
        ("True", "False", (True, False)),
    ],
)
def test_set_row_from_xml_sets_boolean_attributes(
    use_polynomials, pointing_tracking, expected
):
    """pyasdm parses XML booleans with bool(text) ("false" -> True);
    set_row_from_xml gives the rows the boolean values written in the XML (F77)."""
    import pyasdm

    row = pyasdm.PointingRow(pyasdm.ASDM().getPointing())
    xml = _pointing_row_xml(use_polynomials, pointing_tracking)
    assert synth.set_row_from_xml(row, xml) is row
    assert (row.getUsePolynomials(), row.getPointingTracking()) == expected
    assert row.getNumSample() == 1


def test_set_row_from_xml_leaves_non_boolean_attributes():
    """Only attributes that parsed as bool are corrected: a string attribute whose
    text reads "true"/"false" keeps its (string) value."""

    class _Row:
        def setFromXML(self, xml):
            self._flag, self._name = True, "false"  # pyasdm-style parse

        def getFlag(self):
            return self._flag

        def setFlag(self, value):
            self._flag = bool(value)

        def getName(self):
            return self._name

        def setName(self, value):
            raise AssertionError("string attribute overwritten")

    row = synth.set_row_from_xml(
        _Row(), "<row><flag> false </flag><name> false </name></row>"
    )
    assert row.getFlag() is False
    assert row.getName() == "false"


def test_synthetic_tables_carry_written_boolean_values(make_synthetic_asdm):
    """The synthetic ASDM rows hold the booleans of their XML (F77): Pointing
    usePolynomials=False / pointingTracking=True (also once read back from a
    binary Pointing table), ExecBlock aborted=False."""
    import contextlib

    import pyasdm

    spec = synth.interferometric_spec(with_pointing=True, name="uid___A002_X1234_X56b0")
    spec.pointing_as_bin = True
    asdm, _ = synth._build_tables(spec)
    pointing_rows = asdm.getPointing().get()
    assert len(pointing_rows) == spec.num_antenna * sum(
        len(scan.subscans) for scan in spec.scans
    )
    assert {row.getUsePolynomials() for row in pointing_rows} == {False}
    assert {row.getPointingTracking() for row in pointing_rows} == {True}
    assert [row.getAborted() for row in asdm.getExecBlock().get()] == [False]

    truth = make_synthetic_asdm(spec)
    assert os.path.isfile(os.path.join(truth.path, "Pointing.bin"))
    read_back = pyasdm.ASDM()
    with contextlib.redirect_stdout(io.StringIO()):
        read_back.setFromFile(truth.path)
    read_rows = read_back.getPointing().get()
    assert len(read_rows) == len(pointing_rows)
    assert {row.getUsePolynomials() for row in read_rows} == {False}
    assert {row.getPointingTracking() for row in read_rows} == {True}


@pytest.mark.parametrize(
    "truth_fixture", ["synth_interferometric", "synth_full_pol", "synth_mosaic"]
)
def test_expected_cross_values_match_bdf_bytes(request, truth_fixture):
    """The expected-value functions agree with an independent decoding of the
    raw crossData read by pyasdm (guards the oracle itself)."""
    import pyasdm

    truth = request.getfixturevalue(truth_fixture)
    nbl = len(truth.cross_baselines)
    for spw in truth.spws:
        refs = truth.integrations(spw.spw_id)
        expected = synth.expected_visibility(truth, spw.spw_id, refs)
        for tidx, ref in enumerate(refs):
            reader = pyasdm.bdf.BDFReader()
            reader.open(ref.bdf.path)
            try:
                for _ in range(ref.index + 1):
                    subset = reader.getSubset()
            finally:
                reader.close()
            decoded = synth.decode_raw_cross_subset(
                truth, ref.bdf, subset["crossData"]["arr"], spw.spw_id
            )
            np.testing.assert_array_equal(decoded, expected[tidx, :nbl])


# ---------------------------------------------------------------------------
# Engine detection (F27)
# ---------------------------------------------------------------------------


def test_guess_can_open(synth_interferometric, tmp_path):
    """guess_can_open is True for an ASDM directory (str or Path) and False for
    other directories, files and non-path objects; engine auto-detection picks
    xradio_asdm for an ASDM directory (F27)."""
    entrypoint = xr.backends.list_engines()[ENGINE]
    path = synth_interferometric.path
    assert entrypoint.guess_can_open(path) is True
    assert entrypoint.guess_can_open(Path(path)) is True
    assert entrypoint.guess_can_open(str(tmp_path)) is False
    assert entrypoint.guess_can_open(os.path.join(path, "ASDM.xml")) is False
    assert entrypoint.guess_can_open(os.path.join(str(tmp_path), "missing")) is False
    assert entrypoint.guess_can_open(io.BytesIO(b"")) is False
    assert xr.backends.plugins.guess_engine(path) == ENGINE


# ---------------------------------------------------------------------------
# VISIBILITY / FLAG values (F01, F03, F04, F11, F13, F14, F15, F18, F46, K2, K3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(VALUE_CASES))
def test_visibility_values_full_load(request, case):
    """VISIBILITY of every partition equals the values written to the BDFs, for
    cross and auto baselines and every SPW of a multi-SPW / multi-baseband BDF,
    including 1-channel SPWs, full-pol data, int32 data with scale factor,
    interleaved DD tables; dtype complex64 (F03, F13, F18, F46, F66, F20, K3)."""
    truth, ps, partitions = _case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        ds = node.ds
        assert correlated_data_name(ds) == "VISIBILITY"
        values = ds.VISIBILITY.values
        expected = synth.expected_visibility(truth, part.spw_id, part.integrations)
        assert_values(values, expected, f"VISIBILITY spw {part.spw_id}")
        assert values.dtype == np.complex64
        assert ds.VISIBILITY.dtype == np.complex64


@pytest.mark.parametrize("case", list(VALUE_CASES))
def test_flag_values_full_load(request, case):
    """FLAG is boolean (lazily declared and loaded) and flagged == (BDF flag word
    != 0) at the right (time, baseline, polarization), for every SPW including
    full-pol multi-baseband data and uneven SPW counts per baseband (F01, F14,
    F15, K2)."""
    truth, ps, partitions = _case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        ds = node.ds
        assert ds.FLAG.dtype == bool
        expected = synth.expected_flags(truth, part.spw_id, part.integrations)
        assert expected.any() and not expected.all()
        assert_values(ds.FLAG.values, expected, f"FLAG spw {part.spw_id}")


@pytest.mark.parametrize(
    "variant",
    [
        "float32",
        "int32_scaled",
        "int16_scaled",
        "two_apc",
        "two_apc_reversed",
        "only_ap_corrected",
    ],
)
def test_visibility_dtype_and_apc_variants(dtype_variants, variant, caplog):
    """FLOAT32 data are used as is, INT32/INT16 data are divided by the BDF
    scaleFactor, the result is complex64; with two APC values the
    AP_UNCORRECTED data are loaded, whatever their position; a single APC is
    loaded whatever it is, with one warning when the partition is opened when it
    is not AP_UNCORRECTED (K3, K6, F11)."""
    truth = dtype_variants[variant]
    with caplog.at_level(logging.WARNING):
        ps = open_ps(truth.path)
    apc_warnings = [
        rec
        for rec in caplog.records
        if "only the atmospheric phase correction" in rec.getMessage()
    ]
    assert len(apc_warnings) == (1 if variant == "only_ap_corrected" else 0)
    (node,) = msv4_nodes(ps)
    cfg = truth.config(truth.spws[0].config_idx)
    loaded_apc = cfg.apc[cfg.loaded_apc_index]
    assert loaded_apc in node.ds.attrs["data_groups"]["base"]["description"]
    refs = truth.integrations(truth.spws[0].spw_id)
    vis = node.ds.VISIBILITY.values
    assert_values(vis, synth.expected_visibility(truth, truth.spws[0].spw_id, refs))
    assert vis.dtype == np.complex64
    assert_values(
        node.ds.FLAG.values, synth.expected_flags(truth, truth.spws[0].spw_id, refs)
    )


def test_num_bin_greater_than_one_is_rejected(make_synthetic_asdm):
    """numBin > 1 is not supported and must raise NotImplementedError (or, if
    detected when opening, no partition opens -> RuntimeError), never return
    silently wrong data (K6, F11, K13)."""
    truth = make_synthetic_asdm(
        synth.small_dtype_spec("FLOAT32_TYPE", 1.0, num_bin=2, name="uid___A002_X77_X9")
    )
    try:
        ps = open_ps(truth.path)
    except (RuntimeError, NotImplementedError):
        return
    for node in msv4_nodes(ps):
        with pytest.raises(NotImplementedError):
            node.ds.VISIBILITY.values  # noqa: B018


@pytest.mark.parametrize(
    "spw_id, freq_key",
    [
        (0, slice(3, 6)),
        (0, slice(5, None)),
        (0, 6),
        (0, slice(0, 2)),
        (0, slice(1, 8, 3)),
        (2, slice(1, 4)),
        (2, slice(2, None)),
        (2, 3),
        (2, slice(0, 2)),
    ],
)
def test_visibility_frequency_selection(
    ps_interferometric, synth_interferometric, spw_id, freq_key
):
    """Channel selections starting at a nonzero channel return those channels,
    for cross and auto baselines, in SPWs at different BDF positions (F04,
    F21)."""
    truth = synth_interferometric
    ds = node_for_spw(ps_interferometric, truth, spw_id).ds
    expected = synth.expected_visibility(truth, spw_id, truth.integrations(spw_id))
    assert_values(
        ds.VISIBILITY.isel(frequency=freq_key).values,
        expected[:, :, freq_key],
        f"VISIBILITY frequency={freq_key}",
    )


def test_full_pol_autocorrelations(ps_full_pol, synth_full_pol):
    """CROSS_AND_AUTO full-pol autos: XY is complex Re(XY) + i Im(XY) and YX =
    conj(XY); XX and YY are real (F18)."""
    truth = synth_full_pol
    nbl = len(truth.cross_baselines)
    for node in msv4_nodes(ps_full_pol):
        ds = node.ds
        assert list(ds.polarization.values) == ["XX", "XY", "YX", "YY"]
        autos = ds.VISIBILITY.isel(baseline_id=slice(nbl, None)).values
        assert np.all(autos[..., 1].imag != 0)
        np.testing.assert_array_equal(autos[..., 2], np.conj(autos[..., 1]))
        np.testing.assert_array_equal(autos[..., 0].imag, 0)
        np.testing.assert_array_equal(autos[..., 3].imag, 0)


# ---------------------------------------------------------------------------
# Lazy indexing vs eager (F05, F10, F16, F21, F22, F56, K4)
# ---------------------------------------------------------------------------

N_CROSS = 6  # synth_interferometric: 4 antennas -> 6 cross + 4 auto baselines

LAZY_KEYS = {
    "time_int_first": {"time": 0},
    "time_int_last": {"time": -1},
    "time_int_in_2nd_bdf": {"time": 4},
    "time_step2": {"time": slice(None, None, 2)},
    "time_step3_from1": {"time": slice(1, None, 3)},
    "time_reversed": {"time": slice(None, None, -1)},
    "time_neg_step": {"time": slice(8, 1, -3)},
    "time_across_bdfs": {"time": slice(2, 6)},
    "time_empty": {"time": slice(4, 4)},
    "time_list": {"time": [7, 0, 3]},
    "bl_int_cross": {"baseline_id": 2},
    "bl_int_auto": {"baseline_id": N_CROSS + 1},
    "bl_cross_only_to_boundary": {"baseline_id": slice(0, N_CROSS)},
    "bl_cross_tail_to_boundary": {"baseline_id": slice(4, N_CROSS)},
    "bl_autos_only": {"baseline_id": slice(N_CROSS, None)},
    "bl_across_boundary": {"baseline_id": slice(N_CROSS - 1, N_CROSS + 2)},
    "bl_step": {"baseline_id": slice(1, None, 3)},
    "bl_reversed_step": {"baseline_id": slice(None, None, -2)},
    "bl_empty": {"baseline_id": slice(N_CROSS, N_CROSS)},
    "freq_int_first": {"frequency": 0},
    "freq_int": {"frequency": 5},
    "freq_from0": {"frequency": slice(0, 4)},
    "freq_mid": {"frequency": slice(3, 6)},
    "freq_step_from0": {"frequency": slice(0, None, 3)},
    "freq_neg_step": {"frequency": slice(6, 0, -2)},
    "freq_empty": {"frequency": slice(2, 2)},
    "pol_int_first": {"polarization": 0},
    "pol_int_second": {"polarization": 1},
    "pol_slice": {"polarization": slice(1, 2)},
    "uvw_label_int": {"uvw_label": 2},
    "uvw_label_slice": {"uvw_label": slice(0, 2)},
    "ints_time_bl_pol": {"time": 3, "baseline_id": 2, "polarization": 0},
    "mixed": {
        "time": slice(1, 9, 2),
        "baseline_id": slice(3, 9),
        "frequency": slice(2, 7, 2),
        "polarization": -1,
    },
    "auto_int_freq_int": {"time": slice(5, None), "baseline_id": 8, "frequency": 3},
    "uvw_time_int_label_int": {"time": 6, "uvw_label": 1},
}

VAR_DIMS = {
    "VISIBILITY": ("time", "baseline_id", "frequency", "polarization"),
    "FLAG": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "frequency", "polarization"),
    "UVW": ("time", "baseline_id", "uvw_label"),
}

LAZY_CASES = [
    pytest.param(var, key_name, id=f"{var}-{key_name}")
    for var, dims in VAR_DIMS.items()
    for key_name, key in LAZY_KEYS.items()
    if set(key) <= set(dims)
]


@pytest.fixture(scope="module")
def lazy_reference(ps_interferometric, synth_interferometric):
    """Node of SPW 0 (8 channels, never fully loaded) and the eager full load of
    every variable from a separately opened tree. Lazy selections are compared
    with the eager arrays, so these tests isolate indexing from decoding (the
    values themselves are checked against the truth by the *_full_load
    tests)."""
    truth = synth_interferometric
    node = node_for_spw(ps_interferometric, truth, 0)
    eager = node_for_spw(open_ps(truth.path), truth, 0).ds
    reference = {var: np.asarray(eager[var].values) for var in VAR_DIMS}
    for var in VAR_DIMS:
        assert reference[var].shape == eager[var].shape
    assert reference["FLAG"].dtype == bool
    return node.ds, reference


@pytest.mark.parametrize("var, key_name", LAZY_CASES)
def test_lazy_isel_matches_eager(lazy_reference, var, key_name):
    """isel with int keys, steps, negative steps, empty slices, lists and slices
    ending exactly at the cross/auto boundary returns the same values, shape
    and dtype as indexing the eagerly loaded full array in numpy (F05, F10,
    F16, F21, F22, F56, K4)."""
    ds, reference = lazy_reference
    isel = LAZY_KEYS[key_name]
    expected = reference[var][numpy_key(VAR_DIMS[var], isel)]
    result = ds[var].isel(isel)
    assert result.shape == expected.shape
    values = result.values
    assert_values(values, expected, f"{var}.isel({isel})")
    assert values.dtype == expected.dtype


CHUNK_CASES = [
    pytest.param(
        var, chunks, id=f"{var}-" + "-".join(f"{k}{v}" for k, v in chunks.items())
    )
    for var, dims in VAR_DIMS.items()
    for chunks in [
        {"time": 3},
        {"baseline_id": 3},
        {"frequency": 3},
        {"polarization": 1},
        {"uvw_label": 1},
        {"time": 4, "baseline_id": 4, "frequency": 5},
    ]
    if set(chunks) <= set(dims)
]


@pytest.mark.parametrize("var, chunks", CHUNK_CASES)
def test_dask_chunks_match_eager(lazy_reference, var, chunks):
    """Dask chunking along each dimension (chunk boundaries inside BDFs, at the
    cross/auto boundary, at nonzero channels) gives the full array (F04, F05,
    F10, F21, F22)."""
    ds, reference = lazy_reference
    chunked = ds[var].chunk(chunks)
    assert_values(chunked.values, reference[var], f"{var} chunks={chunks}")


# time chunks of 4 deliberately do not align with the BDFs (preferred time chunks),
# so that chunks spanning BDF boundaries are checked. xarray warns about that.
@pytest.mark.filterwarnings(
    "ignore:The specified chunks separate the stored chunks:UserWarning"
)
def test_open_with_chunks(synth_interferometric):
    """Opening with dask chunks gives the true VISIBILITY / FLAG values (F04,
    F05, F21), also with time chunks that span BDF boundaries."""
    truth = synth_interferometric
    ps = open_ps(truth.path, chunks={"time": 4, "frequency": 3, "baseline_id": 3})
    for spw in truth.spws:
        ds = node_for_spw(ps, truth, spw.spw_id).ds
        refs = truth.integrations(spw.spw_id)
        assert ds.VISIBILITY.chunks is not None
        assert_values(
            ds.VISIBILITY.values, synth.expected_visibility(truth, spw.spw_id, refs)
        )
        assert_values(ds.FLAG.values, synth.expected_flags(truth, spw.spw_id, refs))


def test_weight_materialises_only_the_selection(peak_memory):
    """Reading one WEIGHT element must not allocate the whole partition (F06, K3:
    WEIGHT float64, only the selected block is materialised)."""
    from xradio.measurement_set._utils._asdm import asdm_backend_arrays

    # 3.2 PB of float64 if allocated in full: always refused (address space)
    shape = (100_000, 100_000, 10_000, 4)
    var = xr.Variable(
        ("time", "baseline_id", "frequency", "polarization"),
        xr.core.indexing.LazilyIndexedArray(asdm_backend_arrays.WeightArray(shape)),
    )
    assert var.dtype == np.float64
    block, peak = peak_memory(lambda: var[5, 10:12, 100:103, :].values)
    assert block.shape == (2, 3, 4)
    assert block.dtype == np.float64
    assert peak < 10 * 1024**2


def test_lazy_data_after_chdir(synth_interferometric, monkeypatch, tmp_path):
    """An ASDM opened (open_asdm) with a relative path still loads after the
    working directory changes (F78)."""
    from xradio.measurement_set._utils._asdm.open_asdm import open_asdm

    truth = synth_interferometric
    parent, name = os.path.split(truth.path)
    monkeypatch.chdir(parent)
    ps = open_asdm(name)
    monkeypatch.chdir(tmp_path)
    ds = node_for_spw(ps, truth, 0).ds
    assert_values(
        ds.VISIBILITY.values,
        synth.expected_visibility(truth, 0, truth.integrations(0)),
    )


# ---------------------------------------------------------------------------
# Time (F02, F07, F19, F25, F45, K1, K5)
# ---------------------------------------------------------------------------


TIME_CASES = list(VALUE_CASES) + ["single_dish"]


def _time_case(request, case):
    if case == "single_dish":
        truth = request.getfixturevalue("synth_single_dish_simple")
        return (
            truth,
            request.getfixturevalue("ps_single_dish_simple"),
            truth.expected_partitions(),
        )
    return _case(request, case)


@pytest.mark.parametrize("case", TIME_CASES)
def test_time_coordinate_is_true_utc(request, case):
    """The time coordinate, decoded with astropy using its own format/scale
    attributes, gives the true UTC instants of the integrations, is strictly
    increasing, and is labelled with the ASDM time constants (F02, F07, F25,
    K1)."""
    from xradio.measurement_set._utils._asdm._utils.time import (
        ASDM_TIME_FORMAT,
        ASDM_TIME_SCALE,
    )

    truth, ps, partitions = _time_case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        time = node.ds.time
        assert np.all(np.diff(time.values) > 0), "time is not strictly increasing"
        assert_true_utc(time, part.integrations)
        np.testing.assert_allclose(time.values, part.unix_times, rtol=0, atol=1e-6)
        assert time.attrs["format"] == ASDM_TIME_FORMAT
        assert time.attrs["scale"] == ASDM_TIME_SCALE


@pytest.mark.parametrize("case", TIME_CASES)
def test_time_centroid_and_effective_integration_time(request, case):
    """TIME_CENTROID equals the time coordinate broadcast over baselines (or
    antennas; same measure attrs), EFFECTIVE_INTEGRATION_TIME equals the
    integration duration, and time.attrs["integration_time"] is the duration
    (F02, F45, K1). Both variables are optional in SpectrumXds, so they are
    only checked there when present."""
    truth, ps, partitions = _time_case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        ds = node.ds
        intervals = synth.expected_intervals(part.integrations)
        assert ds.time.attrs["integration_time"]["data"] == pytest.approx(intervals[0])
        if ds.attrs["type"] == "spectrum" and "TIME_CENTROID" not in ds:
            continue
        second_dim = ds.TIME_CENTROID.dims[1]
        ntime, nrow = ds.sizes["time"], ds.sizes[second_dim]
        expected_tc = np.broadcast_to(part.unix_times[:, None], (ntime, nrow))
        np.testing.assert_allclose(
            ds.TIME_CENTROID.values, expected_tc, rtol=0, atol=1e-6
        )
        for attr in ("format", "scale", "units"):
            assert ds.TIME_CENTROID.attrs[attr] == ds.time.attrs[attr]
        if ds.attrs["type"] == "spectrum" and "EFFECTIVE_INTEGRATION_TIME" not in ds:
            continue
        np.testing.assert_allclose(
            ds.EFFECTIVE_INTEGRATION_TIME.values,
            np.broadcast_to(intervals[:, None], (ntime, nrow)),
            rtol=1e-9,
        )


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize(
    "fixture_name", ["synth_interferometric", "synth_single_dish_simple"]
)
def test_time_variables_are_writeable(request, fixture_name, cache):
    """Opened without chunks (the default), TIME_CENTROID and
    EFFECTIVE_INTEGRATION_TIME are writeable like the other data variables:
    in-place arithmetic, writing to .values and item assignment after load()
    work (they were read-only broadcast views)."""
    truth = request.getfixturevalue(fixture_name)
    ps = open_ps(truth.path, cache=cache)
    for node in msv4_nodes(ps):
        ds = node.to_dataset()
        names = ["TIME_CENTROID", "EFFECTIVE_INTEGRATION_TIME", "FLAG"]
        names.append("SPECTRUM" if ds.attrs["type"] == "spectrum" else "VISIBILITY")
        for name in names:
            assert ds[name].values.flags.writeable, name
        expected = ds["TIME_CENTROID"].values.copy()
        ds["TIME_CENTROID"] += 2.0
        expected += 2.0
        np.testing.assert_array_equal(ds["TIME_CENTROID"].values, expected)
        loaded = ds.load()
        loaded["EFFECTIVE_INTEGRATION_TIME"][0, 0] = 0.0
        loaded["TIME_CENTROID"].values[-1, -1] = 1.0
        assert loaded["EFFECTIVE_INTEGRATION_TIME"].values[0, 0] == 0.0
        assert loaded["TIME_CENTROID"].values[-1, -1] == 1.0
        # the other elements of the row are unchanged (no shared memory)
        np.testing.assert_array_equal(
            loaded["TIME_CENTROID"].values[-1, :-1], expected[-1, :-1]
        )


def test_interleaved_configs_give_monotonic_time(ps_interleaved, synth_interleaved):
    """With interleaved WVR/SQLD/CHANNEL_AVERAGE/FULL_RESOLUTION configs in Main
    (default open), every partition's BDFs are in time order (F07, K7 BDFPath
    ordered by Main time)."""
    truth = synth_interleaved
    partitions = truth.expected_partitions()
    assert len(partitions) == 4
    for node in msv4_nodes(ps_interleaved):
        spw_id = truth.spw_id_from_frequency(node.ds.frequency.values)
        refs = truth.integrations(spw_id)
        times = node.ds.time.values
        assert np.all(np.diff(times) > 0), f"non-monotonic time for SPW {spw_id}"
        np.testing.assert_allclose(times, synth.expected_times(refs), rtol=0, atol=1e-6)


def test_packed_radiometer_partitions(ps_interleaved_radiometer, synth_interleaved):
    """RADIOMETER partitions are typed "radiometer". A packed WVR BDF
    (dimensionality 0, numTime=N) contributes N integrations with their own
    times and data (F19, F79, K5, K12)."""
    truth = synth_interleaved
    partitions = truth.expected_partitions(
        include_processor_types=["RADIOMETER"],
        include_spectral_resolution_types=["FULL_RESOLUTION", "BASEBAND_WIDE"],
    )
    for node, part in matched_nodes(ps_interleaved_radiometer, truth, partitions):
        ds = node.ds
        assert ds.attrs["type"] == "radiometer"
        assert ds.sizes["time"] == len(part.integrations)
        np.testing.assert_allclose(ds.time.values, part.unix_times, rtol=0, atol=1e-6)
        data = ds[correlated_data_name(ds)].values
        expected = synth.expected_visibility(truth, part.spw_id, part.integrations)
        assert_values(np.real(data), expected.real, f"radiometer spw {part.spw_id}")
        if np.iscomplexobj(data):
            np.testing.assert_array_equal(np.imag(data), 0)
    wvr_spw = truth.spws[0]
    assert wvr_spw.pol.sd == ("I",)
    wvr = node_for_spw(ps_interleaved_radiometer, truth, wvr_spw.spw_id)
    assert wvr.ds.sizes["time"] == 4 * len(truth.spec.scans[0].subscans)


# ---------------------------------------------------------------------------
# Per-time metadata: scan_name, field_name (F29, F31, F43, K8, K9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["interferometric", "mosaic", "mosaic_no_field_axis"])
def test_scan_and_field_name_per_time(request, case):
    """scan_name and field_name follow each integration's own scan and field
    (no cycling), and field names are unique "<fieldName>_<fieldId>" (F29,
    F43, K8)."""
    truth, ps, partitions = _case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        ds = node.ds
        expected_scans = [str(ref.bdf.scan_number) for ref in part.integrations]
        expected_fields = [
            truth.make_field_name(ref.bdf.field_id) for ref in part.integrations
        ]
        assert list(map(str, ds.scan_name.values)) == expected_scans
        assert list(map(str, ds.field_name.values)) == expected_fields


def test_multi_field_partition_field_and_source(ps_mosaic_no_field_axis, synth_mosaic):
    """With partition_scheme=[] a mosaic scan gives one partition with several
    fields: field_and_source has one entry per field (ascending fieldId) with
    FIELD_PHASE_CENTER_DIRECTION from Field.phaseDir (F30, F31, K9)."""
    truth = synth_mosaic
    partitions = truth.expected_partitions(partition_scheme=[])
    multi = [part for part in partitions if len(part.field_ids) > 1]
    assert len(multi) == 1
    for node, part in matched_nodes(ps_mosaic_no_field_axis, truth, partitions):
        fsx = node["field_and_source_base_xds"].ds
        assert list(map(str, fsx.field_name.values)) == [
            truth.make_field_name(fid) for fid in part.field_ids
        ]
        expected_dirs = np.array(
            [truth.spec.fields[fid].phase_dir for fid in part.field_ids]
        )
        np.testing.assert_allclose(
            fsx.FIELD_PHASE_CENTER_DIRECTION.transpose(
                "field_name", "sky_dir_label"
            ).values,
            expected_dirs,
            rtol=0,
            atol=1e-12,
        )


def test_field_phase_center_from_phase_dir(ps_mosaic, synth_mosaic):
    """Interferometric FIELD_PHASE_CENTER_DIRECTION comes from Field.phaseDir,
    not Field.referenceDir (F30), with the Field direction frame (F69)."""
    truth = synth_mosaic
    field = truth.spec.fields[1]
    assert field.reference_dir != field.phase_dir
    for node, part in matched_nodes(ps_mosaic, truth, truth.expected_partitions()):
        fsx = node["field_and_source_base_xds"].ds
        (fid,) = part.field_ids
        assert list(map(str, fsx.field_name.values)) == [truth.make_field_name(fid)]
        direction = fsx.FIELD_PHASE_CENTER_DIRECTION
        np.testing.assert_allclose(
            direction.sel(field_name=truth.make_field_name(fid)).values,
            truth.spec.fields[fid].phase_dir,
            rtol=0,
            atol=1e-12,
        )
        assert direction.attrs["frame"].lower() == "icrs"


def test_combined_field_and_source_keeps_mosaic_pointings(ps_mosaic, synth_mosaic):
    """The combined field_and_source dataset keeps every mosaic pointing that
    shares a fieldName, so the mean phase center is right (F29)."""
    truth = synth_mosaic
    combined = ps_mosaic.xr_ps.get_combined_field_and_source_xds()
    names = sorted(map(str, combined.field_name.values))
    assert names == sorted(truth.make_field_name(fid) for fid in range(3))


def test_single_dish_reference_center_from_reference_dir(
    ps_single_dish_simple, synth_single_dish_simple
):
    """Single-dish field_and_source has FIELD_REFERENCE_CENTER_DIRECTION from
    Field.referenceDir (F30, K9)."""
    truth = synth_single_dish_simple
    field = truth.spec.fields[0]
    assert field.reference_dir != field.phase_dir
    for node in msv4_nodes(ps_single_dish_simple):
        fsx = node["field_and_source_base_xds"].ds
        assert "FIELD_REFERENCE_CENTER_DIRECTION" in fsx
        np.testing.assert_allclose(
            fsx.FIELD_REFERENCE_CENTER_DIRECTION.sel(
                field_name=truth.make_field_name(0)
            ).values,
            field.reference_dir,
            rtol=0,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Intents and partitioning (F34, F35, F44, K7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ["interferometric", "full_pol", "mosaic", "mosaic_no_field_axis"]
)
def test_scan_intents_attribute(request, case):
    """scan_name.attrs["scan_intents"] is a list with one "INTENT#SUBINTENT"
    string per intent of the partition (F34, F44, K7)."""
    truth, ps, partitions = _case(request, case)
    for node, part in matched_nodes(ps, truth, partitions):
        intents = node.ds.scan_name.attrs["scan_intents"]
        assert isinstance(intents, list)
        assert all(isinstance(intent, str) for intent in intents)
        assert sorted(intents) == list(part.obs_modes)


def test_query_by_scan_intent(ps_mosaic, synth_mosaic, ps_full_pol):
    """xr_ps.query(scan_intents="INTENT#SUBINTENT") with the default exact
    matching selects the partitions with that intent (F34, F44)."""
    partitions = synth_mosaic.expected_partitions()
    for mode in ("OBSERVE_TARGET#ON_SOURCE", "CALIBRATE_PHASE#ON_SOURCE"):
        selected = ps_mosaic.xr_ps.query(scan_intents=mode)
        expected = [part for part in partitions if mode in part.obs_modes]
        assert len(selected.children) == len(expected) > 0
    selected = ps_full_pol.xr_ps.query(scan_intents="CALIBRATE_WVR#ON_SOURCE")
    assert len(selected.children) == len(ps_full_pol.children)


def test_single_dish_subscan_intents_split_partitions(
    ps_single_dish, synth_single_dish
):
    """ON/OFF and HOT/AMBIENT subscans of one scan go to different partitions
    labelled with their subscan intent; query selects them (F34, K7)."""
    truth = synth_single_dish
    partitions = truth.expected_partitions()
    assert len(partitions) == 8  # 2 SPWs x {ON, OFF, HOT, AMBIENT}
    for node, part in matched_nodes(ps_single_dish, truth, partitions):
        assert sorted(node.ds.scan_name.attrs["scan_intents"]) == list(part.obs_modes)
    off = ps_single_dish.xr_ps.query(scan_intents="OBSERVE_TARGET#OFF_SOURCE")
    assert len(off.children) == 2
    for node in msv4_nodes(off):
        refs = truth.integrations(
            truth.spw_id_from_frequency(node.ds.frequency.values),
            obs_modes=("OBSERVE_TARGET#OFF_SOURCE",),
        )
        np.testing.assert_allclose(
            node.ds.time.values, synth.expected_times(refs), rtol=0, atol=1e-6
        )


@pytest.mark.parametrize("scheme", [["antennaId"], ["FIELD_ID"], ["fieldId", "foo"]])
def test_invalid_partition_scheme_raises(synth_interferometric, scheme):
    """Unsupported partition_scheme names raise a ValueError listing the
    allowed names (F35, K7)."""
    from xradio.measurement_set._utils._asdm.open_asdm import open_asdm

    with pytest.raises(ValueError, match="subscanNumber"):
        open_asdm(synth_interferometric.path, partition_scheme=scheme)


@pytest.mark.parametrize(
    "scheme", [["fieldId", "scanNumber"], ["subscanNumber"], ["scanNumber", "fieldId"]]
)
def test_partition_schemes(synth_mosaic, scheme):
    """Optional partition axes split the data as expected; every partition holds
    the right integrations (K7, F31)."""
    truth = synth_mosaic
    ps = open_ps(truth.path, partition_scheme=scheme)
    for node, part in matched_nodes(ps, truth, truth.expected_partitions(scheme)):
        np.testing.assert_allclose(
            node.ds.time.values, part.unix_times, rtol=0, atol=1e-6
        )


# ---------------------------------------------------------------------------
# UVW and data groups (F02, F22, F23, F30, F42, K10)
# ---------------------------------------------------------------------------


def test_base_data_group_has_uvw(ps_interferometric, ps_full_pol, ps_mosaic):
    """Interferometric partitions have UVW and data_groups["base"]["uvw"] ==
    "UVW" (F42)."""
    for ps in (ps_interferometric, ps_full_pol, ps_mosaic):
        for node in msv4_nodes(ps):
            ds = node.ds
            assert "UVW" in ds.data_vars
            assert ds.attrs["data_groups"]["base"]["uvw"] == "UVW"


@pytest.mark.parametrize(
    "case", ["interferometric", "mosaic", "mosaic_no_field_axis", "full_pol"]
)
def test_uvw_matches_independent_projection(request, case):
    """UVW is consistent with an independent projection of the ITRF baselines
    on the phase center of each integration's field at its true UTC time: |UVW|
    == |baseline|, w == +-(P1 - P2).s(t) within 1 cm on <= 1.4 km baselines,
    autos zero. Detects time errors of ~0.1 s or more, a referenceDir phase
    center and per-partition (instead of per-time) phase centers (F02, F23,
    F30, K1, K10)."""
    truth, ps, partitions = _case(request, case)
    baselines = synth.baseline_vectors_itrf(truth)
    for node, part in matched_nodes(ps, truth, partitions):
        uvw = node.ds.UVW.transpose("time", "baseline_id", "uvw_label").values
        assert uvw.shape == (len(part.integrations), len(baselines), 3)
        np.testing.assert_allclose(
            np.linalg.norm(uvw, axis=-1),
            np.broadcast_to(np.linalg.norm(baselines, axis=-1), uvw.shape[:2]),
            rtol=0,
            atol=1e-3,
        )
        directions = np.array(
            [truth.spec.fields[ref.bdf.field_id].phase_dir for ref in part.integrations]
        )
        unit = synth.phase_center_itrs_unit_vectors(part.unix_times, directions)
        w_expected = unit @ baselines.T
        w = uvw[..., 2]
        sign = np.sign(np.sum(w * w_expected))
        assert sign != 0
        np.testing.assert_allclose(w, sign * w_expected, rtol=0, atol=0.01)
        np.testing.assert_allclose(uvw[:, -truth.num_antenna :], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Single dish (F41, K12)
# ---------------------------------------------------------------------------


def test_single_dish_gives_spectrum_xds(
    ps_single_dish_simple, synth_single_dish_simple
):
    """An all-AUTO_ONLY ASDM gives SpectrumXds nodes: type "spectrum", dims
    (time, antenna_name, frequency, polarization), real float32 SPECTRUM with
    the auto-correlation values, boolean FLAG, no UVW, data group
    correlated_data "SPECTRUM" (F41, K12, K2, K3)."""
    truth = synth_single_dish_simple
    partitions = truth.expected_partitions()
    assert len(partitions) == 2
    for node, part in matched_nodes(ps_single_dish_simple, truth, partitions):
        ds = node.ds
        assert ds.attrs["type"] == "spectrum"
        assert correlated_data_name(ds) == "SPECTRUM"
        assert "UVW" not in ds.data_vars
        assert ds.attrs["data_groups"]["base"].get("uvw") is None
        dims = ("time", "antenna_name", "frequency", "polarization")
        for var in ("SPECTRUM", "FLAG", "WEIGHT"):
            assert ds[var].dims == dims
        assert list(map(str, ds.antenna_name.values)) == truth.antenna_names
        spectrum = ds.SPECTRUM.values
        assert_values(
            spectrum, synth.expected_spectrum(truth, part.spw_id, part.integrations)
        )
        assert spectrum.dtype == np.float32
        assert ds.SPECTRUM.dtype == np.float32
        assert_values(
            ds.FLAG.values, synth.expected_flags(truth, part.spw_id, part.integrations)
        )
        assert_values(
            ds.SPECTRUM.isel(antenna_name=1, frequency=slice(1, None)).values,
            synth.expected_spectrum(truth, part.spw_id, part.integrations)[:, 1, 1:],
        )


# ---------------------------------------------------------------------------
# Schema, accessors, I/O (F33, F39, F41, F81, K13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ps_fixture",
    [
        "ps_interferometric",
        "ps_full_pol",
        "ps_single_dish",
        "ps_single_dish_simple",
        "ps_mosaic",
        "ps_mosaic_no_field_axis",
        "ps_interleaved",
        "ps_interleaved_radiometer",
    ],
)
def test_schema_and_accessors(request, ps_fixture):
    """check_datatree reports no issues, every node is a typed MSv4, and the
    processing-set accessors work (summary, combined field_and_source,
    frequency axis) (F33, F39, F41, F81)."""
    ps = request.getfixturevalue(ps_fixture)
    assert ps.attrs["type"] == "processing_set"
    issues = check_datatree(ps)
    assert not issues, str(issues)
    nodes = msv4_nodes(ps)
    assert nodes
    for node in nodes:
        assert node.attrs["type"] in MSV4_TYPES
        data_var = correlated_data_name(node.ds)
        assert "encoding" not in node.ds[data_var].attrs
        assert "encoding" not in node.ds.FLAG.attrs
        assert isinstance(node.ds.attrs["observation_info"]["release_date"], str)
    summary = ps.xr_ps.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(nodes)
    ps.xr_ps.get_combined_field_and_source_xds()
    ps.xr_ps.get_freq_axis()


@pytest.mark.parametrize("truth_fixture", ["synth_interferometric", "synth_full_pol"])
def test_to_zarr_round_trip(request, truth_fixture, tmp_path):
    """The processing set can be written to zarr and read back with
    open_processing_set with identical data (incl. an ExecBlock releaseDate,
    stored as ISO string) (F33, K1, K2, K3)."""
    from xradio.measurement_set import open_processing_set

    truth = request.getfixturevalue(truth_fixture)
    ps = open_ps(truth.path)
    out = str(tmp_path / "ps.zarr")
    ps.to_zarr(out)
    ps_back = open_processing_set(out)
    assert sorted(ps_back.children) == sorted(ps.children)
    for name, node in ps.children.items():
        ds, back = node.ds, ps_back[name].ds
        spw_id = truth.spw_id_from_frequency(back.frequency.values)
        refs = truth.integrations(spw_id)
        assert_values(
            back.VISIBILITY.values, synth.expected_visibility(truth, spw_id, refs)
        )
        assert_values(back.FLAG.values, synth.expected_flags(truth, spw_id, refs))
        np.testing.assert_allclose(
            back.time.values, synth.expected_times(refs), atol=1e-6
        )
        assert back.VISIBILITY.dtype == ds.VISIBILITY.dtype == np.complex64
        release = back.attrs["observation_info"]["release_date"]
        assert isinstance(release, str)
        if truth.spec.release_date_ns is not None:
            assert release.startswith("2024-07-21T04:26:40")
    assert not check_datatree(ps_back)


def test_failed_partitions_are_skipped(make_synthetic_asdm):
    """A partition whose open fails is skipped (no empty node), the others are
    usable; if no partition can be opened, RuntimeError is raised (F39, F48,
    K13)."""
    from xradio.measurement_set._utils._asdm.open_asdm import open_asdm

    truth = make_synthetic_asdm(synth.mosaic_spec(name="uid___A002_X88_X1"))
    broken = [bdf for bdf in truth.bdfs if bdf.field_id == 2]
    assert len(broken) == 1
    os.remove(broken[0].path)
    ps = open_asdm(truth.path)
    expected = [part for part in truth.expected_partitions() if 2 not in part.field_ids]
    for node, part in matched_nodes(ps, truth, expected):
        assert node.attrs["type"] == "visibility"
        assert_values(
            node.ds.VISIBILITY.values,
            synth.expected_visibility(truth, part.spw_id, part.integrations),
        )
    assert not check_datatree(ps)
    assert len(ps.xr_ps.summary()) == len(expected)

    for bdf in truth.bdfs:
        if os.path.exists(bdf.path):
            os.remove(bdf.path)
    with pytest.raises(RuntimeError):
        open_asdm(truth.path)


# ---------------------------------------------------------------------------
# Pointing (F08, F24, F36, F37, F38, F40, K11)
# ---------------------------------------------------------------------------


def test_pointing_xds(synth_interferometric_pointing):
    """with_pointing=True gives each MSv4 a pointing_xds whose time_pointing
    decodes (own format/scale attrs) to the true sample times, is strictly
    increasing, covers the partition's time range, and has POINTING_BEAM ==
    target for zero offsets (F08, F24, F37, F38, F40, K1, K11)."""
    truth = synth_interferometric_pointing
    ps = open_ps(truth.path, with_pointing=True)
    sample_times = truth.pointing_times_unix
    for node in msv4_nodes(ps):
        assert "pointing_xds" in node.children
        pxds = node["pointing_xds"].ds
        tp = pxds.time_pointing
        values = np.asarray(tp.values, dtype=float)
        assert np.all(np.diff(values) > 0)
        decoded = decode_time(tp)
        expected = Time(sample_times, format="unix", scale="utc")
        idx = np.argmin(np.abs(sample_times[None, :] - values[:, None]), axis=1)
        assert np.max(np.abs((decoded - expected[idx]).to_value("s"))) < 1e-5
        spw_id = truth.spw_id_from_frequency(node.ds.frequency.values)
        part_times = synth.expected_times(truth.integrations(spw_id))
        inside = (sample_times >= part_times.min()) & (sample_times <= part_times.max())
        assert inside.any()
        assert np.isin(np.round(sample_times[inside], 6), np.round(values, 6)).all()
        assert list(map(str, pxds.antenna_name.values)) == truth.antenna_names
        beam = pxds.POINTING_BEAM.transpose(
            "time_pointing", "antenna_name", "local_sky_dir_label"
        ).values
        np.testing.assert_allclose(beam, truth.pointing_target[idx], rtol=0, atol=1e-9)


def test_create_pointing_xds_time_range(
    synth_interferometric_pointing, synth_interferometric
):
    """create_pointing_xds(asdm, time_range) restricts the pointing to the rows
    overlapping the range (unix seconds), returns None when no row is in range
    or the Pointing table is empty, and the default (no range) gives every
    sample (F08, F40, F76, K11)."""
    import contextlib

    import pyasdm

    from xradio.measurement_set._utils._asdm.create_pointing_xds import (
        create_pointing_xds,
    )

    def read_asdm(path):
        asdm = pyasdm.ASDM()
        with contextlib.redirect_stdout(io.StringIO()):
            asdm.setFromFile(path)
        return asdm

    truth = synth_interferometric_pointing
    asdm = read_asdm(truth.path)
    samples = truth.pointing_times_unix
    full = create_pointing_xds(asdm)
    np.testing.assert_allclose(full.time_pointing.values, samples, rtol=0, atol=1e-5)
    assert_true_utc_unix(full.time_pointing, samples)

    nsample = truth.spec.pointing_num_sample
    row_duration = samples[nsample] - samples[0] if len(samples) > nsample else 10.0
    t_lo, t_hi = samples[nsample + 1], samples[2 * nsample - 2]
    part = create_pointing_xds(asdm, time_range=(t_lo, t_hi))
    values = part.time_pointing.values
    assert np.all((values >= t_lo - row_duration) & (values <= t_hi + row_duration))
    inside = samples[(samples >= t_lo) & (samples <= t_hi)]
    assert np.isin(np.round(inside, 6), np.round(values, 6)).all()

    assert create_pointing_xds(asdm, time_range=(1.0e9, 1.0e9 + 10.0)) is None
    assert create_pointing_xds(read_asdm(synth_interferometric.path)) is None


def test_with_pointing_and_empty_pointing_table(synth_interferometric):
    """with_pointing=True on an ASDM without Pointing rows opens every
    partition, without pointing_xds (F76, K11)."""
    truth = synth_interferometric
    ps = open_ps(truth.path, with_pointing=True)
    nodes = msv4_nodes(ps)
    assert len(nodes) == len(truth.expected_partitions())
    for node in nodes:
        assert node.attrs["type"] == "visibility"
        assert "pointing_xds" not in node.children


# ---------------------------------------------------------------------------
# Antenna / baseline metadata (F28 with zero antenna offsets, F83)
# ---------------------------------------------------------------------------


def test_baseline_and_antenna_metadata(ps_interferometric, synth_interferometric):
    """Baseline antenna names follow the BDF order (cross baselines then autos)
    and ANTENNA_POSITION equals the pad positions (zero antenna offsets)."""
    truth = synth_interferometric
    names = truth.antenna_names
    pairs = truth.cross_baselines + [(ant, ant) for ant in range(truth.num_antenna)]
    for node in msv4_nodes(ps_interferometric):
        ds = node.ds
        assert list(map(str, ds.baseline_antenna1_name.values)) == [
            names[i] for i, _ in pairs
        ]
        assert list(map(str, ds.baseline_antenna2_name.values)) == [
            names[j] for _, j in pairs
        ]
        axds = node["antenna_xds"].ds
        assert list(map(str, axds.antenna_name.values)) == names
        np.testing.assert_allclose(
            axds.ANTENNA_POSITION.transpose("antenna_name", ...).values,
            truth.antenna_positions,
            rtol=0,
            atol=1e-6,
        )
