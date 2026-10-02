"""
Value tests of the loader injected into pyasdm BDFReader.getNDArrays() and of the
decoders of the crossData/autoData binary components, with real (synthetic) BDF
files whose values encode their position (see synthetic_bdf.py).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_get_ndarray_load_function import (
    load_auto_data_one_spw,
    load_cross_data_one_spw,
    load_visibilities_one_spw_to_ndarray,
    output_array,
    read_component_block,
)
from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    calc_bdf_spw_layout,
)


def _load_synthetic_bdf():
    name = "xradio_tests_asdm_bdf_synthetic_bdf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("synthetic_bdf.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


sbdf = _load_synthetic_bdf()

FULL = (slice(None), slice(None), slice(None), slice(None))


def _write_and_open(bdf_name, tmp_path):
    bdef = sbdf.BDF_DEFS[bdf_name]()
    _path, reader, bdf_descr = sbdf.write_and_open(bdef, tmp_path)
    return bdef, reader, bdf_descr


def _read_subset_components(reader) -> list[dict]:
    """Raw binary components (as pyasdm loads them) of every subset."""
    subsets = []
    while reader.hasSubset():
        subset = reader.getSubset(loadOnlyComponents={"crossData", "autoData"})
        subsets.append(
            {
                name: subset[name]["arr"]
                for name in ("crossData", "autoData")
                if subset[name]["present"]
            }
        )
    return subsets


def _component_file(tmp_path, values: np.ndarray, prefix: bytes = b"junk-prefix\n"):
    """File with some bytes and then the binary component, positioned at the component."""
    path = tmp_path / "component.bin"
    path.write_bytes(prefix + values.tobytes() + b"\ntrailing junk")
    bdf_file = open(path, "rb")
    bdf_file.seek(len(prefix))
    return bdf_file


def _expected_component_rows(bdef, expected, component, array_slice):
    """Part of the expected (time, baseline, frequency, pol) array of one component."""
    nbl = bdef.num_cross_baselines
    rows = np.arange(expected.shape[1])[array_slice[1]]
    rows = rows[rows < nbl] if component == "crossData" else rows[rows >= nbl]
    return expected[array_slice[0]][:, rows][:, :, array_slice[2], array_slice[3]]


CALLBACK_CASES = [
    # (BDF, (baseband, spw), array_slice)
    ("uneven", (0, 0), FULL),
    ("uneven", (0, 1), FULL),
    ("uneven", (1, 0), FULL),
    # channel selections not starting at 0 (F04)
    ("uneven", (0, 0), (slice(0, 1), slice(None), slice(4, 6), slice(None))),
    ("uneven", (1, 0), (slice(0, 1), slice(2, 8), slice(3, 4), slice(1, 2))),
    # cross-only / auto-only / mixed baseline selections
    ("uneven", (1, 0), (slice(0, 1), slice(1, 3), slice(None), slice(None))),
    ("uneven", (1, 0), (slice(0, 1), slice(7, 10), slice(None), slice(None))),
    ("uneven", (0, 0), (slice(0, 1), slice(5, 8), slice(1, 7), slice(0, 1))),
    ("full_pol_int32", (0, 0), FULL),
    ("full_pol_int32", (1, 0), (slice(0, 1), slice(None), slice(1, 2), slice(1, 3))),
    ("mixed_pols_int16", (0, 0), FULL),
    ("mixed_pols_int16", (0, 1), FULL),
    ("two_apc", (0, 0), FULL),
    ("two_apc", (1, 0), (slice(0, 1), slice(1, 4), slice(1, 3), slice(None))),
    ("big_endian", (0, 0), FULL),
    ("big_endian", (1, 0), FULL),
    ("bins_other_spw", (1, 0), FULL),
    ("auto_only", (0, 1), FULL),
    ("auto_only", (0, 0), (slice(0, 1), slice(1, 3), slice(2, 5), slice(1, 2))),
    ("auto_only_full_pol", (0, 0), FULL),
    ("auto_only_full_pol", (0, 0), (slice(0, 1), 1, slice(1, 3), slice(1, 2))),
    # packed: TIM samples of the (only) subset
    ("packed_wvr", (0, 0), FULL),
    ("packed_wvr", (0, 0), (slice(1, 4), slice(1, 2), slice(2, 4), slice(None))),
]


@pytest.mark.parametrize("subset_idx", [0, 2])
@pytest.mark.parametrize("bdf_name, bb_spw, array_slice", CALLBACK_CASES)
def test_load_visibilities_one_spw_to_ndarray_values(
    tmp_path, bdf_name, bb_spw, array_slice, subset_idx
):
    """Direct calls with the file positioned at the binary component (F03, F04, F18,
    F11, F20)."""
    bdef, reader, bdf_descr = _write_and_open(bdf_name, tmp_path)
    subsets = _read_subset_components(reader)
    subset_idx = min(subset_idx, len(subsets) - 1)
    ntim = bdef.packed_num_time or 1
    tims = range(subset_idx * ntim, (subset_idx + 1) * ntim)
    expected = sbdf.expected_visibility(bdef, bb_spw, tims)
    overall_spw_idx = bdef.overall_spw_idx(bb_spw)
    norm_slice = tuple(
        slice(key, key + 1) if isinstance(key, int) else key for key in array_slice
    )

    for component, values in subsets[subset_idx].items():
        with _component_file(tmp_path, values) as bdf_file:
            vis = load_visibilities_one_spw_to_ndarray(
                component,
                overall_spw_idx,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                array_slice,
            )
        expected_part = _expected_component_rows(bdef, expected, component, norm_slice)
        if expected_part.shape[1] == 0:
            assert vis is None
            continue
        assert vis.dtype == np.complex64
        assert vis.shape == expected_part.shape
        np.testing.assert_array_equal(vis, expected_part.astype(np.complex64))


@pytest.mark.parametrize(
    "bdf_name, bb_spw, array_slice",
    [case for case in CALLBACK_CASES if case[0] != "packed_wvr" or case[2] == FULL],
)
def test_load_visibilities_one_spw_with_get_ndarrays(
    tmp_path, bdf_name, bb_spw, array_slice
):
    """Through the real pyasdm BDFReader.getNDArrays(), every subset."""
    bdef, reader, bdf_descr = _write_and_open(bdf_name, tmp_path)
    expected = sbdf.expected_visibility(bdef, bb_spw)
    norm_slice = tuple(
        slice(key, key + 1) if isinstance(key, int) else key for key in array_slice
    )
    ntim = bdef.packed_num_time or 1
    subset_slice = (slice(0, ntim), *array_slice[1:])

    loaded = []
    while reader.hasSubset():
        ndarrays = reader.getNDArrays(
            ["visibilities"],
            bdef.overall_spw_idx(bb_spw),
            load_visibilities_one_spw_to_ndarray,
            (bdf_descr, ["crossData", "autoData"], None, subset_slice),
        )
        loaded.append(ndarrays["visibilities"])
    vis = np.concatenate(loaded)

    expected_sel = expected[:, norm_slice[1], norm_slice[2], norm_slice[3]]
    assert vis.dtype == np.complex64
    assert vis.shape == expected_sel.shape
    np.testing.assert_array_equal(vis, expected_sel.astype(np.complex64))


def test_load_visibilities_one_spw_component_not_selected(tmp_path):
    bdef, reader, bdf_descr = _write_and_open("uneven", tmp_path)
    values = _read_subset_components(reader)[0]["autoData"]
    with _component_file(tmp_path, values) as bdf_file:
        # not in components_to_load
        assert (
            load_visibilities_one_spw_to_ndarray(
                "autoData",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData"],
                None,
                FULL,
            )
            is None
        )
        # cross baselines only selected
        assert (
            load_visibilities_one_spw_to_ndarray(
                "autoData",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                (slice(0, 1), slice(0, 6), slice(None), slice(None)),
            )
            is None
        )
        with pytest.raises(ValueError, match="Unexpected binary component"):
            load_visibilities_one_spw_to_ndarray(
                "zeroLags",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["zeroLags"],
                None,
                FULL,
            )


def test_load_visibilities_one_spw_num_bin_rejected(tmp_path):
    """numBin > 1 in the SPW loaded: error, not silently wrong data (K6, F11)."""
    bdef, reader, bdf_descr = _write_and_open("bins_other_spw", tmp_path)
    values = _read_subset_components(reader)[0]["crossData"]
    with _component_file(tmp_path, values) as bdf_file:
        with pytest.raises(NotImplementedError, match="numBin"):
            load_visibilities_one_spw_to_ndarray(
                "crossData",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                FULL,
            )


def test_load_visibilities_one_spw_inconsistent_size(tmp_path):
    """The binary component size must match the header description (F11)."""
    bdef, reader, bdf_descr = _write_and_open("uneven", tmp_path)
    values = _read_subset_components(reader)[0]["crossData"]
    # The header announces 2 APCs but the data have only one
    bdf_descr["apc"] = [
        pyasdm.enumerations.AtmPhaseCorrection.AP_UNCORRECTED,
        pyasdm.enumerations.AtmPhaseCorrection.AP_CORRECTED,
    ]
    with _component_file(tmp_path, values) as bdf_file:
        with pytest.raises(ValueError, match="crossData binary component has"):
            load_visibilities_one_spw_to_ndarray(
                "crossData",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                FULL,
            )


def test_read_component_block_array_and_file(tmp_path):
    values = np.arange(100, dtype=">i4")
    expected = np.array([[[12, 13], [22, 23]], [[52, 53], [62, 63]]])
    block = read_component_block(values, 12, (2, 2, 2), (40, 10, 1))
    np.testing.assert_array_equal(block, expected)
    assert not block.flags.writeable

    with _component_file(tmp_path, values) as bdf_file:
        block = read_component_block(
            bdf_file, 12, (2, 2, 2), (40, 10, 1), np.dtype(">i4"), values.size
        )
        np.testing.assert_array_equal(block, expected)
        # one contiguous read of the needed span only
        assert bdf_file.tell() == len(b"junk-prefix\n") + 64 * 4


def test_read_component_block_errors(tmp_path):
    values = np.arange(10, dtype="<f4")
    with pytest.raises(ValueError, match="out of the binary component"):
        read_component_block(values, 5, (2, 3), (3, 1))
    with _component_file(tmp_path, values, prefix=b"") as bdf_file:
        # the header announces more values than the file has
        with pytest.raises(ValueError, match="Unexpected end of BDF file"):
            read_component_block(
                bdf_file, 9, (1, 8), (1, 1), np.dtype("<f4"), elements_count=20
            )


def _radiometer_cross_and_auto_descr():
    stokes = pyasdm.enumerations.StokesParameter
    spw = {
        "crossPolProducts": [stokes.XX, stokes.YY],
        "sdPolProducts": [stokes.XX, stokes.YY],
        "scaleFactor": 2.0,
        "numSpectralPoint": 3,
        "numBin": 1,
    }
    return {
        "dimensionality": 1,
        "num_time": 0,
        "processor_type": pyasdm.enumerations.ProcessorType.SPECTROMETER,
        "correlation_mode": pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO,
        "apc": [pyasdm.enumerations.AtmPhaseCorrection.AP_UNCORRECTED],
        "num_antenna": 3,
        "basebands": [
            {"name": "BB_1", "spectralWindows": [dict(spw)]},
            {"name": "BB_2", "spectralWindows": [dict(spw)]},
        ],
    }


def test_load_cross_data_one_spw_real_values_not_correlator():
    """crossData with real values (data not from the CORRELATOR)."""
    layout = calc_bdf_spw_layout(_radiometer_cross_and_auto_descr(), (1, 0))
    # 3 baselines x 2 SPWs x 3 channels x 2 pols, real int16 codes
    values = np.arange(3 * 2 * 3 * 2, dtype=np.int16)
    vis = load_cross_data_one_spw(values, layout, (0, 1), (1, 3), (1, 3), (0, 2))
    codes = values.reshape(3, 2, 3, 2)[1:3, 1, 1:3, :]
    assert vis.dtype == np.complex64
    np.testing.assert_array_equal(vis, (codes / 2.0)[np.newaxis].astype(complex))

    # Complex values are still read as (re, im) pairs
    values = np.arange(3 * 2 * 3 * 2 * 2, dtype=np.int16)
    vis = load_cross_data_one_spw(values, layout, (0, 1), (0, 3), (0, 3), (0, 2))
    pairs = values.reshape(3, 2, 3, 2, 2)[:, 1]
    np.testing.assert_array_equal(vis[0], (pairs[..., 0] + 1j * pairs[..., 1]) / 2.0)


def test_load_auto_data_one_spw_full_pol_decoding():
    """XX, Re(XY), Im(XY), YY -> [XX, XY, conj(XY), YY] for 4 output pols (F18)."""
    stokes = pyasdm.enumerations.StokesParameter
    descr = _radiometer_cross_and_auto_descr()
    for baseband in descr["basebands"]:
        baseband["spectralWindows"][0]["crossPolProducts"] = [
            stokes.XX,
            stokes.XY,
            stokes.YX,
            stokes.YY,
        ]
        baseband["spectralWindows"][0]["sdPolProducts"] = [
            stokes.XX,
            stokes.XY,
            stokes.YY,
        ]
    layout = calc_bdf_spw_layout(descr, (0, 0))
    values = (np.arange(3 * 2 * 3 * 4) + 0.5).astype(np.float32)
    vis = load_auto_data_one_spw(values, layout, (0, 1), (0, 3), (0, 3), (0, 4))
    raw = values.reshape(3, 2, 3, 4)[:, 0]
    xy = raw[..., 1] + 1j * raw[..., 2]
    expected = np.stack([raw[..., 0], xy, np.conj(xy), raw[..., 3]], axis=-1)
    np.testing.assert_array_equal(vis[0], expected)
    assert (vis[..., [0, 3]].imag == 0).all()

    vis = load_auto_data_one_spw(values, layout, (0, 1), (1, 2), (2, 3), (2, 3))
    np.testing.assert_array_equal(vis, expected[np.newaxis, 1:2, 2:3, 2:3])


@pytest.mark.parametrize(
    "bdf_name, bb_spw, array_slice",
    [
        ("uneven", (1, 0), FULL),
        (
            "full_pol_int32",
            (0, 0),
            (slice(0, 1), slice(1, 6), slice(1, 4), slice(1, 4)),
        ),
        ("packed_wvr", (0, 0), (slice(1, 5), slice(None), slice(None), slice(None))),
    ],
)
def test_load_visibilities_one_spw_reads_in_blocks(
    tmp_path, monkeypatch, bdf_name, bb_spw, array_slice
):
    """Reads limited to a few rows (several reads per component) give the same
    values, also from a file whose component does not start at 0."""
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        pyasdm_get_ndarray_load_function,
    )

    monkeypatch.setattr(pyasdm_get_ndarray_load_function, "MAX_VALUES_PER_READ", 7)
    bdef, reader, bdf_descr = _write_and_open(bdf_name, tmp_path)
    subset = _read_subset_components(reader)[0]
    ntim = bdef.packed_num_time or 1
    expected = sbdf.expected_visibility(bdef, bb_spw, range(ntim))
    for component, values in subset.items():
        with _component_file(tmp_path, values) as bdf_file:
            vis = load_visibilities_one_spw_to_ndarray(
                component,
                bdef.overall_spw_idx(bb_spw),
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                array_slice,
            )
        expected_part = _expected_component_rows(bdef, expected, component, array_slice)
        np.testing.assert_array_equal(vis, expected_part.astype(np.complex64))


def _norm(array_slice):
    return tuple(
        slice(key, key + 1) if isinstance(key, int) else key for key in array_slice
    )


def _garbage_view(shape, dtype=np.complex64):
    """View into a larger array filled with garbage (NaN), and the larger array."""
    big = np.full((shape[0] + 1, shape[1] + 2, *shape[2:]), np.nan, dtype=dtype)
    return big[1:, 1:-1], big


@pytest.mark.parametrize("pass_layout", [False, True])
@pytest.mark.parametrize("bdf_name, bb_spw, array_slice", CALLBACK_CASES)
def test_load_visibilities_one_spw_writes_into_out(
    tmp_path, bdf_name, bb_spw, array_slice, pass_layout
):
    """With out, the selected cross and auto rows go into their place in out (here a
    view of a larger array filled with garbage), every element of out is written,
    nothing around it is touched, and None is returned so that getNDArrays does not
    concatenate the components (F20)."""
    bdef, reader, bdf_descr = _write_and_open(bdf_name, tmp_path)
    subset = _read_subset_components(reader)[0]
    ntim = bdef.packed_num_time or 1
    expected = sbdf.expected_visibility(bdef, bb_spw, range(ntim))[_norm(array_slice)]
    layout = calc_bdf_spw_layout(bdf_descr, bb_spw) if pass_layout else None
    out, big = _garbage_view(expected.shape)
    loaded = set()

    for component, values in subset.items():
        with _component_file(tmp_path, values) as bdf_file:
            result = load_visibilities_one_spw_to_ndarray(
                component,
                bdef.overall_spw_idx(bb_spw),
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                array_slice,
                out,
                layout,
                loaded,
            )
        assert result is None

    np.testing.assert_array_equal(out, expected.astype(np.complex64))
    assert np.isnan(big[0]).all()
    assert np.isnan(big[:, 0]).all() and np.isnan(big[:, -1]).all()
    nbl = bdef.num_cross_baselines
    rows = np.arange(nbl + bdef.num_antenna)[_norm(array_slice)[1]]
    expected_loaded = set()
    if (rows < nbl).any():
        expected_loaded.add("crossData")
    if (rows >= nbl).any():
        expected_loaded.add("autoData")
    assert loaded == expected_loaded


@pytest.mark.parametrize(
    "bdf_name, bb_spw, array_slice",
    [
        ("uneven", (0, 0), FULL),
        (
            "full_pol_int32",
            (1, 0),
            (slice(0, 1), slice(1, 6), slice(1, 2), slice(1, 4)),
        ),
        ("auto_only_full_pol", (0, 0), FULL),
        ("packed_wvr", (0, 0), FULL),
    ],
)
def test_get_ndarrays_with_out(tmp_path, bdf_name, bb_spw, array_slice):
    """Through the real pyasdm BDFReader.getNDArrays(): with out, getNDArrays
    returns None as "visibilities" (no assembly / concatenation in pyasdm) and
    every subset is written into its part of the result."""
    bdef, reader, bdf_descr = _write_and_open(bdf_name, tmp_path)
    ntim = bdef.packed_num_time or 1
    subset_slice = (slice(0, ntim), *array_slice[1:])
    expected = sbdf.expected_visibility(bdef, bb_spw)[(slice(None), *array_slice[1:])]
    result = np.full(expected.shape, np.nan, dtype=np.complex64)
    layout = calc_bdf_spw_layout(bdf_descr, bb_spw)

    subset_idx = 0
    while reader.hasSubset():
        loaded = set()
        ndarrays = reader.getNDArrays(
            ["visibilities"],
            bdef.overall_spw_idx(bb_spw),
            load_visibilities_one_spw_to_ndarray,
            (
                bdf_descr,
                ["crossData", "autoData"],
                None,
                subset_slice,
                result[subset_idx * ntim : (subset_idx + 1) * ntim],
                layout,
                loaded,
            ),
        )
        assert ndarrays["visibilities"] is None
        assert loaded == ({"autoData"} if bdef.auto_only else {"crossData", "autoData"})
        subset_idx += 1

    np.testing.assert_array_equal(result, expected.astype(np.complex64))


def test_output_array_checks():
    out = np.zeros((2, 3, 4, 1), dtype=np.complex64)
    assert output_array(out, (2, 3, 4, 1), np.complex64) is out
    new = output_array(None, (2, 3), bool)
    assert new.shape == (2, 3) and new.dtype == bool
    with pytest.raises(ValueError, match="expected a writeable array"):
        output_array(out, (2, 3, 4, 2), np.complex64)
    with pytest.raises(ValueError, match="expected a writeable array"):
        output_array(out.astype(np.complex128), (2, 3, 4, 1), np.complex64)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="expected a writeable array"):
        output_array(out, (2, 3, 4, 1), np.complex64)


def test_load_visibilities_one_spw_out_with_wrong_shape(tmp_path):
    bdef, reader, bdf_descr = _write_and_open("uneven", tmp_path)
    values = _read_subset_components(reader)[0]["crossData"]
    with _component_file(tmp_path, values) as bdf_file:
        with pytest.raises(ValueError, match="Output array with shape"):
            load_visibilities_one_spw_to_ndarray(
                "crossData",
                0,
                bdf_file,
                values.dtype,
                values.size,
                bdf_descr,
                ["crossData", "autoData"],
                None,
                FULL,
                # only the cross rows: the shape of the whole selection is expected
                np.zeros((1, 6, 8, 2), dtype=np.complex64),
            )


def test_decoders_write_every_element_of_out():
    """The decoders write every element of out (imaginary part 0 for real values),
    so out can be uninitialized memory (F20)."""
    # real crossData (data not from the CORRELATOR)
    layout = calc_bdf_spw_layout(_radiometer_cross_and_auto_descr(), (1, 0))
    values = np.arange(3 * 2 * 3 * 2, dtype=np.int16)
    out, big = _garbage_view((1, 2, 2, 2))
    vis = load_cross_data_one_spw(
        values, layout, (0, 1), (1, 3), (1, 3), (0, 2), out=out
    )
    assert vis is out
    codes = values.reshape(3, 2, 3, 2)[1:3, 1, 1:3, :]
    np.testing.assert_array_equal(out, (codes / 2.0)[np.newaxis].astype(complex))
    assert np.isnan(big[0]).all() and np.isnan(big[:, [0, -1]]).all()

    # real autoData (dual polarization)
    values = (np.arange(3 * 2 * 3 * 2) + 0.5).astype(np.float32)
    out, _big = _garbage_view((1, 3, 3, 2))
    load_auto_data_one_spw(values, layout, (0, 1), (0, 3), (0, 3), (0, 2), out=out)
    np.testing.assert_array_equal(
        out[0], values.reshape(3, 2, 3, 2)[:, 1].astype(np.complex64)
    )


@pytest.mark.parametrize("antenna_range", [(0, 3), (1, 2)])
@pytest.mark.parametrize("dtype", ["<f4", ">f4"])
@pytest.mark.parametrize("pol_range", [(0, 4), (1, 4), (2, 3), (3, 4)])
def test_full_pol_auto_decoding_into_strided_out(pol_range, dtype, antenna_range):
    """Full-polarization autoData into a strided view of uninitialized memory:
    [XX, XY, conj(XY), YY] (F18, F20). One antenna (rows dimension of length 1)
    also guards against a numpy 2.5 bug: np.negative with a strided float32 input
    and a strided output reads the input as contiguous."""
    stokes = pyasdm.enumerations.StokesParameter
    descr = _radiometer_cross_and_auto_descr()
    for baseband in descr["basebands"]:
        spw = baseband["spectralWindows"][0]
        spw["crossPolProducts"] = [stokes.XX, stokes.XY, stokes.YX, stokes.YY]
        spw["sdPolProducts"] = [stokes.XX, stokes.XY, stokes.YY]
    layout = calc_bdf_spw_layout(descr, (1, 0))
    values = (np.arange(3 * 2 * 3 * 4) + 0.5).astype(dtype)
    raw = values.reshape(3, 2, 3, 4)[slice(*antenna_range), 1].astype(np.float64)
    xy = raw[..., 1] + 1j * raw[..., 2]
    expected = np.stack([raw[..., 0], xy, np.conj(xy), raw[..., 3]], axis=-1)
    npol = pol_range[1] - pol_range[0]
    out, big = _garbage_view((1, len(raw), 3, npol))
    load_auto_data_one_spw(
        values, layout, (0, 1), antenna_range, (0, 3), pol_range, out=out
    )
    np.testing.assert_array_equal(out[0], expected[..., slice(*pol_range)])
    assert np.isnan(big[0]).all() and np.isnan(big[:, [0, -1]]).all()


def _rows_per_read(row_len: int, max_values: int) -> int:
    return max(1, max_values // row_len)


@pytest.mark.parametrize("max_values_per_read", [None, 7, 100, 1000])
@pytest.mark.parametrize(
    "baselines", [slice(None), slice(3, 17)], ids=["all_rows", "some_rows"]
)
def test_file_reads_per_component_are_bounded(
    tmp_path, monkeypatch, max_values_per_read, baselines
):
    """Regression guard for the performance part of F20: rows are read in
    contiguous blocks, ceil(rows / rows_per_read) reads per integration and binary
    component (rows_per_read = MAX_VALUES_PER_READ // row_len, at least 1): one
    read per integration and component with the default bound, never one per
    baseline/antenna, and every read is bounded by MAX_VALUES_PER_READ (or one row
    span)."""
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        pyasdm_get_ndarray_load_function as module,
    )

    if max_values_per_read is not None:
        monkeypatch.setattr(module, "MAX_VALUES_PER_READ", max_values_per_read)
    max_values = module.MAX_VALUES_PER_READ
    bdef = sbdf.BDFDef([[sbdf._dual(8), sbdf._dual(3)], [sbdf._full(4)]], num_antenna=6)
    _path, reader, bdf_descr = sbdf.write_and_open(bdef, tmp_path)
    bb_spw = (1, 0)
    layout = calc_bdf_spw_layout(bdf_descr, bb_spw)
    array_slice = (slice(0, 1), baselines, slice(None), slice(None))
    rows = np.arange(layout.num_baselines)[baselines]
    num_cross_rows = int((rows < layout.num_cross_baselines).sum())
    num_auto_rows = len(rows) - num_cross_rows
    expected_reads_per_integration = -(
        -num_cross_rows // _rows_per_read(layout.cross_row_len, max_values)
    ) + -(-num_auto_rows // _rows_per_read(layout.auto_row_len, max_values))
    if max_values_per_read is None:
        assert expected_reads_per_integration == 2

    reads = []
    real_fromfile = np.fromfile

    def counting_fromfile(file, dtype=float, count=-1, **kwargs):
        reads.append(count)
        return real_fromfile(file, dtype=dtype, count=count, **kwargs)

    expected = sbdf.expected_visibility(bdef, bb_spw)[:, baselines]
    result = np.empty(expected.shape, dtype=np.complex64)
    monkeypatch.setattr(np, "fromfile", counting_fromfile)
    subset_idx = 0
    while reader.hasSubset():
        reader.getNDArrays(
            ["visibilities"],
            bdef.overall_spw_idx(bb_spw),
            load_visibilities_one_spw_to_ndarray,
            (
                bdf_descr,
                ["crossData", "autoData"],
                None,
                array_slice,
                result[subset_idx : subset_idx + 1],
                layout,
            ),
        )
        subset_idx += 1
    monkeypatch.undo()

    assert len(reads) == bdef.num_subsets * expected_reads_per_integration
    assert max(reads) <= max(max_values, layout.cross_row_len)
    np.testing.assert_array_equal(result, expected.astype(np.complex64))
