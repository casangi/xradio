"""
Value tests of the per-BDF visibility and flag loaders, with real (synthetic) BDF
files whose values encode their position (see synthetic_bdf.py). The expected
arrays are computed from the BDF definitions only.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_arr_trees import (
    SKIP_SUBSET_COMPONENTS,
    load_flags_all_subsets_from_trees,
    load_flags_subset_from_tree,
    load_subset_with_get_ndarrays,
    load_subset_with_get_subset,
    load_vis_subset_from_tree,
    load_visibilities_all_subsets_from_trees,
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


class SpyReader:
    """Wraps a BDFReader and records the calls that move through the subsets."""

    def __init__(self, reader):
        self.reader = reader
        self.calls = []

    def hasSubset(self):
        self.calls.append(("hasSubset",))
        return self.reader.hasSubset()

    def getSubset(self, loadOnlyComponents=None):
        subset = self.reader.getSubset(loadOnlyComponents=loadOnlyComponents)
        loaded = {
            name
            for name in ("crossData", "autoData", "flags", "actualTimes")
            if subset[name]["arr"] is not None
        }
        self.calls.append(("getSubset", set(loadOnlyComponents), loaded))
        return subset

    def getNDArrays(self, **kwargs):
        self.calls.append(("getNDArrays",))
        return self.reader.getNDArrays(**kwargs)

    def getPath(self):
        return self.reader.getPath()

    def reads(self):
        return [call for call in self.calls if call[0] != "hasSubset"]


def _open(bdf_name, tmp_path, **overrides):
    bdef = sbdf.BDF_DEFS[bdf_name]()
    for key, value in overrides.items():
        setattr(bdef, key, value)
    _path, reader, bdf_descr = sbdf.write_and_open(bdef, tmp_path)
    return bdef, reader, bdf_descr


def _norm(array_slice):
    return tuple(
        slice(key, key + 1) if isinstance(key, int) else key for key in array_slice
    )


def _selections(bdef, bb_spw):
    """Selections (time, baseline, frequency, polarization) valid for an SPW."""
    spw = bdef.basebands[bb_spw[0]][bb_spw[1]]
    nchan = spw.nchan
    npol = len(spw.sd_pols) if bdef.auto_only else len(spw.cross_pols)
    nbl = bdef.num_cross_baselines
    nant = bdef.num_antenna
    ntim = bdef.num_integrations
    return {
        "full": FULL,
        # time selection starting inside the BDF (F13)
        "time_inside": (slice(1, ntim), slice(None), slice(None), slice(None)),
        "time_int": (ntim - 1, slice(None), slice(None), slice(None)),
        # cross baselines only (F59), or first antennas for AUTO_ONLY
        "first_rows": (slice(None), slice(0, 2), slice(None), slice(None)),
        "row_int": (slice(None), 1, slice(None), slice(None)),
        # autos only
        "autos": (slice(None), slice(nbl, nbl + nant), slice(None), slice(None)),
        # last channel / last polarization (F04)
        "last_chan_pol": (
            slice(0, ntim),
            slice(1, nbl + nant),
            slice(nchan - 1, nchan),
            slice(npol - 1, npol),
        ),
        "mixed": (
            slice(ntim - 2, ntim),
            slice(max(nbl - 1, 0), nbl + 2),
            slice(nchan // 2, nchan),
            slice(0, npol),
        ),
    }


VIS_CASES = [
    (bdf_name, bb_spw)
    for bdf_name in sbdf.BDF_DEFS
    for bb_spw in sbdf.all_bb_spws(sbdf.BDF_DEFS[bdf_name]())
    if not (bdf_name == "bins_other_spw" and bb_spw == (0, 0))
]
SELECTION_NAMES = list(_selections(sbdf.BDF_DEFS["uneven"](), (0, 0)))


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
@pytest.mark.parametrize("selection_name", SELECTION_NAMES)
@pytest.mark.parametrize("bdf_name, bb_spw", VIS_CASES)
def test_load_visibilities_all_subsets_from_trees_values(
    tmp_path, bdf_name, bb_spw, selection_name, load_one_spw_from_file
):
    """Visibility values for every SPW, selection and both load paths (F03, F04,
    F11, F13, F18, F19, F20, F59)."""
    bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
    array_slice = _selections(bdef, bb_spw)[selection_name]
    expected = sbdf.expected_visibility(bdef, bb_spw)[_norm(array_slice)]

    vis = load_visibilities_all_subsets_from_trees(
        reader,
        None,
        bb_spw,
        bdf_descr,
        array_slice,
        load_one_spw_from_file=load_one_spw_from_file,
    )

    assert vis.dtype == np.complex64
    assert vis.shape == expected.shape
    np.testing.assert_array_equal(vis, expected.astype(np.complex64))


@pytest.mark.parametrize(
    "bdf_name, bb_spw, chunks",
    [
        ("uneven", (0, 0), (1, 4, 3, 1)),
        ("uneven", (1, 0), (2, 3, 2, 2)),
        ("full_pol_int32", (0, 0), (2, 5, 3, 3)),
        ("auto_only_full_pol", (0, 0), (2, 1, 3, 2)),
        ("packed_wvr", (0, 0), (2, 2, 3, 1)),
    ],
)
def test_load_visibilities_chunks_assemble_full_array(
    tmp_path, bdf_name, bb_spw, chunks
):
    """Chunked loads (as done by dask/xarray) give the full array (F04, F13)."""
    bdef = sbdf.BDF_DEFS[bdf_name]()
    path = sbdf.write_bdf(bdef, str(tmp_path / "bdf.bin"))
    expected = sbdf.expected_visibility(bdef, bb_spw)
    assembled = np.zeros(expected.shape, dtype=np.complex64)
    starts = [
        range(0, length, chunk)
        for length, chunk in zip(expected.shape, chunks, strict=False)
    ]
    for t0 in starts[0]:
        for b0 in starts[1]:
            for f0 in starts[2]:
                for p0 in starts[3]:
                    key = tuple(
                        slice(start, min(start + chunk, length))
                        for start, chunk, length in zip(
                            (t0, b0, f0, p0), chunks, expected.shape, strict=False
                        )
                    )
                    reader, bdf_descr = sbdf.open_bdf(path)
                    assembled[key] = load_visibilities_all_subsets_from_trees(
                        reader, None, bb_spw, bdf_descr, key
                    )
    np.testing.assert_array_equal(assembled, expected.astype(np.complex64))


def test_packed_bdf_gives_all_integrations(tmp_path):
    """A packed BDF (dimensionality 0, numTime=5 in one subset) contributes its 5
    integrations (F19, K5)."""
    bdef, reader, bdf_descr = _open("packed_wvr", tmp_path)
    vis = load_visibilities_all_subsets_from_trees(
        reader, None, (0, 0), bdf_descr, FULL
    )
    assert vis.shape == (5, 3, 4, 1)
    np.testing.assert_array_equal(vis, sbdf.expected_visibility(bdef, (0, 0)))
    # every integration has different values
    assert len(np.unique(vis[:, 0, 0, 0])) == 5

    _bdef, reader, bdf_descr = _open("packed_wvr", tmp_path)
    flags = load_flags_all_subsets_from_trees(
        reader, None, bdf_descr, (0, 0), (slice(2, 4), slice(None), None, None)
    )
    np.testing.assert_array_equal(flags, sbdf.expected_flags(bdef, (0, 0))[2:4])


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
def test_visibilities_skipped_subsets_are_not_loaded(tmp_path, load_one_spw_from_file):
    """Subsets before the time selection are skipped without loading data, and no
    subset is read after it (F12)."""
    bdef, reader, bdf_descr = _open("uneven", tmp_path, num_subsets=5)
    spy = SpyReader(reader)
    vis = load_visibilities_all_subsets_from_trees(
        spy,
        None,
        (1, 0),
        bdf_descr,
        (slice(2, 3), slice(None), slice(None), slice(None)),
        load_one_spw_from_file=load_one_spw_from_file,
    )
    np.testing.assert_array_equal(vis, sbdf.expected_visibility(bdef, (1, 0), [2]))

    reads = spy.reads()
    assert len(reads) == 3
    for skipped in reads[:2]:
        assert skipped == ("getSubset", SKIP_SUBSET_COMPONENTS, set())
    if load_one_spw_from_file:
        assert reads[2] == ("getNDArrays",)
    else:
        assert reads[2] == (
            "getSubset",
            {"crossData", "autoData"},
            {"crossData", "autoData"},
        )
    # stops after the last subset selected (subsets 3, 4 are never tested/read)
    assert spy.calls.count(("hasSubset",)) == 3


def test_visibilities_cross_only_does_not_load_autos(tmp_path):
    bdef, reader, bdf_descr = _open("uneven", tmp_path)
    spy = SpyReader(reader)
    array_slice = (slice(0, 1), slice(0, 3), slice(None), slice(None))
    vis = load_visibilities_all_subsets_from_trees(
        spy, None, (0, 1), bdf_descr, array_slice, load_one_spw_from_file=False
    )
    np.testing.assert_array_equal(
        vis, sbdf.expected_visibility(bdef, (0, 1))[array_slice]
    )
    assert spy.reads() == [("getSubset", {"crossData"}, {"crossData"})]


@pytest.mark.parametrize(
    "array_slice",
    [
        (slice(2, 5), slice(None), slice(None), slice(None)),
        (slice(3, None), slice(None), slice(None), slice(None)),
        (7, slice(None), slice(None), slice(None)),
    ],
)
def test_visibilities_time_out_of_range(tmp_path, array_slice):
    _bdef, reader, bdf_descr = _open("uneven", tmp_path)
    with pytest.raises(ValueError, match="BDF has only 3 integrations"):
        load_visibilities_all_subsets_from_trees(
            reader, None, (0, 0), bdf_descr, array_slice
        )


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
def test_visibilities_num_bin_rejected(tmp_path, load_one_spw_from_file):
    _bdef, reader, bdf_descr = _open("bins_other_spw", tmp_path)
    with pytest.raises(NotImplementedError, match="numBin"):
        load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (0, 0),
            bdf_descr,
            FULL,
            load_one_spw_from_file=load_one_spw_from_file,
        )


def test_visibilities_two_apc_loads_uncorrected(tmp_path):
    """With 2 APCs the AP_UNCORRECTED data are loaded (K6, F11)."""
    bdef, reader, bdf_descr = _open("two_apc", tmp_path)
    vis = load_visibilities_all_subsets_from_trees(
        reader, None, (1, 0), bdf_descr, FULL
    )
    nbl = bdef.num_cross_baselines
    expected = sbdf.expected_visibility(bdef, (1, 0))
    np.testing.assert_array_equal(vis, expected)
    # the AP_CORRECTED values (first APC in this BDF) are different
    corrected = sbdf.cross_true_value(
        bdef, 0, 0, 1, 0, 0, np.arange(3)[:, None], np.arange(2)[None, :]
    )
    assert not np.isin(vis[0, :nbl], corrected).any()


def test_load_vis_subset_from_tree_missing_components(tmp_path):
    bdef, reader, bdf_descr = _open("uneven", tmp_path)
    subset = reader.getSubset(loadOnlyComponents={"crossData"})
    cross_only = (slice(0, 1), slice(0, 6), slice(None), slice(None))
    vis = load_vis_subset_from_tree(subset, None, (0, 1), bdf_descr, cross_only)
    np.testing.assert_array_equal(
        vis, sbdf.expected_visibility(bdef, (0, 1), [0])[:, 0:6]
    )
    # autos selected but autoData not loaded
    with pytest.raises(RuntimeError, match="'autoData' was not loaded"):
        load_vis_subset_from_tree(subset, None, (0, 1), bdf_descr, FULL)
    with pytest.raises(RuntimeError, match="'crossData' not present"):
        load_vis_subset_from_tree({}, None, (0, 1), bdf_descr, cross_only)


FLAG_CASES = [
    (bdf_name, bb_spw)
    for bdf_name in sbdf.BDF_DEFS
    for bb_spw in sbdf.all_bb_spws(sbdf.BDF_DEFS[bdf_name]())
    if bdf_name not in ("bins_other_spw",)
]


@pytest.mark.parametrize("selection_name", SELECTION_NAMES)
@pytest.mark.parametrize("bdf_name, bb_spw", FLAG_CASES)
def test_load_flags_all_subsets_from_trees_values(
    tmp_path, bdf_name, bb_spw, selection_name
):
    """Flag values: bool, flagged == (word != 0), per baseline/antenna, time and
    polarization selections (F01, F14)."""
    bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
    array_slice = _norm(_selections(bdef, bb_spw)[selection_name])
    expected = sbdf.expected_flags(bdef, bb_spw)[
        array_slice[0], array_slice[1], array_slice[3]
    ]

    flags = load_flags_all_subsets_from_trees(
        reader, None, bdf_descr, bb_spw, _selections(bdef, bb_spw)[selection_name]
    )

    assert flags.dtype == bool
    assert flags.shape == expected.shape
    np.testing.assert_array_equal(flags, expected)


def test_flag_words_with_any_bit_are_flagged(tmp_path):
    """All the distinct flag bits (including the sign bit) give flagged (K2, F01)."""
    bdef, reader, bdf_descr = _open("uneven", tmp_path, num_subsets=6)
    flags = load_flags_all_subsets_from_trees(reader, None, bdf_descr, (0, 0), FULL)
    words = np.array(
        [
            [
                [sbdf.flag_word(tim, row, 0, pol) for pol in range(2)]
                for row in range(10)
            ]
            for tim in range(6)
        ]
    )
    assert set(np.unique(words[flags])) == {1, 16, 2**30, -(2**31), 1 | 16 | 2**30}
    assert (words[~flags] == 0).all()


def test_flags_subset_without_flags_component(tmp_path):
    """A subset without flags gives False for its integration only (F14)."""
    bdef, reader, bdf_descr = _open(
        "uneven", tmp_path, missing_components={1: {"flags"}}
    )
    array_slice = (slice(None), slice(2, 9), None, slice(1, 2))
    flags = load_flags_all_subsets_from_trees(
        reader, None, bdf_descr, (1, 0), array_slice
    )
    expected = sbdf.expected_flags(bdef, (1, 0))[:, 2:9, 1:2]
    expected[1] = False
    assert flags.shape == (3, 7, 1)
    np.testing.assert_array_equal(flags, expected)

    subset = {"flags": {"present": False, "arr": None}}
    flags = load_flags_subset_from_tree(subset, None, bdf_descr, (0, 0), array_slice)
    np.testing.assert_array_equal(flags, np.zeros((1, 7, 1), dtype=bool))


def test_flags_skipped_subsets_are_not_loaded(tmp_path):
    """Flags: subsets out of the time selection are skipped and the loop stops after
    the selection (F12, F14)."""
    bdef, reader, bdf_descr = _open("uneven", tmp_path, num_subsets=5)
    spy = SpyReader(reader)
    flags = load_flags_all_subsets_from_trees(
        spy, None, bdf_descr, (0, 0), (slice(1, 3), slice(None), None, None)
    )
    np.testing.assert_array_equal(flags, sbdf.expected_flags(bdef, (0, 0))[1:3])
    assert spy.reads() == [
        ("getSubset", SKIP_SUBSET_COMPONENTS, set()),
        ("getSubset", {"flags"}, {"flags"}),
        ("getSubset", {"flags"}, {"flags"}),
    ]
    assert spy.calls.count(("hasSubset",)) == 3


def test_flags_num_bin_rejected(tmp_path):
    _bdef, reader, bdf_descr = _open("bins_other_spw", tmp_path)
    with pytest.raises(NotImplementedError, match="numBin"):
        load_flags_all_subsets_from_trees(reader, None, bdf_descr, (0, 0), FULL)


def test_load_subset_with_get_subset():
    mock_bdf_reader = mock.create_autospec(pyasdm.bdf.BDFReader, instance=True)
    mock_bdf_reader.getSubset.return_value = {"flags": {}}
    subset = load_subset_with_get_subset(mock_bdf_reader, ["flags", "autoData"])
    assert subset == {"flags": {}}
    mock_bdf_reader.getSubset.assert_called_once_with(
        loadOnlyComponents={"flags", "autoData"}
    )
    mock_bdf_reader.hasSubset.assert_not_called()

    # An empty selection would make pyasdm load every binary component
    with pytest.raises(ValueError, match="No binary component"):
        load_subset_with_get_subset(mock_bdf_reader, [])

    mock_bdf_reader.getSubset.side_effect = pyasdm.exceptions.BDFReaderException(
        "broken subset"
    )
    mock_bdf_reader.getPath.return_value = "/path/to/bdf"
    with pytest.raises(RuntimeError, match="/path/to/bdf.*broken subset"):
        load_subset_with_get_subset(mock_bdf_reader, ["flags"])


def test_load_subset_with_get_ndarrays():
    mock_bdf_reader = mock.create_autospec(pyasdm.bdf.BDFReader, instance=True)
    result = {"visibilities": np.ones((1, 3, 2, 2), dtype=np.complex64)}
    mock_bdf_reader.getNDArrays.return_value = result

    def load_spw_function(*args):
        return None

    params = ({"foo": "bar"}, ["autoData"])
    assert (
        load_subset_with_get_ndarrays(mock_bdf_reader, 3, load_spw_function, params)
        is result
    )
    mock_bdf_reader.getNDArrays.assert_called_once_with(
        arrayNames=["visibilities"],
        spwId=3,
        loadOneSPWFunction=load_spw_function,
        loadOneSPWFunctionParams=params,
    )
    mock_bdf_reader.getSubset.assert_not_called()

    mock_bdf_reader.getNDArrays.side_effect = pyasdm.exceptions.BDFReaderException(
        "broken"
    )
    with pytest.raises(RuntimeError, match="getNDArrays.*broken"):
        load_subset_with_get_ndarrays(mock_bdf_reader, 3, load_spw_function, params)


@pytest.mark.parametrize(
    "bdf_name, bb_spw, array_slice",
    [
        ("full_pol_int32", (1, 0), (slice(None), slice(None), None, slice(None))),
        ("full_pol_int32", (0, 0), (slice(0, 1), slice(2, 5), None, slice(1, 3))),
        ("packed_wvr", (0, 0), (slice(1, 4), slice(1, 3), None, slice(None))),
        ("flags_per_baseband", (1, 1), (slice(None), slice(4, 8), None, 1)),
    ],
)
def test_load_flags_subset_from_tree_values(tmp_path, bdf_name, bb_spw, array_slice):
    """One subset: bool flags of the SPW, full-pol auto flags as [XX, XY, XY, YY]."""
    bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
    subset = reader.getSubset(loadOnlyComponents={"flags"})
    ntim = bdef.packed_num_time or 1
    norm_slice = _norm(array_slice)
    expected = sbdf.expected_flags(bdef, bb_spw, range(ntim))[
        norm_slice[0], norm_slice[1], norm_slice[3]
    ]
    flags = load_flags_subset_from_tree(subset, None, bdf_descr, bb_spw, array_slice)
    assert flags.dtype == bool
    np.testing.assert_array_equal(flags, expected)


def _garbage_view(shape, dtype=np.complex64):
    """View into a larger array filled with garbage, and the larger array."""
    fill = True if np.dtype(dtype) == bool else np.nan
    big = np.full((shape[0] + 1, shape[1] + 2, *shape[2:]), fill, dtype=dtype)
    return big[1:, 1:-1], big


def _untouched(big, fill):
    return (big[0] == fill).all() and (big[:, [0, -1]] == fill).all()


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
@pytest.mark.parametrize("selection_name", ["full", "time_inside", "mixed", "autos"])
@pytest.mark.parametrize(
    "bdf_name, bb_spw",
    [
        ("uneven", (1, 0)),
        ("full_pol_int32", (0, 0)),
        ("auto_only_full_pol", (0, 0)),
        ("packed_wvr", (0, 0)),
    ],
)
def test_load_visibilities_into_out(
    tmp_path, bdf_name, bb_spw, selection_name, load_one_spw_from_file
):
    """With out (a view of a larger array, uninitialized), every subset is written
    into its part of out, every element is written and out is returned (F20)."""
    bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
    array_slice = _selections(bdef, bb_spw)[selection_name]
    expected = sbdf.expected_visibility(bdef, bb_spw)[_norm(array_slice)]
    out, big = _garbage_view(expected.shape)

    vis = load_visibilities_all_subsets_from_trees(
        reader,
        None,
        bb_spw,
        bdf_descr,
        array_slice,
        load_one_spw_from_file=load_one_spw_from_file,
        out=out,
    )

    assert vis is out
    np.testing.assert_array_equal(out, expected.astype(np.complex64))
    assert np.isnan(big[0]).all() and np.isnan(big[:, [0, -1]]).all()


@pytest.mark.parametrize("selection_name", ["full", "time_inside", "mixed"])
@pytest.mark.parametrize(
    "bdf_name, bb_spw",
    [("uneven", (1, 0)), ("full_pol_int32", (0, 0)), ("packed_wvr", (0, 0))],
)
def test_load_flags_into_out(tmp_path, bdf_name, bb_spw, selection_name):
    bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
    array_slice = _norm(_selections(bdef, bb_spw)[selection_name])
    expected = sbdf.expected_flags(bdef, bb_spw)[
        array_slice[0], array_slice[1], array_slice[3]
    ]
    for fill in (True, False):
        reader, bdf_descr = sbdf.open_bdf(reader.getPath())
        big = np.full(
            (expected.shape[0] + 1, expected.shape[1] + 2, *expected.shape[2:]), fill
        )
        out = big[1:, 1:-1]
        flags = load_flags_all_subsets_from_trees(
            reader, None, bdf_descr, bb_spw, array_slice, out=out
        )
        assert flags is out
        np.testing.assert_array_equal(out, expected)
        assert _untouched(big, fill)


def test_flags_subset_without_flags_component_into_out(tmp_path):
    bdef, reader, bdf_descr = _open(
        "uneven", tmp_path, missing_components={1: {"flags"}}
    )
    expected = sbdf.expected_flags(bdef, (1, 0))
    expected[1] = False
    out = np.ones(expected.shape, dtype=bool)
    load_flags_all_subsets_from_trees(reader, None, bdf_descr, (1, 0), FULL, out=out)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
def test_open_ended_time_selection(tmp_path, load_one_spw_from_file):
    """Time slice without stop: until the last integration, or the length of out."""
    open_ended = (slice(1, None), slice(None), slice(None), slice(None))
    for bdf_name in ("uneven", "packed_wvr"):
        bdef, reader, bdf_descr = _open(bdf_name, tmp_path)
        expected = sbdf.expected_visibility(bdef, (0, 0))
        vis = load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (0, 0),
            bdf_descr,
            open_ended,
            load_one_spw_from_file=load_one_spw_from_file,
        )
        np.testing.assert_array_equal(vis, expected[1:].astype(np.complex64))

        # out gives the length of the time selection
        reader, bdf_descr = sbdf.open_bdf(reader.getPath())
        out = np.full((2, *expected.shape[1:]), np.nan, dtype=np.complex64)
        vis = load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (0, 0),
            bdf_descr,
            open_ended,
            load_one_spw_from_file=load_one_spw_from_file,
            out=out,
        )
        assert vis is out
        np.testing.assert_array_equal(out, expected[1:3].astype(np.complex64))

    reader, bdf_descr = sbdf.open_bdf(reader.getPath())
    with pytest.raises(ValueError, match="Output array with shape"):
        load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (0, 0),
            bdf_descr,
            open_ended,
            out=np.zeros((2, 3), dtype=np.complex64),
        )


@pytest.mark.parametrize("load_one_spw_from_file", [True, False])
def test_visibilities_component_missing_in_subset(tmp_path, load_one_spw_from_file):
    """A subset without a binary component that the selection needs gives an error,
    not unwritten (uninitialized) values in the result."""
    bdef, reader, bdf_descr = _open(
        "uneven", tmp_path, missing_components={1: {"autoData"}}
    )
    with pytest.raises(RuntimeError, match="autoData.*not present"):
        load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (0, 0),
            bdf_descr,
            FULL,
            load_one_spw_from_file=load_one_spw_from_file,
        )

    # cross baselines only: the subset without autoData is fine
    reader, bdf_descr = sbdf.open_bdf(reader.getPath())
    cross_only = (slice(None), slice(0, 6), slice(None), slice(None))
    vis = load_visibilities_all_subsets_from_trees(
        reader,
        None,
        (0, 0),
        bdf_descr,
        cross_only,
        load_one_spw_from_file=load_one_spw_from_file,
    )
    np.testing.assert_array_equal(
        vis, sbdf.expected_visibility(bdef, (0, 0))[:, 0:6].astype(np.complex64)
    )


@pytest.mark.parametrize("load", ["vis_get_ndarrays", "vis_get_subset", "flags"])
def test_layout_calculated_once_per_bdf_without_apc_warnings(
    tmp_path, monkeypatch, load
):
    """The SPW layout is calculated once per BDF load (not for every binary
    component of every subset), and loading data whose only APC is AP_CORRECTED
    logs no warning (it was logged 2N+1 times per chunk for N subsets)."""
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        load_from_pyasdm_arr_trees as trees_module,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        pyasdm_get_ndarray_load_function as callback_module,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf import shapes

    layout_calls = []
    warnings = []

    def counting_layout(*args, **kwargs):
        layout_calls.append(args)
        return shapes.calc_bdf_spw_layout(*args, **kwargs)

    for module in (trees_module, callback_module):
        monkeypatch.setattr(module, "calc_bdf_spw_layout", counting_layout)
    monkeypatch.setattr(
        shapes,
        "xradio_logger",
        lambda: mock.Mock(warning=lambda msg, *args, **kwargs: warnings.append(msg)),
    )

    bdef, reader, bdf_descr = _open(
        "uneven", tmp_path, num_subsets=6, apc=("AP_CORRECTED",)
    )
    if load == "flags":
        loaded = load_flags_all_subsets_from_trees(
            reader, None, bdf_descr, (1, 0), FULL
        )
        expected = sbdf.expected_flags(bdef, (1, 0))
    else:
        loaded = load_visibilities_all_subsets_from_trees(
            reader,
            None,
            (1, 0),
            bdf_descr,
            FULL,
            load_one_spw_from_file=load == "vis_get_ndarrays",
        )
        # the only APC (AP_CORRECTED) is loaded
        expected = sbdf.expected_visibility(bdef, (1, 0)).astype(np.complex64)

    np.testing.assert_array_equal(loaded, expected)
    assert len(layout_calls) == 1
    assert warnings == []


@pytest.fixture(scope="module")
def large_bdf_path(tmp_path_factory):
    """BDF with 64 antennas (2016 baselines), 2 full-polarization SPWs and 3
    subsets: about 1 MB of complex64 visibilities per integration for SPW 0."""
    bdef = sbdf.BDFDef([[sbdf._full(16)], [sbdf._full(8)]], num_antenna=64)
    path = str(tmp_path_factory.mktemp("large_bdf") / "bdf.bin")
    sbdf.write_bdf(bdef, path)
    return bdef, path


def test_one_integration_load_peak_memory(large_bdf_path, monkeypatch, peak_memory):
    """Regression guard for the memory part of F20: loading one integration
    (getNDArrays, the default path) allocates the result and one read block, not
    per-component arrays, their concatenated copy (pyasdm) and a per-subset copy
    (peak was about 3x the result). With out, nothing of the size of the result is
    allocated. Reads are limited to 64 rows (as for real data, where a read block
    is small relative to the result)."""
    from xradio.measurement_set._utils._asdm._utils._bdf import (
        pyasdm_get_ndarray_load_function as callback_module,
    )

    bdef, path = large_bdf_path
    layout_row_len = (16 + 8) * 4 * 2
    monkeypatch.setattr(callback_module, "MAX_VALUES_PER_READ", 64 * layout_row_len)
    key = (slice(1, 2), slice(None), slice(None), slice(None))
    expected = sbdf.expected_visibility(bdef, (0, 0), [1]).astype(np.complex64)

    reader, bdf_descr = sbdf.open_bdf(path)
    vis, peak = peak_memory(
        lambda: load_visibilities_all_subsets_from_trees(
            reader, None, (0, 0), bdf_descr, key
        )
    )
    np.testing.assert_array_equal(vis, expected)
    assert vis.nbytes > 1e6
    assert peak / vis.nbytes < 1.5

    reader, bdf_descr = sbdf.open_bdf(path)
    out = np.empty(expected.shape, dtype=np.complex64)
    vis, peak = peak_memory(
        lambda: load_visibilities_all_subsets_from_trees(
            reader, None, (0, 0), bdf_descr, key, out=out
        )
    )
    assert vis is out
    np.testing.assert_array_equal(out, expected)
    assert peak / out.nbytes < 0.4
