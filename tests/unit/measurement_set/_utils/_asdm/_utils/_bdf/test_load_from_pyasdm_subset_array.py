"""
Tests of the reshape-based ("subset array") BDF loaders.

Two kinds of tests:

- with mocked BDF readers (``create_autospec``) returning hand-built binary
  components that follow the layouts of the BDF specification, with realistic
  shapes (from ``define_flag_shape`` / ``define_visibility_shape``) and
  position-dependent values;
- with real MIME BDFs written by the synthetic ASDM writer
  (``synthetic_asdm.py``) and read with ``pyasdm.bdf.BDFReader``, comparing the
  loaded values against the expected values computed independently by the
  writer.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils._bdf.load_from_pyasdm_subset_array import (
    _SKIP_ALL_BINARY_COMPONENTS,
    _auto_polarization_map,
    _dim_slice,
    _find_apc_index,
    _find_flags_layout,
    _load_vis_subset_auto_data,
    _load_vis_subset_cross_data,
    _split_baseline_slice,
    _time_range,
    define_flag_shape,
    define_visibility_shape,
    load_flags_all_subsets,
    load_visibilities_all_subsets,
)


def _load_synthetic_asdm():
    """Load synthetic_asdm.py (same module object as the conftest fixture)."""
    name = "xradio_tests_asdm_synthetic_asdm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parents[2] / "synthetic_asdm.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


synth = _load_synthetic_asdm()

StokesParameter = pyasdm.enumerations.StokesParameter
CROSS_AND_AUTO = pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO
AUTO_ONLY = pyasdm.enumerations.CorrelationMode.AUTO_ONLY

POL_PRODUCTS = {
    1: [StokesParameter.XX],
    2: [StokesParameter.XX, StokesParameter.YY],
    3: [StokesParameter.XX, StokesParameter.XY, StokesParameter.YY],
    4: [
        StokesParameter.XX,
        StokesParameter.XY,
        StokesParameter.YX,
        StokesParameter.YY,
    ],
}

ALL = (slice(None), slice(None), slice(None), slice(None))

#: int32 flag words, with several distinct non-zero bits (any bit set => flagged)
FLAG_WORDS = np.array([0, 1, 0, 16, 2**30, 0, -(2**31), 3, 0], dtype=np.int32)


# From uid___A002_Xc33ac1_X136e (AUTO_ONLY)
bdf_descr_X136e = {
    "dimensionality": 1,
    "num_time": 0,
    "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
    "binary_types": [
        "flags",
        "actualTimes",
        "actualDurations",
        "zeroLags",
        "crossData",
        "autoData",
    ],
    "correlation_mode": pyasdm.enumerations.CorrelationMode.AUTO_ONLY,
    "apc": [],
    "num_antenna": 10,
    "basebands": [
        {
            "name": "BB_1",
            "spectralWindows": [
                {
                    "crossPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 512,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "2",
                },
            ],
        },
        {
            "name": "BB_2",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                },
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "2",
                },
            ],
        },
        {
            "name": "BB_3",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 2048,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.LSB,
                    "sw": "1",
                }
            ],
        },
        {
            "name": "BB_4",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [
                        pyasdm.enumerations.StokesParameter.XX,
                        pyasdm.enumerations.StokesParameter.YY,
                    ],
                    "scaleFactor": None,
                    "numSpectralPoint": 1024,
                    "numBin": 1,
                    "sideband": pyasdm.enumerations.NetSideband.USB,
                    "sw": "1",
                }
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_bdf_descr(
    num_antenna: int = 3,
    num_baseband: int = 2,
    spws_per_baseband: int = 2,
    num_chan: int = 4,
    cross_pol: int = 4,
    sd_pol: int = 3,
    correlation_mode=CROSS_AND_AUTO,
    apc: tuple[str, ...] = ("AP_UNCORRECTED",),
    num_time: int = 0,
    num_bin: int = 1,
    scale_factor: float | None = 4.0,
) -> dict:
    """A BDF description (as made from a BDF header) with a regular layout."""
    auto_only = correlation_mode == AUTO_ONLY
    basebands = []
    for bb_idx in range(num_baseband):
        spws = [
            {
                "crossPolProducts": [] if auto_only else POL_PRODUCTS[cross_pol],
                "sdPolProducts": POL_PRODUCTS[sd_pol],
                "scaleFactor": None if auto_only else scale_factor,
                "numSpectralPoint": num_chan,
                "numBin": num_bin,
                "sideband": pyasdm.enumerations.NetSideband.USB,
                "sw": str(spw_idx + 1),
            }
            for spw_idx in range(spws_per_baseband)
        ]
        basebands.append({"name": f"BB_{bb_idx + 1}", "spectralWindows": spws})
    return {
        "dimensionality": 0 if num_time else 1,
        "num_time": num_time,
        "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
        "binary_types": ["flags", "crossData", "autoData"],
        "correlation_mode": correlation_mode,
        "apc": []
        if auto_only
        else [pyasdm.enumerations.AtmPhaseCorrection.literal(name) for name in apc],
        "num_antenna": num_antenna,
        "basebands": basebands,
    }


def mock_bdf_reader(subsets: list[dict], path: str = "/asdm/ASDMBinary/uid_X1_X2"):
    """Autospec'd BDFReader returning the given subsets."""
    reader = mock.create_autospec(pyasdm.bdf.BDFReader, instance=True)
    reader.getPath.return_value = path
    reader.hasSubset.side_effect = [True] * len(subsets) + [False]
    reader.getSubset.side_effect = subsets
    return reader


def make_flag_words(
    num_tim: int,
    num_baseline: int,
    num_antenna: int,
    num_baseband: int,
    num_spw: int,
    cross_pol: int,
    sd_pol: int,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Position-dependent int32 flag words in the BDF layout: for every TIM sample,
    the cross block (BAL BAB SPW POL) then the auto block (ANT BAB SPW POL).
    Returns the 1-D array and the (tim, row, bb, spw, pol) cross/auto blocks.
    """
    cross_idx = np.arange(num_tim * num_baseline * num_baseband * num_spw * cross_pol)
    cross = FLAG_WORDS[(cross_idx * 7 + offset) % len(FLAG_WORDS)].reshape(
        num_tim, num_baseline, num_baseband, num_spw, cross_pol
    )
    auto_idx = np.arange(num_tim * num_antenna * num_baseband * num_spw * sd_pol)
    auto = FLAG_WORDS[(auto_idx * 5 + offset + 1) % len(FLAG_WORDS)].reshape(
        num_tim, num_antenna, num_baseband, num_spw, sd_pol
    )
    words = np.concatenate(
        [cross.reshape(num_tim, -1), auto.reshape(num_tim, -1)], axis=1
    ).ravel()
    return words, cross, auto


def expected_flags_from_blocks(cross, auto, baseband_idx, spw_idx, num_pol):
    """(tim, baseline, pol) bool flags of one SPW, autos mapped to num_pol pols."""
    auto_spw = auto[:, :, baseband_idx, spw_idx, :]
    if auto_spw.shape[-1] == 3 and num_pol == 4:
        auto_spw = auto_spw[..., [0, 1, 1, 2]]
    blocks = [auto_spw != 0]
    if cross is not None:
        blocks.insert(0, cross[:, :, baseband_idx, spw_idx, :] != 0)
    return np.concatenate(blocks, axis=1)


def bdf_description(header) -> dict:
    """BDF description from a pyasdm BDFHeader (as robust_load_data_flags does)."""
    return {
        "dimensionality": header.getDimensionality(),
        "num_time": header.getNumTime(),
        "processor_type": header.getProcessorType(),
        "binary_types": header.getBinaryTypes(),
        "correlation_mode": header.getCorrelationMode(),
        "apc": header.getAPClist(),
        "num_antenna": header.getNumAntenna(),
        "basebands": header.getBasebandsList(),
    }


def load_from_synthetic_bdf(bdf_path: str, spw, what: str, array_slice):
    """Load visibilities ('vis') or flags ('flags') of one SPW from a real BDF."""
    reader = pyasdm.bdf.BDFReader()
    reader.open(bdf_path)
    try:
        bdf_descr = bdf_description(reader.getHeader())
        idxs = (spw.baseband_index, spw.index_in_baseband)
        if what == "vis":
            shape = define_visibility_shape(bdf_descr, idxs)
            return load_visibilities_all_subsets(
                reader, shape, idxs, bdf_descr, array_slice
            )
        shape = define_flag_shape(bdf_descr, idxs)
        return load_flags_all_subsets(reader, shape, idxs, array_slice)
    finally:
        reader.close()


def _single_scan_spec(cfg, name: str, num_integrations: int = 3, num_antenna=3):
    return synth.ASDMSpec(
        configs=[cfg],
        scans=[synth.ScanSpec([synth.SubscanSpec(0, "ON_SOURCE", num_integrations)])],
        fields=[synth.FieldSpec("J0423-0120", (1.1487, -0.0234))],
        num_antenna=num_antenna,
        name=name,
    )


def _full_pol_regular_spec(apc=("AP_UNCORRECTED", "AP_CORRECTED")):
    """Full-pol, INT32 (scale 4), 2 basebands x 2 SPWs of 4 channels, 2 APCs."""
    cfg = synth.ConfigSpec(
        basebands=[
            synth.BasebandSpec("BB_1", [synth.SpwSpec(4, "full")] * 2),
            synth.BasebandSpec("BB_2", [synth.SpwSpec(4, "full")] * 2),
        ],
        cross_type="INT32_TYPE",
        scale_factor=4.0,
        apc=apc,
    )
    return _single_scan_spec(cfg, "uid___A002_X1_X51")


def _dual_pol_int16_spec():
    """Dual-pol, INT16 (scale 3), 2 basebands x 1 SPW, APC order corrected first."""
    cfg = synth.ConfigSpec(
        basebands=[
            synth.BasebandSpec("BB_1", [synth.SpwSpec(4, "dual")]),
            synth.BasebandSpec("BB_2", [synth.SpwSpec(4, "dual")]),
        ],
        cross_type="INT16_TYPE",
        scale_factor=3.0,
        apc=("AP_CORRECTED", "AP_UNCORRECTED"),
    )
    return _single_scan_spec(cfg, "uid___A002_X1_X52", num_antenna=4)


def _float32_spec():
    """Dual-pol FLOAT32 data with scaleFactor 2 (float data are used as is)."""
    cfg = synth.ConfigSpec(
        basebands=[
            synth.BasebandSpec("BB_1", [synth.SpwSpec(3, "dual")]),
            synth.BasebandSpec("BB_2", [synth.SpwSpec(3, "dual")]),
            synth.BasebandSpec("BB_3", [synth.SpwSpec(3, "dual")]),
        ],
        cross_type="FLOAT32_TYPE",
        scale_factor=2.0,
    )
    return _single_scan_spec(cfg, "uid___A002_X1_X53")


def _auto_only_full_pol_spec():
    """AUTO_ONLY (single dish), 3 sdPolProducts, 2 basebands x 1 SPW."""
    cfg = synth.ConfigSpec(
        basebands=[
            synth.BasebandSpec("BB_1", [synth.SpwSpec(4, "full")]),
            synth.BasebandSpec("BB_2", [synth.SpwSpec(4, "full")]),
        ],
        correlation_mode="AUTO_ONLY",
    )
    return _single_scan_spec(cfg, "uid___A002_X1_X54")


def _packed_wvr_spec():
    """WVR-like packed BDF: RADIOMETER, AUTO_ONLY, dimensionality 0, numTime=4."""
    cfg = synth.ConfigSpec(
        basebands=[
            synth.BasebandSpec(
                "NOBB", [synth.SpwSpec(4, "stokes_i", chan_freq_start=1.83e11)]
            )
        ],
        correlation_mode="AUTO_ONLY",
        processor_type="RADIOMETER",
        packed_num_time=4,
    )
    return _single_scan_spec(cfg, "uid___A002_X1_X55", num_integrations=2)


SYNTHETIC_SPECS = {
    "full_pol_int32_2apc": _full_pol_regular_spec,
    "dual_pol_int16_apc_reversed": _dual_pol_int16_spec,
    "float32_scaled": _float32_spec,
    "auto_only_full_pol": _auto_only_full_pol_spec,
    "packed_wvr": _packed_wvr_spec,
}


@pytest.fixture(scope="module")
def synthetic_truths(make_synthetic_asdm):
    truths = {}
    for name, make_spec in SYNTHETIC_SPECS.items():
        truth = make_synthetic_asdm(make_spec())
        synth.verify_bdf_roundtrip(truth)
        truths[name] = truth
    return truths


def expected_values(truth, spw_id: int, bdf):
    """
    Expected (visibility, flags) of one SPW in one BDF, flags without the
    frequency axis. The expected visibilities are computed for the
    AP_UNCORRECTED data (APC code offset 0 in the writer), whatever the
    position of AP_UNCORRECTED in the APC list.
    """
    integrations = [synth.IntegrationRef(bdf, idx) for idx in range(bdf.num_times)]
    cfg = truth.config(bdf.config_idx)
    truth_uncorrected = dataclasses.replace(
        truth,
        spec=dataclasses.replace(
            truth.spec,
            configs=[
                dataclasses.replace(config, apc=("AP_UNCORRECTED",))
                if config is cfg
                else config
                for config in truth.spec.configs
            ],
        ),
    )
    vis = synth.expected_visibility(truth_uncorrected, spw_id, integrations)
    flags = synth.expected_flags(truth, spw_id, integrations)[:, :, 0, :]
    return vis, flags


# ---------------------------------------------------------------------------
# define_visibility_shape / define_flag_shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_bdf_descr, input_baseband_spw_idxs, expected_shape",
    [
        (bdf_descr_X136e, (0, 0), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (0, 1), (1, 45, 10, 4, 2, 512, 2, 2)),
        (bdf_descr_X136e, (1, 0), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (1, 1), (1, 45, 10, 4, 2, 1024, 2, 2)),
        (bdf_descr_X136e, (2, 0), (1, 45, 10, 4, 1, 2048, 2, 2)),
        # full pol interferometric: 4 output products (crossPolProducts)
        (make_bdf_descr(), (1, 1), (1, 3, 3, 2, 2, 4, 4, 2)),
        # AUTO_ONLY full pol: output products are the 3 sdPolProducts
        (
            make_bdf_descr(correlation_mode=AUTO_ONLY, num_antenna=5),
            (0, 1),
            (1, 10, 5, 2, 2, 4, 3, 2),
        ),
        # packed (dimensionality 0): numTime integrations per subset
        (
            make_bdf_descr(correlation_mode=AUTO_ONLY, sd_pol=1, num_time=7),
            (0, 0),
            (7, 3, 3, 2, 2, 4, 1, 2),
        ),
    ],
)
def test_define_visibility_shape(
    input_bdf_descr, input_baseband_spw_idxs, expected_shape
):
    shape = define_visibility_shape(input_bdf_descr, input_baseband_spw_idxs)
    assert shape == expected_shape


def test_define_visibility_shape_num_bin_not_supported():
    bdf_descr = make_bdf_descr()
    bdf_descr["basebands"][1]["spectralWindows"][1]["numBin"] = 2
    with pytest.raises(NotImplementedError, match="numBin > 1"):
        define_visibility_shape(bdf_descr, (1, 1))
    # the shape of an SPW without bins is still defined
    assert define_visibility_shape(bdf_descr, (0, 0)) == (1, 3, 3, 2, 2, 4, 4, 2)


def test_define_flag_shape_X136e():
    shape = define_flag_shape(bdf_descr_X136e, (0, 0))
    assert shape == {"cross": (), "auto": (1, 10, 4, 2, 2)}


@pytest.mark.parametrize(
    "bdf_descr, idxs, expected",
    [
        (
            make_bdf_descr(num_antenna=4),
            (1, 0),
            {"cross": (1, 6, 2, 2, 4), "auto": (1, 4, 2, 2, 3)},
        ),
        (
            make_bdf_descr(num_antenna=4, cross_pol=2, sd_pol=2, spws_per_baseband=1),
            (1, 0),
            {"cross": (1, 6, 2, 1, 2), "auto": (1, 4, 2, 1, 2)},
        ),
        (
            make_bdf_descr(correlation_mode=AUTO_ONLY, sd_pol=1, num_time=5),
            (0, 1),
            {"cross": (), "auto": (5, 3, 2, 2, 1)},
        ),
    ],
)
def test_define_flag_shape(bdf_descr, idxs, expected):
    assert define_flag_shape(bdf_descr, idxs) == expected


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, dim_len, expected",
    [
        (None, 5, slice(0, 5)),
        (slice(None), 5, slice(0, 5)),
        (slice(1, 3), 5, slice(1, 3)),
        (slice(1, 3, 1), 5, slice(1, 3)),
        (slice(2, 99), 5, slice(2, 5)),
        (slice(-2, None), 5, slice(3, 5)),
        (slice(4, 2), 5, slice(4, 4)),
        (0, 5, slice(0, 1)),
        (4, 5, slice(4, 5)),
        (-1, 5, slice(4, 5)),
        (np.int64(2), 5, slice(2, 3)),
    ],
)
def test__dim_slice(key, dim_len, expected):
    assert _dim_slice(key, dim_len, "baseline") == expected


@pytest.mark.parametrize(
    "key, error",
    [(5, IndexError), (-6, IndexError), (slice(0, 4, 2), ValueError), ("a", TypeError)],
)
def test__dim_slice_errors(key, error):
    with pytest.raises(error):
        _dim_slice(key, 5, "baseline")


@pytest.mark.parametrize(
    "key, expected",
    [
        (None, (0, None)),
        (slice(None), (0, None)),
        (slice(3, None), (3, None)),
        (slice(2, 5), (2, 5)),
        (slice(2, 5, 1), (2, 5)),
        (slice(5, 2), (5, 5)),
        (4, (4, 5)),
        (np.int32(0), (0, 1)),
    ],
)
def test__time_range(key, expected):
    assert _time_range(key) == expected


@pytest.mark.parametrize("key", [slice(0, 4, 2), slice(-1, None), -1])
def test__time_range_errors(key):
    with pytest.raises(ValueError):
        _time_range(key)


@pytest.mark.parametrize(
    "baseline_slice, cross_len, expected",
    [
        (slice(0, 9), 6, (slice(0, 6), slice(0, 3))),
        (slice(0, 6), 6, (slice(0, 6), None)),
        (slice(2, 4), 6, (slice(2, 4), None)),
        (slice(6, 9), 6, (None, slice(0, 3))),
        (slice(7, 8), 6, (None, slice(1, 2))),
        (slice(5, 7), 6, (slice(5, 6), slice(0, 1))),
        (slice(0, 3), 0, (None, slice(0, 3))),
        (slice(3, 3), 6, (None, None)),
    ],
)
def test__split_baseline_slice(baseline_slice, cross_len, expected):
    assert _split_baseline_slice(baseline_slice, cross_len) == expected


@pytest.mark.parametrize(
    "apc, expected",
    [
        ([], (0, 1)),
        (None, (0, 1)),
        (["AP_CORRECTED"], (0, 1)),
        (["AP_UNCORRECTED", "AP_CORRECTED"], (0, 2)),
        (["AP_CORRECTED", "AP_UNCORRECTED"], (1, 2)),
        (
            [
                pyasdm.enumerations.AtmPhaseCorrection.literal("AP_CORRECTED"),
                pyasdm.enumerations.AtmPhaseCorrection.literal("AP_UNCORRECTED"),
            ],
            (1, 2),
        ),
    ],
)
def test__find_apc_index(apc, expected):
    assert _find_apc_index(apc) == expected


def test__find_apc_index_without_uncorrected():
    with pytest.raises(NotImplementedError, match="AP_UNCORRECTED"):
        _find_apc_index(["AP_CORRECTED", "AP_MIXED"])


@pytest.mark.parametrize(
    "sd_pol, pol, expected",
    [(1, 1, [0]), (2, 2, [0, 1]), (3, 3, [0, 1, 2]), (3, 4, [0, 1, 1, 2])],
)
def test__auto_polarization_map(sd_pol, pol, expected):
    np.testing.assert_array_equal(_auto_polarization_map(sd_pol, pol), expected)


@pytest.mark.parametrize("sd_pol, pol", [(1, 2), (2, 4), (2, 1)])
def test__auto_polarization_map_unsupported(sd_pol, pol):
    with pytest.raises(ValueError, match="Unsupported polarization products"):
        _auto_polarization_map(sd_pol, pol)


# ---------------------------------------------------------------------------
# flags layout (F15)
# ---------------------------------------------------------------------------


def test__find_flags_layout_full_pol_per_spw():
    # Full pol: 4 cross products but 3 sd products. The auto block must use 3
    # products, and the (bb, spw) indices must be kept.
    shape = define_flag_shape(make_bdf_descr(num_antenna=3), (1, 1))
    size = 1 * 3 * 2 * 2 * 4 + 1 * 3 * 2 * 2 * 3
    shapes, idxs = _find_flags_layout(shape, size, (1, 1))
    assert shapes == shape
    assert idxs == (1, 1)


def test__find_flags_layout_per_baseband():
    shape = define_flag_shape(make_bdf_descr(cross_pol=2, sd_pol=2), (1, 1))
    size = 3 * 2 * 2 + 3 * 2 * 2
    shapes, idxs = _find_flags_layout(shape, size, (1, 1))
    assert shapes == {"cross": (1, 3, 2, 1, 2), "auto": (1, 3, 2, 1, 2)}
    assert idxs == (1, 0)


def test__find_flags_layout_all_basebands():
    shape = define_flag_shape(make_bdf_descr(cross_pol=2, sd_pol=2), (1, 1))
    size = 3 * 2 + 3 * 2
    shapes, idxs = _find_flags_layout(shape, size, (1, 1))
    assert shapes == {"cross": (1, 3, 1, 1, 2), "auto": (1, 3, 1, 1, 2)}
    assert idxs == (0, 0)


def test__find_flags_layout_auto_only_packed():
    shape = define_flag_shape(
        make_bdf_descr(correlation_mode=AUTO_ONLY, sd_pol=2, num_time=4), (1, 0)
    )
    shapes, idxs = _find_flags_layout(shape, 4 * 3 * 2 * 2 * 2, (1, 0))
    assert shapes == {"cross": (), "auto": (4, 3, 2, 2, 2)}
    assert idxs == (1, 0)


@pytest.mark.parametrize("size", [0, 1, 37, 4 * 6 * 2 * 2 * 4 + 1, 10**6])
def test__find_flags_layout_unexpected_size(size):
    shape = define_flag_shape(make_bdf_descr(num_antenna=4), (0, 0))
    with pytest.raises(ValueError, match="Unexpected size of the flags"):
        _find_flags_layout(shape, size, (0, 0))


# ---------------------------------------------------------------------------
# load_flags_all_subsets with mocked readers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_baseband, spws_per_baseband, cross_pol, sd_pol",
    [(4, 1, 4, 3), (2, 2, 4, 3), (2, 2, 2, 2), (1, 3, 1, 1)],
)
def test_load_flags_all_subsets_values_every_spw(
    num_baseband, spws_per_baseband, cross_pol, sd_pol
):
    num_antenna = 4
    num_baseline = 6
    bdf_descr = make_bdf_descr(
        num_antenna=num_antenna,
        num_baseband=num_baseband,
        spws_per_baseband=spws_per_baseband,
        cross_pol=cross_pol,
        sd_pol=sd_pol,
    )
    subsets_blocks = [
        make_flag_words(
            1,
            num_baseline,
            num_antenna,
            num_baseband,
            spws_per_baseband,
            cross_pol,
            sd_pol,
            offset=tim,
        )
        for tim in range(3)
    ]
    loaded_per_spw = []
    for bb_idx in range(num_baseband):
        for spw_idx in range(spws_per_baseband):
            shape = define_flag_shape(bdf_descr, (bb_idx, spw_idx))
            reader = mock_bdf_reader(
                [
                    {"flags": {"present": True, "arr": words.copy()}}
                    for words, _, _ in subsets_blocks
                ]
            )
            flags = load_flags_all_subsets(reader, shape, (bb_idx, spw_idx), ALL)

            expected = np.concatenate(
                [
                    expected_flags_from_blocks(cross, auto, bb_idx, spw_idx, cross_pol)
                    for _, cross, auto in subsets_blocks
                ]
            )
            assert flags.dtype == np.dtype(bool)
            assert flags.shape == (3, num_baseline + num_antenna, cross_pol)
            np.testing.assert_array_equal(flags, expected)
            for call in reader.getSubset.call_args_list:
                assert call.kwargs["loadOnlyComponents"] == {"flags"}
            loaded_per_spw.append(flags)

    # sanity check of the test data: the SPWs have different flags
    if len(loaded_per_spw) > 1:
        assert any(
            not np.array_equal(loaded_per_spw[0], other) for other in loaded_per_spw[1:]
        )
    assert any(flags.any() and not flags.all() for flags in loaded_per_spw)


def test_load_flags_all_subsets_nonzero_words_are_flagged():
    # F01: every non-zero bit pattern (including the sign bit) means flagged,
    # and the result is bool even for big-endian int32 words.
    bdf_descr = make_bdf_descr(
        num_antenna=2, num_baseband=1, spws_per_baseband=1, cross_pol=2, sd_pol=2
    )
    shape = define_flag_shape(bdf_descr, (0, 0))
    words = np.array([0, 1, 16, 2**30, -(2**31), 0], dtype=">i4")
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}])

    flags = load_flags_all_subsets(reader, shape, (0, 0), ALL)

    assert flags.dtype == np.dtype(bool)
    np.testing.assert_array_equal(flags, [[[False, True], [True, True], [True, False]]])
    np.testing.assert_array_equal(
        ~flags, [[[True, False], [False, False], [False, True]]]
    )


def test_load_flags_all_subsets_per_baseband_layout():
    # Flags given per baseband (axes BAL ANT BAB POL): both SPWs of a baseband
    # get the flags of their baseband.
    bdf_descr = make_bdf_descr(num_antenna=3, cross_pol=2, sd_pol=2)
    words, cross, auto = make_flag_words(1, 3, 3, 2, 1, 2, 2)
    for bb_idx in range(2):
        for spw_idx in range(2):
            shape = define_flag_shape(bdf_descr, (bb_idx, spw_idx))
            reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}])
            flags = load_flags_all_subsets(reader, shape, (bb_idx, spw_idx), ALL)
            expected = expected_flags_from_blocks(cross, auto, bb_idx, 0, 2)
            np.testing.assert_array_equal(flags, expected)
    assert not np.array_equal(
        expected_flags_from_blocks(cross, auto, 0, 0, 2),
        expected_flags_from_blocks(cross, auto, 1, 0, 2),
    )


def test_load_flags_all_subsets_single_flag_for_all_basebands():
    # Flags without baseband and SPW axes (axes BAL ANT POL)
    bdf_descr = make_bdf_descr(num_antenna=3, cross_pol=2, sd_pol=2)
    words, cross, auto = make_flag_words(1, 3, 3, 1, 1, 2, 2)
    shape = define_flag_shape(bdf_descr, (1, 1))
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}])
    flags = load_flags_all_subsets(reader, shape, (1, 1), ALL)
    np.testing.assert_array_equal(
        flags, expected_flags_from_blocks(cross, auto, 0, 0, 2)
    )


def test_load_flags_all_subsets_auto_only_3pol():
    # AUTO_ONLY with 3 sd products: 3 output products, no XY->YX expansion
    bdf_descr = make_bdf_descr(num_antenna=7, correlation_mode=AUTO_ONLY, sd_pol=3)
    words, _, auto = make_flag_words(1, 0, 7, 2, 2, 0, 3)
    shape = define_flag_shape(bdf_descr, (1, 0))
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}])
    flags = load_flags_all_subsets(reader, shape, (1, 0), ALL)
    assert flags.shape == (1, 7, 3)
    np.testing.assert_array_equal(flags, auto[:, :, 1, 0, :] != 0)


def test_load_flags_all_subsets_packed_time_selection():
    # Packed BDF: one subset with numTime=5 TIM samples, the time selection is
    # in integrations (TIM samples).
    bdf_descr = make_bdf_descr(
        num_antenna=3, correlation_mode=AUTO_ONLY, sd_pol=2, num_time=5
    )
    words, _, auto = make_flag_words(5, 0, 3, 2, 2, 0, 2)
    shape = define_flag_shape(bdf_descr, (0, 1))
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}])
    flags = load_flags_all_subsets(
        reader, shape, (0, 1), (slice(1, 4), slice(1, 3), slice(None), slice(1, 2))
    )
    np.testing.assert_array_equal(flags, auto[1:4, 1:3, 0, 1, 1:2] != 0)


@pytest.mark.parametrize(
    "array_slice, expected_index",
    [
        # (time, baseline, pol) indices into the full (4, 9, 4) expected array
        (
            (slice(1, 3), slice(0, 9), slice(0, 4), slice(0, 4)),
            np.s_[1:3, 0:9, 0:4],
        ),
        ((slice(2, 4), slice(4, 8), slice(0, 4), slice(1, 3)), np.s_[2:4, 4:8, 1:3]),
        # int keys (F16): the dimensions are kept
        ((slice(0, 4), slice(0, 9), slice(0, 4), 0), np.s_[0:4, 0:9, 0:1]),
        ((slice(0, 4), 2, slice(0, 4), 3), np.s_[0:4, 2:3, 3:4]),
        ((3, 7, 1, 2), np.s_[3:4, 7:8, 2:3]),
        # only autos / only cross baselines
        ((slice(0, 4), slice(6, 9), slice(0, 4), slice(0, 4)), np.s_[0:4, 6:9, 0:4]),
        ((slice(0, 4), slice(0, 6), slice(0, 4), slice(2, 4)), np.s_[0:4, 0:6, 2:4]),
    ],
)
def test_load_flags_all_subsets_selections(array_slice, expected_index):
    num_antenna = 4
    bdf_descr = make_bdf_descr(num_antenna=num_antenna)
    blocks = [make_flag_words(1, 6, 4, 2, 2, 4, 3, offset=tim) for tim in range(4)]
    expected_all = np.concatenate(
        [expected_flags_from_blocks(cross, auto, 1, 0, 4) for _, cross, auto in blocks]
    )
    shape = define_flag_shape(bdf_descr, (1, 0))
    reader = mock_bdf_reader(
        [{"flags": {"present": True, "arr": words}} for words, _, _ in blocks]
    )

    flags = load_flags_all_subsets(reader, shape, (1, 0), array_slice)

    assert flags.dtype == np.dtype(bool)
    assert flags.ndim == 3
    np.testing.assert_array_equal(flags, expected_all[expected_index])


def test_load_flags_all_subsets_skips_and_stops_reading():
    bdf_descr = make_bdf_descr(num_antenna=3, cross_pol=2, sd_pol=2)
    blocks = [make_flag_words(1, 3, 3, 2, 2, 2, 2, offset=tim) for tim in range(6)]
    shape = define_flag_shape(bdf_descr, (0, 1))
    reader = mock_bdf_reader(
        [{"flags": {"present": True, "arr": words}} for words, _, _ in blocks]
    )

    flags = load_flags_all_subsets(
        reader, shape, (0, 1), (slice(2, 4), slice(None), slice(None), slice(None))
    )

    expected = np.concatenate(
        [
            expected_flags_from_blocks(cross, auto, 0, 1, 2)
            for _, cross, auto in blocks[2:4]
        ]
    )
    np.testing.assert_array_equal(flags, expected)
    components = [
        call.kwargs["loadOnlyComponents"] for call in reader.getSubset.call_args_list
    ]
    # the binary components of the subsets before the selection are not read,
    # and the reading stops after the last selected integration
    assert components == [
        _SKIP_ALL_BINARY_COMPONENTS,
        _SKIP_ALL_BINARY_COMPONENTS,
        {"flags"},
        {"flags"},
    ]
    assert reader.hasSubset.call_count == 4


def test_load_flags_all_subsets_too_few_integrations():
    bdf_descr = make_bdf_descr(num_antenna=3, cross_pol=2, sd_pol=2)
    words, _, _ = make_flag_words(1, 3, 3, 2, 2, 2, 2)
    shape = define_flag_shape(bdf_descr, (0, 0))
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words}}] * 2)
    with pytest.raises(ValueError, match="has 2 integrations"):
        load_flags_all_subsets(
            reader, shape, (0, 0), (slice(1, 3), slice(None), slice(None), slice(None))
        )


def test_load_flags_all_subsets_absent_flags():
    # Subsets without flags are not flagged (selected shape, bool)
    bdf_descr = make_bdf_descr(num_antenna=5)
    shape = define_flag_shape(bdf_descr, (1, 1))
    reader = mock_bdf_reader([{"flags": {"present": False, "arr": None}}, {}])
    flags = load_flags_all_subsets(
        reader, shape, (1, 1), (slice(None), slice(8, 13), slice(None), slice(1, 3))
    )
    assert flags.dtype == np.dtype(bool)
    assert flags.shape == (2, 5, 2)
    assert not flags.any()


def test_load_flags_all_subsets_unexpected_size():
    # F11/F15: sizes that match no supported layout raise (no silent truncation)
    bdf_descr = make_bdf_descr(num_antenna=3)
    words, _, _ = make_flag_words(1, 3, 3, 2, 2, 4, 3)
    shape = define_flag_shape(bdf_descr, (0, 0))
    reader = mock_bdf_reader([{"flags": {"present": True, "arr": words[:-1]}}])
    with pytest.raises(ValueError, match="Unexpected size of the flags"):
        load_flags_all_subsets(reader, shape, (0, 0), ALL)


@pytest.mark.parametrize(
    "exception",
    [pyasdm.exceptions.BDFReaderException("broken MIME part"), ValueError("bad int")],
)
def test_load_flags_all_subsets_read_error(exception):
    # F61: errors are raised (naming the BDF), never replaced by None
    bdf_descr = make_bdf_descr(num_antenna=3)
    shape = define_flag_shape(bdf_descr, (0, 0))
    reader = mock_bdf_reader([exception], path="/asdm/ASDMBinary/uid_broken")
    with pytest.raises(RuntimeError, match="uid_broken") as exc_info:
        load_flags_all_subsets(reader, shape, (0, 0), ALL)
    assert exc_info.value.__cause__ is exception


def test_load_flags_all_subsets_unsupported_polarizations():
    shape = {"cross": (1, 3, 1, 1, 2), "auto": (1, 3, 1, 1, 1)}
    reader = mock_bdf_reader([])
    with pytest.raises(ValueError, match="Unsupported polarization products"):
        load_flags_all_subsets(reader, shape, (0, 0), ALL)


# ---------------------------------------------------------------------------
# visibility decoders with hand-built arrays
# ---------------------------------------------------------------------------


def test__load_vis_subset_auto_data_full_pol_cross_and_auto():
    # F18: 3 sd products are stored as XX, Re(XY), Im(XY), YY. With 4 output
    # products (CROSS_AND_AUTO) they are [XX, XY, conj(XY), YY].
    guessed_shape = (1, 1, 2, 1, 1, 2, 4, 2)
    raw = np.array(
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]  # antenna 0, channels 0 and 1
        + [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5],  # antenna 1
        dtype=np.float32,
    )
    vis = _load_vis_subset_auto_data(
        raw,
        guessed_shape,
        (0, 0),
        3,
        (0, 1),
        (slice(0, 1), slice(0, 2), slice(0, 2), slice(0, 4)),
    )
    assert vis.dtype == np.complex64
    np.testing.assert_array_equal(vis[0, 0, 0], [0.5, 1.5 + 2.5j, 1.5 - 2.5j, 3.5])
    np.testing.assert_array_equal(
        vis[0, 1, 1], [14.5, 15.5 + 16.5j, 15.5 - 16.5j, 17.5]
    )


def test__load_vis_subset_auto_data_full_pol_auto_only():
    # AUTO_ONLY with 3 sd products: [XX, XY, YY]
    guessed_shape = (1, 0, 1, 1, 1, 1, 3, 2)
    raw = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    vis = _load_vis_subset_auto_data(
        raw,
        guessed_shape,
        (0, 0),
        3,
        (0, 1),
        (slice(0, 1), slice(0, 1), slice(0, 1), slice(0, 3)),
    )
    np.testing.assert_array_equal(vis, [[[[0.5, 1.5 + 2.5j, 3.5]]]])
    # selection of the polarization products
    vis = _load_vis_subset_auto_data(
        raw,
        guessed_shape,
        (0, 0),
        3,
        (0, 1),
        (slice(0, 1), slice(0, 1), slice(0, 1), slice(1, 2)),
    )
    np.testing.assert_array_equal(vis, [[[[1.5 + 2.5j]]]])


def test__load_vis_subset_auto_data_dual_pol_layout():
    # (ANT, BAB, SPW, SPP, POL) layout, select baseband 1, SPW 0
    guessed_shape = (1, 1, 2, 2, 2, 3, 2, 2)
    raw = np.arange(2 * 2 * 2 * 3 * 2, dtype=np.float32)
    vis = _load_vis_subset_auto_data(
        raw,
        guessed_shape,
        (1, 0),
        2,
        (0, 1),
        (slice(0, 1), slice(1, 2), slice(1, 3), slice(0, 2)),
    )
    expected = np.empty((1, 1, 2, 2))
    for chan_idx, chan in enumerate([1, 2]):
        for pol in range(2):
            # antenna 1, baseband 1, spw 0
            expected[0, 0, chan_idx, pol] = (((1 * 2 + 1) * 2 + 0) * 3 + chan) * 2 + pol
    np.testing.assert_array_equal(vis, expected)
    assert vis.dtype == np.complex64


def test__load_vis_subset_cross_data_apc_and_scale():
    # crossData BAL BAB SPW APC SPP POL (re, im), INT32 with scale factor,
    # select the second APC value
    guessed_shape = (1, 3, 3, 1, 2, 2, 2, 2)
    shape = (1, 3, 1, 2, 2, 2, 2, 2)
    raw = np.arange(np.prod(shape), dtype=np.int32)
    vis = _load_vis_subset_cross_data(
        raw,
        guessed_shape,
        (0, 1),
        (1, 2),
        np.float32(4.0),
        (slice(0, 1), slice(1, 3), slice(0, 2), slice(1, 2)),
    )
    full = raw.reshape(shape)
    expected = (full[..., 0] + 1j * full[..., 1])[0, 1:3, 0, 1, 1, 0:2, 1:2] / 4.0
    assert vis.dtype == np.complex64
    np.testing.assert_array_equal(vis, expected[np.newaxis])


def test__load_vis_subset_cross_data_float_not_scaled():
    guessed_shape = (1, 1, 2, 1, 1, 1, 1, 2)
    raw = np.array([3.0, -5.0], dtype=np.float32)
    vis = _load_vis_subset_cross_data(
        raw, guessed_shape, (0, 0), (0, 1), 7.0, (slice(0, 1),) * 4
    )
    np.testing.assert_array_equal(vis, [[[[3.0 - 5.0j]]]])


def test__load_vis_subset_cross_data_real_values():
    # real-valued crossData (data not from the CORRELATOR): imaginary part 0
    guessed_shape = (1, 2, 3, 1, 1, 2, 1, 2)
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    vis = _load_vis_subset_cross_data(
        raw,
        guessed_shape,
        (0, 0),
        (0, 1),
        1.0,
        (slice(0, 1), slice(1, 2), slice(0, 2), slice(0, 1)),
        real_values_allowed=True,
    )
    np.testing.assert_array_equal(vis, [[[[3.0 + 0j], [4.0 + 0j]]]])
    # not accepted for CORRELATOR data
    with pytest.raises(ValueError, match="Unexpected size of the crossData"):
        _load_vis_subset_cross_data(
            raw, guessed_shape, (0, 0), (0, 1), 1.0, (slice(0, 1),) * 4
        )


def test__load_vis_subset_cross_data_size_mismatch():
    # F11: data with 2 APCs while the header says 1 must not be truncated
    guessed_shape = (1, 3, 3, 1, 1, 4, 2, 2)
    raw = np.zeros(3 * 2 * 4 * 2 * 2, dtype=np.float32)
    with pytest.raises(ValueError, match="Unexpected size of the crossData"):
        _load_vis_subset_cross_data(
            raw, guessed_shape, (0, 0), (0, 1), 1.0, (slice(0, 1),) * 4
        )


def test__load_vis_subset_cross_data_integer_without_scale_factor():
    with pytest.raises(ValueError, match="scaleFactor"):
        _load_vis_subset_cross_data(
            np.zeros(2, dtype=np.int16),
            (1, 1, 2, 1, 1, 1, 1, 2),
            (0, 0),
            (0, 1),
            None,
            (slice(0, 1),) * 4,
        )


# ---------------------------------------------------------------------------
# load_visibilities_all_subsets with mocked readers: errors and reading
# ---------------------------------------------------------------------------


def _vis_subset(bdf_descr, value=0):
    """INT32 crossData / autoData arrays of the right sizes for a (regular) descriptor."""
    shape = define_visibility_shape(bdf_descr, (0, 0))
    tims, nbl, nant, nbb, nspw, nchan, npol = shape[:7]
    napc = max(len(bdf_descr["apc"]), 1)
    nsd = len(bdf_descr["basebands"][0]["spectralWindows"][0]["sdPolProducts"])
    nvals = 4 if nsd == 3 else nsd
    return {
        "crossData": {
            "present": True,
            "arr": np.full(
                tims * nbl * nbb * nspw * napc * nchan * npol * 2, value, np.int32
            ),
        },
        "autoData": {
            "present": True,
            "arr": np.full(tims * nant * nbb * nspw * nchan * nvals, value, np.float32),
        },
    }


def test_load_visibilities_all_subsets_auto_data_missing():
    bdf_descr = make_bdf_descr()
    subset = _vis_subset(bdf_descr)
    subset["autoData"] = {"present": False, "arr": None}
    reader = mock_bdf_reader([subset])
    with pytest.raises(ValueError, match="'autoData' not present"):
        load_visibilities_all_subsets(
            reader, define_visibility_shape(bdf_descr, (0, 0)), (0, 0), bdf_descr, ALL
        )


@pytest.mark.parametrize(
    "exception",
    [pyasdm.exceptions.BDFReaderException("broken MIME part"), ValueError("bad int")],
)
def test_load_visibilities_all_subsets_read_error(exception):
    # F61: errors are raised (naming the BDF), never replaced by None
    bdf_descr = make_bdf_descr()
    reader = mock_bdf_reader([exception], path="/asdm/ASDMBinary/uid_broken")
    with pytest.raises(RuntimeError, match="uid_broken") as exc_info:
        load_visibilities_all_subsets(
            reader, define_visibility_shape(bdf_descr, (0, 0)), (0, 0), bdf_descr, ALL
        )
    assert exc_info.value.__cause__ is exception


def test_load_visibilities_all_subsets_two_apc_data_but_one_in_header():
    # F11: the crossData size is validated against the header (APC axis)
    with_apc = make_bdf_descr(apc=("AP_UNCORRECTED", "AP_CORRECTED"))
    bdf_descr = make_bdf_descr(apc=("AP_UNCORRECTED",))
    reader = mock_bdf_reader([_vis_subset(with_apc)])
    with pytest.raises(ValueError, match="Unexpected size of the crossData"):
        load_visibilities_all_subsets(
            reader, define_visibility_shape(bdf_descr, (0, 0)), (0, 0), bdf_descr, ALL
        )


def test_load_visibilities_all_subsets_num_bin():
    # numBin > 1 in any SPW changes the layout of the other SPWs too
    bdf_descr = make_bdf_descr()
    bdf_descr["basebands"][1]["spectralWindows"][1]["numBin"] = 2
    reader = mock_bdf_reader([_vis_subset(make_bdf_descr())])
    with pytest.raises(NotImplementedError, match="numBin > 1"):
        load_visibilities_all_subsets(
            reader, define_visibility_shape(bdf_descr, (0, 0)), (0, 0), bdf_descr, ALL
        )
    reader.getSubset.assert_not_called()


def test_load_visibilities_all_subsets_irregular_layout():
    bdf_descr = make_bdf_descr()
    bdf_descr["basebands"][1]["spectralWindows"][0]["numSpectralPoint"] = 8
    reader = mock_bdf_reader([_vis_subset(make_bdf_descr())])
    with pytest.raises(ValueError, match="same number"):
        load_visibilities_all_subsets(
            reader, define_visibility_shape(bdf_descr, (0, 0)), (0, 0), bdf_descr, ALL
        )


def test_load_visibilities_all_subsets_step_not_supported():
    bdf_descr = make_bdf_descr()
    reader = mock_bdf_reader([_vis_subset(bdf_descr)])
    with pytest.raises(ValueError, match="only steps of 1"):
        load_visibilities_all_subsets(
            reader,
            define_visibility_shape(bdf_descr, (0, 0)),
            (0, 0),
            bdf_descr,
            (slice(None), slice(0, 9, 2), slice(None), slice(None)),
        )


def test_load_visibilities_all_subsets_reads_only_needed_components():
    bdf_descr = make_bdf_descr(num_antenna=3)
    shape = define_visibility_shape(bdf_descr, (0, 0))

    # only cross baselines selected: autoData not read
    reader = mock_bdf_reader([_vis_subset(bdf_descr, 1)] * 3)
    vis = load_visibilities_all_subsets(
        reader, shape, (0, 0), bdf_descr, (slice(1, 2), slice(0, 3), 1, slice(None))
    )
    assert vis.shape == (1, 3, 1, 4)
    np.testing.assert_array_equal(vis, np.full((1, 3, 1, 4), 0.25 + 0.25j))
    assert [
        call.kwargs["loadOnlyComponents"] for call in reader.getSubset.call_args_list
    ] == [_SKIP_ALL_BINARY_COMPONENTS, {"crossData"}]
    assert reader.hasSubset.call_count == 2

    # only autos selected: crossData not read
    reader = mock_bdf_reader([_vis_subset(bdf_descr, 1)] * 3)
    vis = load_visibilities_all_subsets(
        reader, shape, (0, 0), bdf_descr, (slice(None), slice(3, 6), 0, 0)
    )
    assert vis.shape == (3, 3, 1, 1)
    assert [
        call.kwargs["loadOnlyComponents"] for call in reader.getSubset.call_args_list
    ] == [{"autoData"}] * 3


# ---------------------------------------------------------------------------
# Values from real (synthetic) BDFs read with pyasdm
# ---------------------------------------------------------------------------

SELECTIONS = [
    ALL,
    # time slice not starting at 0 (F60), baselines across the cross/auto
    # boundary, frequency and polarization ranges
    (slice(1, 3), slice(1, None), slice(1, 3), slice(1, None)),
    # int keys keep their dimension (F16)
    (2, slice(None), slice(None), 0),
    (slice(0, 3), 1, 0, slice(None)),
    (1, -1, -1, -1),
]


@pytest.mark.parametrize("spec_name", list(SYNTHETIC_SPECS))
@pytest.mark.parametrize("selection_idx", range(len(SELECTIONS)))
def test_values_from_synthetic_bdfs(synthetic_truths, spec_name, selection_idx):
    truth = synthetic_truths[spec_name]
    array_slice = SELECTIONS[selection_idx]
    checked = 0
    for bdf in truth.bdfs:
        for spw in truth.spws:
            if spw.config_idx != bdf.config_idx:
                continue
            expected_vis, expected_flags = expected_values(truth, spw.spw_id, bdf)
            index = tuple(
                slice(key, key + 1 if key != -1 else None)
                if isinstance(key, int)
                else key
                for key in array_slice
            )

            vis = load_from_synthetic_bdf(bdf.path, spw, "vis", array_slice)
            flags = load_from_synthetic_bdf(bdf.path, spw, "flags", array_slice)

            assert vis.dtype == np.complex64
            assert vis.shape == expected_vis[index].shape
            np.testing.assert_allclose(vis, expected_vis[index], rtol=1e-6, atol=0)
            assert flags.dtype == np.dtype(bool)
            flags_index = (index[0], index[1], index[3])
            np.testing.assert_array_equal(flags, expected_flags[flags_index])
            checked += 1
    assert checked > 0


def test_synthetic_full_pol_spws_are_distinct(synthetic_truths):
    # The data of every SPW are different, so a wrong SPW selection fails
    truth = synthetic_truths["full_pol_int32_2apc"]
    bdf = truth.bdfs[0]
    loaded = [load_from_synthetic_bdf(bdf.path, spw, "vis", ALL) for spw in truth.spws]
    flags = [load_from_synthetic_bdf(bdf.path, spw, "flags", ALL) for spw in truth.spws]
    assert len(loaded) == 4
    for idx in range(1, 4):
        assert not np.array_equal(loaded[0], loaded[idx])
        assert not np.array_equal(flags[0], flags[idx])
    # full-pol autos: YX == conj(XY), and they are not real-valued
    autos = loaded[0][:, 3:]
    np.testing.assert_array_equal(autos[..., 2], np.conj(autos[..., 1]))
    assert np.all(autos[..., 1].imag != 0)


def test_synthetic_apc_uncorrected_selected(synthetic_truths):
    # K6: the AP_UNCORRECTED block is loaded whatever its position in the APC
    # list; cross-checked with the writer's raw-array decoder.
    truth = synthetic_truths["dual_pol_int16_apc_reversed"]
    bdf = truth.bdfs[0]
    reader = pyasdm.bdf.BDFReader()
    reader.open(bdf.path)
    try:
        raw = reader.getSubset()["crossData"]["arr"]
    finally:
        reader.close()
    for spw in truth.spws:
        expected = synth.decode_raw_cross_subset(truth, bdf, raw, spw.spw_id)
        vis = load_from_synthetic_bdf(
            bdf.path, spw, "vis", (slice(0, 1), slice(0, 6), slice(None), slice(None))
        )
        np.testing.assert_allclose(vis[0], expected, rtol=1e-6)


def test_synthetic_num_bin_not_supported(make_synthetic_asdm):
    truth = make_synthetic_asdm(
        synth.small_dtype_spec("FLOAT32_TYPE", 1.0, num_bin=2, name="uid___A002_X1_X56")
    )
    bdf = truth.bdfs[0]
    spw = truth.spws[0]
    with pytest.raises(NotImplementedError, match="numBin > 1"):
        load_from_synthetic_bdf(bdf.path, spw, "vis", ALL)

    # the flags have no BIN axis and are still loaded correctly
    _, expected_flags = expected_values(truth, spw.spw_id, bdf)
    flags = load_from_synthetic_bdf(bdf.path, spw, "flags", ALL)
    np.testing.assert_array_equal(flags, expected_flags)
