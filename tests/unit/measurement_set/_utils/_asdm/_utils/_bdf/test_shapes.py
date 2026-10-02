from contextlib import nullcontext as no_raises
from unittest import mock

import numpy as np
import pyasdm
import pytest

from xradio.measurement_set._utils._asdm._utils._bdf import shapes
from xradio.measurement_set._utils._asdm._utils._bdf.shapes import (
    BDFSelection,
    calc_bdf_spw_layout,
    check_data_axes,
    normalize_bdf_selection,
    num_auto_values_per_channel,
    select_apc_index,
    times_per_subset,
)

S = pyasdm.enumerations.StokesParameter
DUAL = [S.XX, S.YY]
FULL = [S.XX, S.XY, S.YX, S.YY]
SD_FULL = [S.XX, S.XY, S.YY]
AP_UNCORRECTED = pyasdm.enumerations.AtmPhaseCorrection.AP_UNCORRECTED
AP_CORRECTED = pyasdm.enumerations.AtmPhaseCorrection.AP_CORRECTED


def _spw(nchan, cross=DUAL, sd=DUAL, num_bin=1, scale_factor=2.0):
    return {
        "crossPolProducts": list(cross),
        "sdPolProducts": list(sd),
        "scaleFactor": scale_factor,
        "numSpectralPoint": nchan,
        "numBin": num_bin,
        "sideband": pyasdm.enumerations.NetSideband.USB,
        "sw": "1",
    }


def _descr(basebands, num_antenna=4, mode="CROSS_AND_AUTO", apc=None, **kwargs):
    descr = {
        "dimensionality": 1,
        "num_time": 0,
        "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
        "correlation_mode": pyasdm.enumerations.CorrelationMode.literal(mode),
        "apc": [AP_UNCORRECTED] if apc is None else apc,
        "num_antenna": num_antenna,
        "basebands": [
            {"name": f"BB_{idx + 1}", "spectralWindows": spws}
            for idx, spws in enumerate(basebands)
        ],
    }
    descr.update(kwargs)
    return descr


@pytest.mark.parametrize("num_sd_pols, expected", [(1, 1), (2, 2), (3, 4), (4, 4)])
def test_num_auto_values_per_channel(num_sd_pols, expected):
    assert num_auto_values_per_channel(num_sd_pols) == expected


@pytest.mark.parametrize(
    "dimensionality, num_time, expected",
    [
        (1, 0, no_raises(1)),
        (1, 7, no_raises(1)),
        (0, 34, no_raises(34)),
        (0, 0, pytest.raises(ValueError, match="numTime")),
    ],
)
def test_times_per_subset(dimensionality, num_time, expected):
    with expected as value:
        assert (
            times_per_subset({"dimensionality": dimensionality, "num_time": num_time})
            == value
        )


def test_calc_bdf_spw_layout_uneven_spws_and_pols():
    # BB_1: [8 chan dual, 1 chan dual], BB_2: [4 chan full-pol]
    descr = _descr(
        [[_spw(8), _spw(1)], [_spw(4, FULL, SD_FULL, scale_factor=4.0)]],
        num_antenna=4,
    )
    cross_row_len = 8 * 2 * 2 + 1 * 2 * 2 + 4 * 4 * 2
    auto_row_len = 8 * 2 + 1 * 2 + 4 * 4

    layout = calc_bdf_spw_layout(descr, (0, 1))
    assert layout.num_cross_baselines == 6
    assert layout.num_antennas == 4
    assert layout.num_baselines == 10
    assert layout.times_per_subset == 1
    assert (layout.num_channels, layout.num_polarizations) == (1, 2)
    assert layout.cross_row_len == cross_row_len
    assert layout.cross_spw_offset == 8 * 2 * 2
    assert layout.auto_row_len == auto_row_len
    assert layout.auto_spw_offset == 8 * 2
    assert layout.cross_size == 6 * cross_row_len
    assert layout.auto_size == 4 * auto_row_len
    assert layout.scale_factor == 2.0
    assert not layout.real_cross_data_allowed

    layout = calc_bdf_spw_layout(descr, (1, 0))
    assert layout.cross_spw_offset == 8 * 2 * 2 + 1 * 2 * 2
    assert layout.auto_spw_offset == 8 * 2 + 1 * 2
    assert (layout.num_cross_pols, layout.num_sd_pols) == (4, 3)
    assert layout.num_polarizations == 4
    assert layout.num_auto_values == 4
    assert layout.scale_factor == 4.0


@pytest.mark.parametrize(
    "apc, bb_spw, expected_apc_index, expected_offset",
    [
        # BIN APC SPP POL order: the AP_UNCORRECTED block of the SPW
        ([AP_UNCORRECTED, AP_CORRECTED], (0, 0), 0, 0),
        ([AP_CORRECTED, AP_UNCORRECTED], (0, 0), 1, 3 * 2 * 2),
        ([AP_CORRECTED, AP_UNCORRECTED], (0, 1), 1, 2 * 3 * 2 * 2 + 5 * 2 * 2),
        # a single (non-list) APC value is also accepted
        (AP_UNCORRECTED, (0, 1), 0, 3 * 2 * 2),
    ],
)
def test_calc_bdf_spw_layout_apc(apc, bb_spw, expected_apc_index, expected_offset):
    descr = _descr([[_spw(3), _spw(5)]], num_antenna=3, apc=apc)
    num_apc = len(apc) if isinstance(apc, list) else 1

    layout = calc_bdf_spw_layout(descr, bb_spw)
    assert layout.apc_index == expected_apc_index
    assert layout.cross_spw_offset == expected_offset
    assert layout.cross_row_len == num_apc * (3 + 5) * 2 * 2
    # autoData have no APC axis
    assert layout.auto_row_len == (3 + 5) * 2


def test_calc_bdf_spw_layout_single_corrected_apc_loaded_without_warning(
    monkeypatch,
):
    """The only APC is loaded whatever it is. The layout, calculated for every BDF
    and chunk loaded, does not warn (the warning was repeated for every binary
    component of every subset)."""
    warnings = []
    monkeypatch.setattr(
        shapes,
        "xradio_logger",
        lambda: mock.Mock(warning=lambda msg, *args, **kwargs: warnings.append(msg)),
    )
    descr = _descr([[_spw(3)]], num_antenna=3, apc=[AP_CORRECTED])
    layout = calc_bdf_spw_layout(descr, (0, 0))
    assert layout.apc_index == 0
    assert layout.cross_spw_offset == 0
    assert warnings == []


@pytest.mark.parametrize(
    "apc, warn, expected_index, expected_warnings",
    [
        ([AP_CORRECTED], True, 0, 1),
        ([AP_CORRECTED], False, 0, 0),
        (["AP_CORRECTED"], True, 0, 1),
        ([AP_UNCORRECTED], True, 0, 0),
        ([AP_CORRECTED, AP_UNCORRECTED], True, 1, 0),
        ([], True, 0, 0),
        (None, True, 0, 0),
    ],
)
def test_select_apc_index_warning(
    monkeypatch, apc, warn, expected_index, expected_warnings
):
    """select_apc_index warns (when asked to) only when the only APC is not
    AP_UNCORRECTED."""
    warnings = []
    monkeypatch.setattr(
        shapes,
        "xradio_logger",
        lambda: mock.Mock(warning=lambda msg, *args, **kwargs: warnings.append(msg)),
    )
    assert select_apc_index(apc, warn=warn) == expected_index
    assert len(warnings) == expected_warnings
    if expected_warnings:
        assert "AP_CORRECTED" in warnings[0]


def test_calc_bdf_spw_layout_apc_without_uncorrected_raises():
    apc = [AP_CORRECTED, AP_CORRECTED]
    descr = _descr([[_spw(3)]], num_antenna=3, apc=apc)
    with pytest.raises(NotImplementedError, match="AP_UNCORRECTED"):
        calc_bdf_spw_layout(descr, (0, 0))


def test_calc_bdf_spw_layout_num_bin():
    descr = _descr([[_spw(3, num_bin=2), _spw(5)]], num_antenna=3)
    with pytest.raises(NotImplementedError, match="numBin"):
        calc_bdf_spw_layout(descr, (0, 0))

    # Bins of other SPWs are included in the row lengths and offsets
    layout = calc_bdf_spw_layout(descr, (0, 1))
    assert layout.cross_spw_offset == 2 * 3 * 2 * 2
    assert layout.cross_row_len == 2 * 3 * 2 * 2 + 5 * 2 * 2
    assert layout.auto_spw_offset == 2 * 3 * 2
    assert layout.auto_row_len == 2 * 3 * 2 + 5 * 2


def test_calc_bdf_spw_layout_auto_only_packed():
    # Like uid___A002_Xac5575_X4089 (WVR): packed, 1 sdPolProduct (I), 4 channels
    descr = _descr(
        [[_spw(4, cross=[], sd=[S.I], scale_factor=None)]],
        num_antenna=43,
        mode="AUTO_ONLY",
        apc=[],
        dimensionality=0,
        num_time=34,
        processor_type=pyasdm.enumerations.ProcessorType.RADIOMETER,
    )
    layout = calc_bdf_spw_layout(descr, (0, 0))
    assert layout.times_per_subset == 34
    assert layout.num_cross_baselines == 0
    assert layout.num_baselines == 43
    assert layout.num_polarizations == 1
    assert layout.auto_row_len == 4
    assert layout.auto_size == 34 * 43 * 4
    assert layout.cross_size == 0
    assert layout.scale_factor == 1.0
    assert layout.real_cross_data_allowed


def test_calc_bdf_spw_layout_auto_only_full_pol():
    descr = _descr([[_spw(5, cross=[], sd=SD_FULL)]], mode="AUTO_ONLY", apc=[])
    layout = calc_bdf_spw_layout(descr, (0, 0))
    assert layout.num_polarizations == 3
    assert layout.num_auto_values == 4
    assert layout.auto_row_len == 5 * 4


@pytest.mark.parametrize(
    "descr, bb_spw, expected_error",
    [
        (
            _descr([[_spw(3)]], mode="CROSS_ONLY"),
            (0, 0),
            pytest.raises(ValueError, match="correlation mode"),
        ),
        (
            _descr([[_spw(3, cross=FULL, sd=DUAL)]]),
            (0, 0),
            pytest.raises(ValueError, match="crossPolProducts"),
        ),
        (
            _descr([[_spw(3)]]),
            (0, 1),
            pytest.raises(ValueError, match="not found"),
        ),
        (
            _descr([[_spw(3, cross=[], sd=[])]], mode="AUTO_ONLY"),
            (0, 0),
            pytest.raises(ValueError, match="polarization"),
        ),
        (
            _descr([[_spw(3)]], axes={"crossData": ["BAL", "BAB", "SPW", "SIB"]}),
            (0, 0),
            pytest.raises(NotImplementedError, match="SIB"),
        ),
    ],
)
def test_calc_bdf_spw_layout_errors(descr, bb_spw, expected_error):
    with expected_error:
        calc_bdf_spw_layout(descr, bb_spw)


@pytest.mark.parametrize(
    "axes, expected_error",
    [
        (None, no_raises()),
        (
            {
                "crossData": ["BAL", "BAB", "SPW", "APC", "SPP", "POL"],
                "autoData": ["ANT", "BAB", "SPW", "SPP", "POL"],
            },
            no_raises(),
        ),
        (
            {
                "crossData": ["TIM", "BAL", "BAB", "SPW", "BIN", "APC", "SPP", "POL"],
                "autoData": ["TIM", "ANT", "BAB", "SPW", "BIN", "SPP", "POL"],
            },
            no_raises(),
        ),
        (
            {"autoData": ["ANT", "BAB", "SPW", "SPP", "STO"]},
            pytest.raises(NotImplementedError, match="STO"),
        ),
        (
            {"crossData": ["BAL", "BAB", "SPW", "SUB", "SPP", "POL"]},
            pytest.raises(NotImplementedError, match="SUB"),
        ),
        (
            {"autoData": ["ANT", "BAB", "SPW", "APC", "SPP", "POL"]},
            pytest.raises(NotImplementedError, match="APC"),
        ),
        (
            {"crossData": ["BAL", "BAB", "SPW", "SPP", "APC", "POL"]},
            pytest.raises(NotImplementedError, match="order"),
        ),
    ],
)
def test_check_data_axes(axes, expected_error):
    with expected_error:
        check_data_axes(axes)


@pytest.fixture
def layout_4ant_8chan_full_pol():
    # 6 cross baselines + 4 autos, 8 channels, 4 polarizations
    return calc_bdf_spw_layout(
        _descr([[_spw(8, FULL, SD_FULL)]], num_antenna=4), (0, 0)
    )


@pytest.mark.parametrize(
    "array_slice, time_len, expected",
    [
        (None, 1, ((0, 1), (0, 10), (0, 8), (0, 4))),
        ((), None, ((0, None), (0, 10), (0, 8), (0, 4))),
        (
            (slice(None), slice(None), slice(None), slice(None)),
            None,
            ((0, None), (0, 10), (0, 8), (0, 4)),
        ),
        (
            (slice(2, 5, 1), slice(3, 7), slice(1, 100), slice(1, 3)),
            None,
            ((2, 5), (3, 7), (1, 8), (1, 3)),
        ),
        ((3, -1, np.int64(2), -4), 4, ((3, 4), (9, 10), (2, 3), (0, 1))),
        ((slice(1, None),), 3, ((1, 3), (0, 10), (0, 8), (0, 4))),
        (
            (slice(None), slice(-4, None), slice(None, -2), slice(None)),
            None,
            ((0, None), (6, 10), (0, 6), (0, 4)),
        ),
    ],
)
def test_normalize_bdf_selection(
    layout_4ant_8chan_full_pol, array_slice, time_len, expected
):
    selection = normalize_bdf_selection(
        array_slice, layout_4ant_8chan_full_pol, time_len=time_len
    )
    assert isinstance(selection, BDFSelection)
    assert (
        selection.time,
        selection.baseline,
        selection.frequency,
        selection.polarization,
    ) == expected


@pytest.mark.parametrize(
    "array_slice, expected_error",
    [
        ((slice(0, 4, 2),), pytest.raises(ValueError, match="step 1")),
        ((slice(None), slice(None, None, -1)), pytest.raises(ValueError, match="step")),
        ((slice(2, 2),), pytest.raises(ValueError, match="Empty time")),
        ((slice(None), slice(5, 3)), pytest.raises(ValueError, match="Empty baseline")),
        ((slice(None), 10), pytest.raises(IndexError, match="baseline index")),
        ((slice(None), slice(None), -9), pytest.raises(IndexError, match="frequency")),
        ((-1,), pytest.raises(ValueError, match="Negative time")),
        ((slice(-2, None),), pytest.raises(ValueError, match="Negative time")),
        ((slice(None), [0, 1]), pytest.raises(TypeError, match="baseline")),
        ((slice(None),) * 5, pytest.raises(ValueError, match="4 dimensions")),
    ],
)
def test_normalize_bdf_selection_errors(
    layout_4ant_8chan_full_pol, array_slice, expected_error
):
    with expected_error:
        normalize_bdf_selection(array_slice, layout_4ant_8chan_full_pol)


@pytest.mark.parametrize(
    "baseline, expected_cross, expected_auto",
    [
        ((0, 10), (0, 6), (0, 4)),
        ((0, 6), (0, 6), None),
        ((2, 4), (2, 4), None),
        ((5, 7), (5, 6), (0, 1)),
        ((6, 10), None, (0, 4)),
        ((8, 9), None, (2, 3)),
    ],
)
def test_bdf_selection_cross_auto_ranges(baseline, expected_cross, expected_auto):
    selection = BDFSelection((0, 1), baseline, (0, 1), (0, 1))
    assert selection.cross_range(6) == expected_cross
    assert selection.auto_range(6) == expected_auto


def test_bdf_selection_auto_only_ranges():
    selection = BDFSelection((0, 1), (1, 3), (0, 1), (0, 1))
    assert selection.cross_range(0) is None
    assert selection.auto_range(0) == (1, 3)
