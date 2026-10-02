import pickle
from contextlib import nullcontext as no_raises

import pyasdm
import pytest


@pytest.mark.parametrize(
    "input_correlation_mode, expected_error",
    [
        (
            pyasdm.enumerations.CorrelationMode.CROSS_ONLY,
            pytest.raises(
                RuntimeError, match="Unsupported correlation mode CROSS_ONLY"
            ),
        ),
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, no_raises()),
        (pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO, no_raises()),
    ],
)
def test_check_correlation_mode(input_correlation_mode, expected_error):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        check_correlation_mode,
    )

    with expected_error:
        check_correlation_mode(input_correlation_mode)


@pytest.mark.parametrize(
    "data_array_names, binary_types, bdf_path, expected_error",
    [
        ([], [], "foo_non_existent.nope", no_raises()),
        (
            ["flags"],
            [],
            "foo_non_existent.nope",
            pytest.raises(RuntimeError, match="does not have"),
        ),
        (
            ["flags"],
            ["actualTimes", "actualDurations", "flags", "autoData", "crossData"],
            "foo_non_existent.nope",
            no_raises(),
        ),
        (
            ["crossData", "autoData"],
            [],
            "foo_non_existent.nope",
            pytest.raises(RuntimeError, match="does not have"),
        ),
    ],
)
def test_ensure_presence_binary_components(
    data_array_names, binary_types, bdf_path, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        ensure_presence_binary_components,
    )

    with expected_error:
        ensure_presence_binary_components(data_array_names, binary_types, bdf_path)


@pytest.mark.parametrize(
    "input_names, exclude_also_for_flags, expected_error",
    [
        ([], False, no_raises()),
        (
            ["TIM", "BAL", "ANT", "BAB", "SPW", "BIN", "APC", "SPP", "POL", "ANY"],
            False,
            no_raises(),
        ),
        (
            ["TIM", "BAL", "ANT", "BAB", "SPW", "BIN", "APC", "SPP", "POL", "ANY"],
            True,
            pytest.raises(RuntimeError, match="Unsupported dimension"),
        ),
        (["DIM", "STT"], False, no_raises()),
        (["DIM", "STT"], True, no_raises()),
        (["STO"], False, pytest.raises(RuntimeError, match="STO")),
        (["STO"], True, pytest.raises(RuntimeError, match="STO")),
        (["HOL"], False, pytest.raises(RuntimeError, match="HOL")),
        (["HOL"], True, pytest.raises(RuntimeError, match="HOL")),
        (["STO", "DIM", "HOL"], False, pytest.raises(RuntimeError, match="STO")),
        (["STO", "APC", "SPP"], True, pytest.raises(RuntimeError, match="STO")),
        (["APC", "DIM"], False, no_raises()),
        (["APC", "DIM"], True, pytest.raises(RuntimeError, match="APC")),
        (["DIM", "SPP"], False, no_raises()),
        (["DIM", "SPP"], True, pytest.raises(RuntimeError, match="SPP")),
    ],
)
def test_exclude_unsupported_axis_names(
    input_names, exclude_also_for_flags, expected_error
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        exclude_unsupported_axis_names,
    )

    with expected_error:
        exclude_unsupported_axis_names(input_names, exclude_also_for_flags)


@pytest.mark.parametrize(
    "input_correlation_mode, expected_components",
    [
        (pyasdm.enumerations.CorrelationMode.AUTO_ONLY, ["autoData"]),
        (pyasdm.enumerations.CorrelationMode.CROSS_AND_AUTO, ["crossData", "autoData"]),
        (pyasdm.enumerations.CorrelationMode.CROSS_ONLY, ["crossData"]),
    ],
)
def test_required_binary_components(input_correlation_mode, expected_components):
    """Flags are never required; crossData only when there are cross-correlations
    (F58)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        required_binary_components,
    )

    assert required_binary_components(input_correlation_mode) == expected_components


def make_spw(nchan, cross=("XX", "YY"), sd=("XX", "YY"), num_bin=1):
    stokes = pyasdm.enumerations.StokesParameter
    return {
        "crossPolProducts": [getattr(stokes, pol) for pol in cross],
        "sdPolProducts": [getattr(stokes, pol) for pol in sd],
        "scaleFactor": 1.0,
        "numSpectralPoint": nchan,
        "numBin": num_bin,
        "sideband": None,
        "sw": "1",
    }


def make_bdf_descr(
    basebands,
    num_antenna=3,
    apc=("AP_UNCORRECTED",),
    correlation_mode="CROSS_AND_AUTO",
    dimensionality=1,
    num_time=0,
    sizes=None,
    axes=None,
):
    descr = {
        "dimensionality": dimensionality,
        "num_time": num_time,
        "processor_type": pyasdm.enumerations.ProcessorType.CORRELATOR,
        "correlation_mode": pyasdm.enumerations.CorrelationMode.literal(
            correlation_mode
        ),
        "apc": [pyasdm.enumerations.AtmPhaseCorrection.literal(val) for val in apc],
        "num_antenna": num_antenna,
        "basebands": basebands,
    }
    descr["sizes"] = sizes if sizes is not None else {}
    descr["binary_types"] = list(descr["sizes"])
    descr["axes"] = axes if axes is not None else {}
    return descr


# BB_1: 8 + 1 channels, BB_2: 4 channels, dual-pol
basebands_uneven = [
    {"name": "BB_1", "spectralWindows": [make_spw(8), make_spw(1)]},
    {"name": "BB_2", "spectralWindows": [make_spw(4)]},
]


@pytest.mark.parametrize(
    "bdf_descr, expected_sizes",
    [
        # 3 baselines * (8 + 1 + 4) channels * 2 pols * 2 (re, im)
        (make_bdf_descr(basebands_uneven), {"crossData": [156], "autoData": [78]}),
        (
            make_bdf_descr(basebands_uneven, apc=("AP_UNCORRECTED", "AP_CORRECTED")),
            {"crossData": [312], "autoData": [78]},
        ),
        (
            make_bdf_descr(
                [{"name": "BB_1", "spectralWindows": [make_spw(4, num_bin=2)]}]
            ),
            {"crossData": [96], "autoData": [48]},
        ),
        # full-pol: 4 cross pols, 3 sd pols stored as 4 floats
        (
            make_bdf_descr(
                [
                    {
                        "name": "BB_1",
                        "spectralWindows": [
                            make_spw(2, ("XX", "XY", "YX", "YY"), ("XX", "XY", "YY"))
                        ],
                    }
                ]
            ),
            {"crossData": [48], "autoData": [24]},
        ),
        # packed AUTO_ONLY, numTime=4, 1 pol
        (
            make_bdf_descr(
                [{"name": "BB_1", "spectralWindows": [make_spw(4, (), ("XX",))]}],
                num_antenna=5,
                apc=(),
                correlation_mode="AUTO_ONLY",
                dimensionality=0,
                num_time=4,
            ),
            {"crossData": [0], "autoData": [80]},
        ),
    ],
)
def test_expected_data_sizes(bdf_descr, expected_sizes):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        expected_data_sizes,
    )

    assert expected_data_sizes(bdf_descr) == expected_sizes


@pytest.mark.parametrize(
    "apc, expected_index, expected_error",
    [
        ((), 0, no_raises()),
        (("AP_UNCORRECTED",), 0, no_raises()),
        (("AP_CORRECTED",), 0, no_raises()),
        (("AP_UNCORRECTED", "AP_CORRECTED"), 0, no_raises()),
        (("AP_CORRECTED", "AP_UNCORRECTED"), 1, no_raises()),
    ],
)
def test_select_apc_index(apc, expected_index, expected_error):
    """With two APCs, AP_UNCORRECTED is loaded wherever it is (K6, F11)."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        select_apc_index,
    )

    apc_list = [pyasdm.enumerations.AtmPhaseCorrection.literal(val) for val in apc]
    with expected_error:
        assert select_apc_index(apc_list) == expected_index


def test_select_apc_index_without_uncorrected():
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        select_apc_index,
    )

    with pytest.raises(NotImplementedError, match="AP_UNCORRECTED"):
        select_apc_index(["AP_CORRECTED", "AP_MIXED"])


@pytest.mark.parametrize(
    "baseband_spw_idxs, expected_error",
    [
        ((0, 0), no_raises()),
        ((0, 1), pytest.raises(NotImplementedError, match="numBin=2")),
        ((1, 0), no_raises()),
    ],
)
def test_check_num_bin(baseband_spw_idxs, expected_error):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        check_num_bin,
    )

    basebands = [
        {"name": "BB_1", "spectralWindows": [make_spw(8), make_spw(1, num_bin=2)]},
        {"name": "BB_2", "spectralWindows": [make_spw(4, num_bin=None)]},
    ]
    with expected_error:
        check_num_bin(basebands, baseband_spw_idxs, "foo_non_existent.nope")


@pytest.mark.parametrize(
    "bdf_descr, expected_result",
    [
        (
            make_bdf_descr(basebands_uneven, sizes={"crossData": 156, "autoData": 78}),
            no_raises(0),
        ),
        (
            make_bdf_descr(
                basebands_uneven,
                apc=("AP_CORRECTED", "AP_UNCORRECTED"),
                sizes={"crossData": 312, "autoData": 78, "flags": 33},
                axes={"crossData": ["BAL", "BAB", "SPW", "APC", "SPP", "POL"]},
            ),
            no_raises(1),
        ),
        # The header announces 2 APCs but the crossData hold only 1: the data
        # positions would be wrong
        (
            make_bdf_descr(
                basebands_uneven,
                apc=("AP_UNCORRECTED", "AP_CORRECTED"),
                sizes={"crossData": 156, "autoData": 78},
            ),
            pytest.raises(RuntimeError, match="crossData size=156"),
        ),
        (
            make_bdf_descr(basebands_uneven, sizes={"crossData": 156, "autoData": 80}),
            pytest.raises(RuntimeError, match="autoData size=80"),
        ),
        (
            make_bdf_descr(
                basebands_uneven,
                sizes={"crossData": 156, "autoData": 78},
                axes={"crossData": ["BAL", "BAB", "SPW", "SPP", "STO"]},
            ),
            pytest.raises(NotImplementedError, match="STO"),
        ),
        (
            make_bdf_descr(
                [{"name": "BB_1", "spectralWindows": [make_spw(4, num_bin=2)]}],
                sizes={"crossData": 96, "autoData": 48},
            ),
            pytest.raises(NotImplementedError, match="numBin"),
        ),
    ],
)
def test_check_data_components(bdf_descr, expected_result):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        check_data_components,
    )

    with expected_result as expected_apc_index:
        apc_index = check_data_components(bdf_descr, (0, 0), "foo_non_existent.nope")
        assert apc_index == expected_apc_index


def read_bdf_description(bdf_path):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        open_bdf,
    )
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        make_bdf_description,
    )

    with open_bdf(bdf_path) as bdf_reader:
        return make_bdf_description(bdf_reader.getHeader())


@pytest.mark.parametrize(
    "apc, expected_apc_index",
    [
        (("AP_UNCORRECTED",), 0),
        (("AP_UNCORRECTED", "AP_CORRECTED"), 0),
        (("AP_CORRECTED", "AP_UNCORRECTED"), 1),
    ],
)
def test_check_data_components_real_bdf(
    make_synthetic_asdm, synthetic_asdm_module, apc, expected_apc_index
):
    """The sizes announced in real BDF headers match the layout model."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        check_data_components,
    )

    truth = make_synthetic_asdm(
        synthetic_asdm_module.small_dtype_spec("INT16_TYPE", 2.0, apc=apc)
    )
    bdf_descr = read_bdf_description(truth.bdfs[0].path)
    assert set(bdf_descr["binary_types"]) == {"flags", "crossData", "autoData"}
    assert check_data_components(bdf_descr, (0, 0), "x") == expected_apc_index


def test_check_data_components_real_bdf_num_bin(
    make_synthetic_asdm, synthetic_asdm_module
):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        check_data_components,
    )

    truth = make_synthetic_asdm(
        synthetic_asdm_module.small_dtype_spec("FLOAT32_TYPE", 1.0, num_bin=2)
    )
    bdf_descr = read_bdf_description(truth.bdfs[0].path)
    with pytest.raises(NotImplementedError, match="numBin=2"):
        check_data_components(bdf_descr, (0, 0), truth.bdfs[0].path)


# ---------------------------------------------------------------------------
# open_bdf: errors opening / parsing a BDF name the BDF
# ---------------------------------------------------------------------------

#: Contents of broken BDFs: empty (interrupted copy), header that is not MIME,
#: MIME header without boundary, truncated in the middle of the XML header
BROKEN_BDF_CONTENTS = {
    "empty": b"",
    "not_mime": b"this is not a BDF\n",
    "no_boundary": b"MIME-Version: 1.0\nContent-Type: garbage\n",
    "truncated_header": (
        b"MIME-Version: 1.0\n"
        b'Content-Type: multipart/mixed; boundary="MIME_boundary-1";\n'
        b"Content-Description: Correlator\n\n--MIME_boundary-1\n"
        b"Content-Type: text/xml; charset=utf-8\n"
    ),
}


@pytest.mark.parametrize(
    "contents", BROKEN_BDF_CONTENTS.values(), ids=list(BROKEN_BDF_CONTENTS)
)
def test_open_bdf_broken_file(tmp_path, contents):
    """A BDF whose header cannot be parsed raises a BDFOpenError naming the BDF,
    which is both a RuntimeError and a pyasdm BDFReaderException."""
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        BDFOpenError,
        open_bdf,
    )

    bdf_path = tmp_path / "uid___A002_broken_bdf"
    bdf_path.write_bytes(contents)
    with pytest.raises(BDFOpenError, match="uid___A002_broken_bdf") as exc_info:
        with open_bdf(bdf_path):
            pytest.fail("open_bdf must not yield for a broken BDF")
    assert isinstance(exc_info.value, RuntimeError)
    assert isinstance(exc_info.value, pyasdm.exceptions.BDFReaderException)
    assert str(bdf_path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Exception)


def test_open_bdf_missing_file(tmp_path):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        BDFOpenError,
        open_bdf,
    )

    bdf_path = str(tmp_path / "missing_bdf")
    with pytest.raises(
        BDFOpenError, match="Cannot open the BDF .*missing_bdf"
    ) as exc_info:
        with open_bdf(bdf_path):
            pass
    # errors raised in dask workers are pickled
    unpickled = pickle.loads(pickle.dumps(exc_info.value))
    assert type(unpickled) is BDFOpenError
    assert str(unpickled) == str(exc_info.value)


def test_open_bdf_closes_reader():
    """The reader is closed when the header parse fails (no file handle leak),
    at the end of the block, and when the block raises."""
    from unittest import mock

    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        BDFOpenError,
        open_bdf,
    )

    with mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader:
        reader = mock_bdf_reader.return_value
        reader.open.side_effect = pyasdm.exceptions.BDFReaderException("bad header")
        with pytest.raises(BDFOpenError, match="some_bdf.*bad header"):
            with open_bdf("some_bdf"):
                pass
        reader.close.assert_called_once()

        reader.reset_mock()
        reader.open.side_effect = None
        with open_bdf("some_bdf") as opened:
            assert opened is reader
            reader.close.assert_not_called()
        reader.open.assert_called_once_with("some_bdf")
        reader.close.assert_called_once()

        reader.reset_mock()
        with pytest.raises(ValueError, match="in the block"):
            with open_bdf("some_bdf"):
                raise ValueError("in the block")
        reader.close.assert_called_once()


def test_open_bdf_real_bdf(synth_interferometric):
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        open_bdf,
    )

    with open_bdf(synth_interferometric.bdfs[0].path) as bdf_reader:
        assert bdf_reader.getHeader().getNumAntenna() == 4
        assert bdf_reader.hasSubset()
    assert bdf_reader.getPath() is None  # closed


def test_header_checks_name_the_bdf():
    from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
        check_basebands,
        check_correlation_mode,
        exclude_unsupported_axis_names,
    )

    check_basebands([{}], "a_bdf")
    with pytest.raises(RuntimeError, match="number of basebands in BDF a_bdf"):
        check_basebands([], "a_bdf")
    with pytest.raises(RuntimeError, match="number of basebands: "):
        check_basebands([{}] * 5)
    with pytest.raises(RuntimeError, match="mode CROSS_ONLY in BDF a_bdf "):
        check_correlation_mode(pyasdm.enumerations.CorrelationMode.CROSS_ONLY, "a_bdf")
    with pytest.raises(RuntimeError, match="HOL.* in BDF a_bdf"):
        exclude_unsupported_axis_names(["BAL", "HOL"], False, "a_bdf")
