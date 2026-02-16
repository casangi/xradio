from contextlib import nullcontext as no_raises
from unittest import mock

import pytest

import numpy as np

import pyasdm


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


def test_array_slice_to_msv4_indices():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        array_slice_to_msv4_indices,
    )

    slice_def = {
        "time": slice(0, 1),
        "baseline": slice(0, 9),
        "frequency": slice(0, 0.1),
        "polarization": slice(0, 1),
    }
    indices = array_slice_to_msv4_indices(slice_def)
    assert indices == (
        slice_def["time"],
        slice_def["baseline"],
        slice_def["frequency"],
        slice_def["polarization"],
    )


# From uid___A002_Xfd764e_X2197
basebands_example = [
    {
        "name": "BB_1",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "1",
            }
        ],
    },
    {
        "name": "BB_2",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "2",
            }
        ],
    },
    {
        "name": "BB_3",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 36454.168,
                "numSpectralPoint": 480,
                "numBin": 1,
                "sideband": None,
                "sw": "3",
            },
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 36454.168,
                "numSpectralPoint": 480,
                "numBin": 1,
                "sideband": None,
                "sw": "4",
            },
        ],
    },
    {
        "name": "BB_4",
        "spectralWindows": [
            {
                "crossPolProducts": [],
                "sdPolProducts": [],
                "scaleFactor": 103107.95,
                "numSpectralPoint": 960,
                "numBin": 1,
                "sideband": None,
                "sw": "5",
            }
        ],
    },
]


@pytest.mark.parametrize(
    "input_spw_idx, input_basebands, expected_baseband_idx, expected_spw_idx",
    [
        (0, [], 0, 0),
        (1, [], 0, 0),
        (0, basebands_example, 0, 0),
        (1, basebands_example, 1, 0),
        (2, basebands_example, 2, 0),
        (3, basebands_example, 2, 1),
        (4, basebands_example, 3, 0),
        (5, basebands_example, 0, 0),
    ],
)
def test_find_spw_in_basebands_list_empty(
    input_spw_idx, input_basebands, expected_baseband_idx, expected_spw_idx
):
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_spw_in_basebands_list,
    )

    baseband_idx, spw_idx = find_spw_in_basebands_list(
        input_spw_idx, input_basebands, "bogus_path_non_existant.nope"
    )
    assert baseband_idx == expected_baseband_idx
    assert spw_idx == expected_spw_idx


def test_find_different_basebands_spws():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_different_basebands_spws,
    )

    basebands = [
        {
            "name": "BB_1",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 103107.95,
                    "numSpectralPoint": 960,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "1",
                }
            ],
        },
    ]
    result = find_different_basebands_spws(basebands)
    assert result is False


def test_find_different_basebands_spws_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_different_basebands_spws,
    )

    result = find_different_basebands_spws([])
    assert result is False


def test_find_different_basebands_pols_empty():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        find_different_basebands_pols,
    )

    result = find_different_basebands_pols([])
    assert result is False


def test_load_visibilities_from_bdf():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_from_bdf,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # mock_bdf_reader.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = {}
        mock_bdf_header.getBasebandsList.side_effect = ["foo", "bar"]
        with pytest.raises(RuntimeError, match="basebands"):
            load_visibilities_from_bdf(
                "/inexistent/foo/path/", 0, {}, never_reshape_from_all_spws=True
            )
        mock_bdf_header.getBasebandsList()
        mock_bdf_header.getBasebandsList.assert_called()


def test_load_visibilities_all_subsets():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_all_subsets,
    )

    basebands = [
        {
            "name": "BB_1",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 103107.95,
                    "numSpectralPoint": 960,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "1",
                }
            ],
        },
        {
            "name": "BB_2",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 103107.95,
                    "numSpectralPoint": 960,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "2",
                }
            ],
        },
        {
            "name": "BB_3",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 36454.168,
                    "numSpectralPoint": 480,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "3",
                },
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 36454.168,
                    "numSpectralPoint": 480,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "4",
                },
            ],
        },
        {
            "name": "BB_4",
            "spectralWindows": [
                {
                    "crossPolProducts": [],
                    "sdPolProducts": [],
                    "scaleFactor": 103107.95,
                    "numSpectralPoint": 960,
                    "numBin": 1,
                    "sideband": None,
                    "sw": "5",
                }
            ],
        },
    ]

    bdf_descr = {
        "basebands": basebands,
        "processor_type": "CORRELATOR",
    }

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        with pytest.raises(RuntimeError, match="autoData not present"):
            load_visibilities_all_subsets(
                mock_bdf_reader, (1, 36, 9, 4, 2, 512, 2, 2), (0, 0), bdf_descr
            )
        mock_bdf_header.getBasebandsList()
        mock_bdf_header.getBasebandsList.assert_called()


# Will likely also need a ProcessorType.RADIOMETER/SPECTROMETER
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


def test_load_visibilities_all_subsets_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_all_subsets,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {"autoData": {"present": True, "arr": np.zeros((73728))}}
        ]
        visibilities = load_visibilities_all_subsets(
            mock_bdf_reader, (1, 36, 9, 4, 2, 512, 2, 2), (0, 1), bdf_descr_X136e
        )

        assert isinstance(visibilities, np.ndarray)
        assert visibilities.size == 9216
        assert visibilities.shape == (1, 9, 512, 2)
        assert visibilities.dtype == np.dtype("float64")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1
