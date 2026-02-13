from contextlib import nullcontext as no_raises
from unittest import mock

import pytest


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
        mock_bdf_reader.hasSubset.side_effects = [True, False]
        mock_bdf_reader.getSubset.side_effects = {}
        mock_bdf_header.getBasebandsList.side_effects = ["foo", "bar"]
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
        with pytest.raises(UnboundLocalError, match="vis_subset_auto"):
            load_visibilities_all_subsets(
                mock_bdf_reader, (1, 1, 1, 1, 1, 1), (0, 0), bdf_descr
            )
        mock_bdf_header.getBasebandsList()
        mock_bdf_header.getBasebandsList.assert_called()
