from unittest import mock

import numpy as np

import pytest

import pyasdm

example_guessed_shape_auto_only = {
    "auto": (5, 9, 1, 64, 2),
    "cross": None,
}
example_guessed_shape = {
    "cross": (3, 36, 1, 64, 2, 2),
    "auto": (5, 9, 1, 64, 2),
}


def test_add_cross_and_auto_flag_shapes_without_cross():
    pass


@pytest.mark.parametrize(
    "input_guessed_shape, expected_out_shape",
    [
        (example_guessed_shape, (3, 45, 1, 64, 2, 2)),
        (example_guessed_shape_auto_only, (5, 9, 1, 64, 2)),
    ],
)
def test_add_cross_and_auto_flag_shapes_with_cross(
    input_guessed_shape, expected_out_shape
):
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        add_cross_and_auto_flag_shapes,
    )

    added = add_cross_and_auto_flag_shapes(input_guessed_shape)
    assert added == expected_out_shape


def test_full_shape_to_output_filled_flags_shape():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        full_shape_to_output_filled_flags_shape,
    )

    out_shape = full_shape_to_output_filled_flags_shape(example_guessed_shape["cross"])
    assert out_shape == (3, 36, 2)
    out_shape = full_shape_to_output_filled_flags_shape(example_guessed_shape["auto"])
    assert out_shape == (5, 9, 2)


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


def test_load_visibilities_all_subsets_from_trees():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    bdf_descr = {
        "basebands": basebands_example,
        "processor_type": "CORRELATOR",
    }
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        with pytest.raises(RuntimeError, match="not present"):
            # Test with load_one_spw_from_file=False => load_vis_subset_from_tree() (also below)
            load_visibilities_all_subsets_from_trees(
                mock_bdf_reader,
                (2, 45, 2, 64, 2, 2),
                (0, 0),
                bdf_descr,
                load_one_spw_from_file=False,
            )
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_vis_subset_from_tree():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_vis_subset_from_tree,
    )

    pyasdm_subset = {}
    guessed_shape = (2, 45, 2, 64, 2, 2)
    bdf_descr = {
        "basebands": basebands_example,
        "processor_type": "CORRELATOR",
    }
    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        with pytest.raises(RuntimeError, match="not present"):
            load_vis_subset_from_tree(pyasdm_subset, guessed_shape, (0, 0), bdf_descr)
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


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


def test_load_visibilities_all_subsets_from_trees_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_visibilities_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        # For load_vis_subset, etc.
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        mock_bdf_reader.getSubset.side_effect = [
            {
                "autoData": {"present": True, "arr": np.zeros((73728))},
                "crossData": {"present": True, "arr": np.zeros((294912))},
            }
        ]
        with pytest.raises(ValueError, match="cannot reshape array"):
            visibilities = load_visibilities_all_subsets_from_trees(
                mock_bdf_reader,
                (1, 36, 9, 4, 2, 512, 2, 2),
                (0, 1),
                bdf_descr_X136e,
                load_one_spw_from_file=False,
            )

            assert isinstance(visibilities, np.ndarray)
            assert visibilities.size == 9216
            assert visibilities.shape == (1, 9, 512, 2)
            assert visibilities.dtype == np.dtype("float64")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 1
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1


# load_vis_subset_cross_data_from_tree
# load_vis_subset_auto_data_from_tree


def test_load_flags_all_subsets_from_trees():
    from xradio.measurement_set._utils._asdm._utils._bdf.pyasdm_load_from_trees import (
        load_visibilities_all_subsets_from_trees,
    )

    with (
        mock.patch("pyasdm.bdf.BDFReader") as mock_bdf_reader,
        mock.patch("pyasdm.bdf.BDFHeader") as mock_bdf_header,
    ):
        bdf_descr = {
            "basebands": basebands_example,
            "processor_type": "CORRELATOR",
        }
        mock_bdf_reader.hasSubset.side_effect = [True, False]
        vis = load_visibilities_all_subsets_from_trees(
            mock_bdf_reader, (2, 45, 2, 64, 2, 2), (0, 0), bdf_descr
        )
        assert vis.size == 0
        assert vis.dtype == np.float64
        mock_bdf_header.getBasebandsList.assert_not_called()
        mock_bdf_header.getSubset.assert_not_called()
        mock_bdf_header.hasSubset.assert_not_called()


def test_load_flags_all_subsets_from_trees_X136e():
    from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
        load_flags_all_subsets_from_trees,
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
        guessed_shape = {
            "cross": (1, 36, 4, 2, 512, 2, 2),
            "auto": (1, 9, 4, 2, 512, 2),
        }
        flags = load_flags_all_subsets_from_trees(
            mock_bdf_reader,
            guessed_shape,
            bdf_descr_X136e,
            (0, 1),
        )
        assert isinstance(flags, np.ndarray)
        assert flags.size == 46080
        assert flags.shape == (1, 45, 512, 2)
        assert flags.dtype == np.dtype("bool")

        mock_bdf_reader.hasSubset.assert_called()
        assert mock_bdf_reader.hasSubset.call_count == 2
        mock_bdf_reader.getSubset.assert_called()
        assert mock_bdf_reader.getSubset.call_count == 1
