from contextlib import nullcontext as no_raises

import numpy as np

import pytest

import pyasdm


def test_open_partition_none():
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(AttributeError, match="no attribute"):
        open_partition(None, ["fieldId"])


def test_open_partition_asdm_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        parts = open_partition(asdm_empty, {"fieldId": [0]})


def test_open_partition_asdm_with_spw_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        parts = open_partition(asdm_with_spw_default, {"fieldId": [0]})


def test_open_partition_asdm_with_spw_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        parts = open_partition(asdm_with_spw_simple, {"fieldId": [0]})


def test_create_coordinates_no_bdf_path(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_coordinates

    partition_descr = {"fieldId": [0], "scanNumber": [0], "BDFPath": []}
    with pytest.raises(ValueError, match="at least one array"):
        coords = create_coordinates(asdm_with_spw_default, partition_descr, False)


def test_create_coordinates_with_spw_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_coordinates

    partition_descr = {
        "fieldId": [0],
        "scanNumber": [0],
        "BDFPath": ["/nonexistent/foo"],
    }
    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="No such file or directory"
    ):
        coords = create_coordinates(asdm_with_spw_default, partition_descr, False)


@pytest.mark.parametrize(
    "num_antenna, expected_output, expected_error",
    [
        (
            1,
            ((0,), (0,)),
            no_raises(),
        ),
        (
            2,
            (np.array([0, 0, 1]), np.array([0, 1, 1])),
            no_raises(),
        ),
        (
            3,
            (np.array([0, 0, 0, 1, 1, 2]), np.array([0, 1, 2, 1, 2, 2])),
            no_raises(),
        ),
    ],
)
def test_generate_baseline_antennax_id_as_in_msv2(
    num_antenna, expected_output, expected_error
):
    from xradio.measurement_set._utils._asdm.open_partition import (
        generate_baseline_antennax_id_as_in_msv2,
    )

    with expected_error:
        antennax_id = generate_baseline_antennax_id_as_in_msv2(num_antenna)
        if len(expected_output[0]) == 1 and len(antennax_id[0]) == 1:
            assert antennax_id[0] == expected_output[0]
            assert antennax_id[1] == expected_output[1]
        else:
            assert (antennax_id[0] == expected_output[0]).all()
            assert (antennax_id[1] == expected_output[1]).all()
