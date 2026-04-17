from contextlib import nullcontext as no_raises

import numpy as np
import pandas as pd
import xarray as xr

import pytest

import pyasdm

from xradio.schema.check import check_datatree


def mock_get_times_from_bdfs(
    bdf_paths: list[str], scans_metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.array([0.1]), np.array([1.0]), np.array([0.101]), np.array([1.0])


def test_open_partition_none():
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(AttributeError, match="no attribute"):
        open_partition(None, ["fieldId"])


def test_open_partition_asdm_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        open_partition(asdm_empty, {"fieldId": [0]})


def test_open_partition_asdm_with_spw_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        open_partition(
            asdm_with_spw_default,
            {"fieldId": [0], "configDescriptionId": [0], "scanNumber": [0]},
        )


def test_open_partition_asdm_with_spw_simple(
    asdm_with_main_execblock_config_processor_sbsummary,
):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(KeyError, match="BDFPath"):
        open_partition(
            asdm_with_main_execblock_config_processor_sbsummary,
            {"fieldId": [0], "configDescriptionId": [0], "scanNumber": [0]},
        )


def test_open_partition_monkeypatched_bdf_asdm_with_spw_simple(
    asdm_with_main_etc_data_description_polarization_field_source, monkeypatch
):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.get_times_from_bdfs",
        mock_get_times_from_bdfs,
    )
    partition = open_partition(
        asdm_with_main_etc_data_description_polarization_field_source,
        {
            "fieldId": [0],
            "configDescriptionId": [0],
            "scanNumber": [0],
            "scanIntent": [0],
            "dataDescriptionId": [0],
            "BDFPath": ["/inexistent_test_path/foo"],
        },
    )
    assert isinstance(partition, xr.DataTree)
    check_datatree(partition)


def test_correlated_xds_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_correlated_xds

    partition_descr = {"fieldId": [0], "scanNumber": [0], "BDFPath": []}
    with pytest.raises(IndexError, match="out of range"):
        create_correlated_xds(asdm_with_spw_default, partition_descr)


def test_create_data_vars_no_bdf_path(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_data_vars

    with pytest.raises(KeyError, match="time"):
        create_data_vars(xr.Dataset(), [""], 0)


def test_create_coordinates_no_bdf_path(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_coordinates

    partition_descr = {"fieldId": [0], "scanNumber": [0], "BDFPath": []}
    with pytest.raises(ValueError, match="at least one array"):
        create_coordinates(asdm_with_spw_default, partition_descr, False)


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
        create_coordinates(asdm_with_spw_default, partition_descr, False)


def test_create_coordinates_with_spw_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm.open_partition import create_coordinates

    partition_descr = {
        "fieldId": [0],
        "scanNumber": [0],
        "BDFPath": ["/nonexistent/foo"],
    }
    with pytest.raises(
        pyasdm.exceptions.BDFReaderException, match="No such file or directory"
    ):
        create_coordinates(asdm_with_spw_simple, partition_descr, False)


def test_create_coordinates_monkeypatched_bdf_with_spw_simple(
    asdm_with_main_data_description_config_description_polarization, monkeypatch
):
    from xradio.measurement_set._utils._asdm.open_partition import create_coordinates

    partition_descr = {
        "fieldId": [0],
        "scanNumber": [0],
        "scanIntent": 0,
        "configDescriptionId": [0],
        "dataDescriptionId": [0],
        "BDFPath": ["/nonexistent/bar"],
    }

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.get_times_from_bdfs",
        mock_get_times_from_bdfs,
    )

    coords, attrs, num_antenna, spw_id, bdf_spw_id, time_vars = create_coordinates(
        asdm_with_main_data_description_config_description_polarization,
        partition_descr,
        False,
    )
    assert isinstance(coords, dict)
    for coo in [
        "time",
        "baseline_id",
        "frequency",
        "polarization",
        "baseline_antenna1_name",
        "baseline_antenna1_name",
        "scan_name",
    ]:
        assert coo in coords
    assert isinstance(attrs, dict)
    assert attrs["frequency"]["units"] == "Hz"
    assert num_antenna == 2
    assert spw_id == 0
    assert bdf_spw_id == 0
    assert isinstance(time_vars, dict)
    for var in ["EFFECTIVE_INTEGRATION_TIME", "TIME_CENTROID"]:
        assert var in time_vars


def test_produce_uvw_data_var():
    from xradio.measurement_set._utils._asdm.open_partition import produce_uvw_data_var

    with pytest.raises(KeyError, match="time"):
        produce_uvw_data_var(xr.Dataset())


def test_produce_weight_data_var():
    from xradio.measurement_set._utils._asdm.open_partition import (
        produce_weight_data_var,
    )

    with pytest.raises(KeyError, match="time"):
        produce_weight_data_var(xr.Dataset())


def test_translate_asdm_tables_spw_id_to_bdf_spw_id(asdm_empty):
    from xradio.measurement_set._utils._asdm.open_partition import (
        translate_asdm_tables_spw_id_to_bdf_spw_id,
    )

    bdf_spw_id = translate_asdm_tables_spw_id_to_bdf_spw_id(
        [0],
        pd.DataFrame({"dataDescriptionId": [0, 1], "spectralWindowId": [14, 16]}),
        pd.DataFrame(),
        [0],
        pd.DataFrame({"configDescriptionId": [0, 1], "dataDescriptionId": [[0], [1]]}),
    )
    assert bdf_spw_id == 0


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
            (np.array([0, 0, 1]), np.array([1, 0, 1])),
            no_raises(),
        ),
        (
            3,
            (np.array([0, 0, 1, 0, 1, 2]), np.array([1, 2, 2, 0, 1, 2])),
            no_raises(),
        ),
    ],
)
def test_generate_baseline_antennax_id_as_in_bdf(
    num_antenna, expected_output, expected_error
):
    from xradio.measurement_set._utils._asdm.open_partition import (
        generate_baseline_antennax_id_as_in_bdf,
    )

    with expected_error:
        antennax_id = generate_baseline_antennax_id_as_in_bdf(num_antenna)
        if len(expected_output[0]) == 1 and len(antennax_id[0]) == 1:
            assert antennax_id[0] == expected_output[0]
            assert antennax_id[1] == expected_output[1]
        else:
            assert (antennax_id[0] == expected_output[0]).all()
            assert (antennax_id[1] == expected_output[1]).all()


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
