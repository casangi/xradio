from contextlib import nullcontext as no_raises

import numpy as np
import pandas as pd
import xarray as xr

import pytest

import pyasdm


def add_main_table(asdm: pyasdm.ASDM):
    main_row_0_xml = """
  <row>
    <time> 5230000651200000000 </time>
    <numAntenna> 2 </numAntenna>
    <timeSampling>INTEGRATION</timeSampling>
    <interval> 24192000000 </interval>
    <numIntegration> 1512 </numIntegration>
    <scanNumber> 1 </scanNumber>
    <subscanNumber> 1 </subscanNumber>
    <dataSize> 1681962 </dataSize>
    <dataUID>
      <EntityRef entityId="uid://A002/X11b94a6/X119f" partId="X00000000" entityTypeName="Main" documentVersion="1"/>
    </dataUID>
    <configDescriptionId> ConfigDescription_0 </configDescriptionId>
    <execBlockId> ExecBlock_0 </execBlockId>
    <fieldId> Field_0 </fieldId>
    <stateId> 1 12 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0 State_0  </stateId>
  </row>
    """
    main_table = asdm.getMain()
    main_row_0 = pyasdm.MainRow(main_table)
    main_row_0.setFromXML(main_row_0_xml)
    main_table.add(main_row_0)


def add_data_description_table(asdm: pyasdm.ASDM):
    data_description_row_0_xml = """
  <row>
    <dataDescriptionId> DataDescription_0 </dataDescriptionId>
    <polOrHoloId> Polarization_0 </polOrHoloId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    data_description_table = asdm.getDataDescription()
    data_description_row_0 = pyasdm.DataDescriptionRow(data_description_table)
    data_description_row_0.setFromXML(data_description_row_0_xml)
    data_description_table.add(data_description_row_0)


def add_polarization_table(asdm: pyasdm.ASDM):
    polarization_row_0_xml = """
  <row>
    <polarizationId> Polarization_0 </polarizationId>
    <numCorr> 2 </numCorr>
    <corrType> 1 2 XX YY</corrType>
    <corrProduct> 2 2 2 X X Y Y</corrProduct>
  </row>
    """
    polarization_table = asdm.getPolarization()
    polarization_row_0 = pyasdm.PolarizationRow(polarization_table)
    polarization_row_0.setFromXML(polarization_row_0_xml)
    polarization_table.add(polarization_row_0)


def add_field_table(asdm: pyasdm.ASDM):
    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
    <time> 5230000639104000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 0 </sourceId>
  </row>
    """
    field_table = asdm.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)


def add_source_table(asdm: pyasdm.ASDM):
    source_row_0_xml = """
  <row>
    <sourceId> 0 </sourceId>
    <timeInterval> 7090683272335387903 4265377529038775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 1.3528024488371877 0.31436086058385826  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J0510+1800 </sourceName>
    <directionCode>J2000</directionCode>
    <numFreq> 4 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 4 2.1998305541101968E11 2.1800401136221878E11 2.3300428785048865E11 2.350043230298097E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <flux> 2 4 4 3.855274498565509 0.0 0.0 0.0 3.8656094704537534 0.0 0.0 0.0 3.7901533014049034 0.0 0.0 0.0 3.7805688135422617 0.0 0.0 0.0  </flux>
    <size> 2 4 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </size>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
"""
    source_table = asdm.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml)
    source_table.add(source_row_0)


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
        parts = open_partition(asdm_empty, {"fieldId": [0]})


def test_open_partition_asdm_with_spw_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(IndexError, match="out of range"):
        parts = open_partition(
            asdm_with_spw_default,
            {"fieldId": [0], "configDescriptionId": [0], "scanNumber": [0]},
        )


def test_open_partition_asdm_with_spw_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    with pytest.raises(KeyError, match="BDFPath"):
        parts = open_partition(
            asdm_with_spw_simple,
            {"fieldId": [0], "configDescriptionId": [0], "scanNumber": [0]},
        )


def test_open_partition_monkeypatched_bdf_asdm_with_spw_simple(
    asdm_with_spw_simple, monkeypatch
):
    from xradio.measurement_set._utils._asdm.open_partition import open_partition

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.get_times_from_bdfs",
        mock_get_times_from_bdfs,
    )
    add_main_table(asdm_with_spw_simple)
    add_data_description_table(asdm_with_spw_simple)
    add_polarization_table(asdm_with_spw_simple)
    add_field_table(asdm_with_spw_simple)
    add_source_table(asdm_with_spw_simple)
    partition = open_partition(
        asdm_with_spw_simple,
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


def test_correlated_xds_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_correlated_xds

    partition_descr = {"fieldId": [0], "scanNumber": [0], "BDFPath": []}
    with pytest.raises(IndexError, match="out of range"):
        coords = create_correlated_xds(asdm_with_spw_default, partition_descr)


def test_create_data_vars_no_bdf_path(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm.open_partition import create_data_vars

    with pytest.raises(KeyError, match="time"):
        coords = create_data_vars(xr.Dataset(), [""], 0)


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
        coords = create_coordinates(asdm_with_spw_simple, partition_descr, False)


def test_create_coordinates_monkeypatched_bdf_with_spw_simple(
    asdm_with_spw_simple, monkeypatch
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

    add_main_table(asdm_with_spw_simple)
    add_data_description_table(asdm_with_spw_simple)
    add_polarization_table(asdm_with_spw_simple)
    coords, attrs, num_antenna, spw_id, bdf_spw_id, time_vars = create_coordinates(
        asdm_with_spw_simple, partition_descr, False
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
        weight = produce_uvw_data_var(xr.Dataset())


def test_produce_weight_data_var():
    from xradio.measurement_set._utils._asdm.open_partition import (
        produce_weight_data_var,
    )

    with pytest.raises(KeyError, match="time"):
        weight = produce_weight_data_var(xr.Dataset())


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
