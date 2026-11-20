import copy
from contextlib import nullcontext as no_raises

import numpy as np
import pandas as pd
import pyasdm
import pytest
import xarray as xr

from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set.schema import FieldSourceXds
from xradio.schema.check import check_dataset


def test_create_field_and_source_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_field_and_source_xds(None, {}, 0, False)


def check_empty_if_field_and_source_xds(xds):
    assert xds.attrs["type"] == "field_and_source"
    center_direction_name = "FIELD_PHASE_CENTER_DIRECTION"
    assert center_direction_name in xds.data_vars
    assert xds.data_vars[center_direction_name].shape == (0, 2)
    issues = check_dataset(xds, FieldSourceXds)
    assert not issues


def test_create_field_and_source_xds_with_asdm_empty(asdm_empty):
    xds = create_field_and_source_xds(asdm_empty, {"fieldId": [0]}, 0, False)
    check_empty_if_field_and_source_xds(xds)


def test_create_field_and_source_xds_with_asdm_default(asdm_with_spw_default):
    xds = create_field_and_source_xds(asdm_with_spw_default, {"fieldId": [0]}, 0, False)
    check_empty_if_field_and_source_xds(xds)


def test_create_field_and_source_xds_with_asdm_simple(asdm_with_spw_simple):
    xds = create_field_and_source_xds(asdm_with_spw_simple, {"fieldId": [0]}, 0, False)
    check_empty_if_field_and_source_xds(xds)


@pytest.mark.parametrize(
    "input_is_single_dish",
    [
        (True),
        (False),
    ],
)
def test_create_field_and_source_xds_with_field_minimum(
    asdm_with_spw_simple, input_is_single_dish
):
    asdm_with_minimum_field = copy.deepcopy(asdm_with_spw_simple)

    # A field row with only the required attributes
    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
  </row>
    """
    field_table = asdm_with_minimum_field.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_minimum_field, {"fieldId": [0]}, 0, input_is_single_dish
    )
    issues = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues


@pytest.mark.parametrize(
    "input_is_single_dish",
    [
        (True),
        (False),
    ],
)
def test_create_field_and_source_xds_with_field_numpoly(
    asdm_with_spw_simple, input_is_single_dish
):
    """uses an anomalous field table with numPoly > 1 (never seen in ALMA?)"""
    asdm_with_minimum_field = copy.deepcopy(asdm_with_spw_simple)

    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 2 </numPoly>
    <delayDir> 2 2 2 1.1487030439690096 -0.023431362760917465 0.01 0.001  </delayDir>
    <phaseDir> 2 2 2 1.1487030439690096 -0.023431362760917465 0.01 0.001  </phaseDir>
    <referenceDir> 2 2 2 1.148703043969797 -0.02343136276090743 0.01 0.001  </referenceDir>
  </row>
    """
    field_table = asdm_with_minimum_field.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_minimum_field, {"fieldId": [0]}, 0, input_is_single_dish
    )
    issues = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues


@pytest.mark.parametrize(
    "input_is_single_dish",
    [
        (True),
        (False),
    ],
)
def test_create_field_and_source_xds_with_field_ephemeris_id(
    asdm_with_spw_simple, input_is_single_dish
):
    """uses a field table with some ephemeris ids"""
    asdm_with_minimum_field = copy.deepcopy(asdm_with_spw_simple)

    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
    <ephemerisId> 0 </ephemerisId>
  </row>
    """
    field_table = asdm_with_minimum_field.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_minimum_field, {"fieldId": [0]}, 0, input_is_single_dish
    )
    issues = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues


@pytest.mark.parametrize("input_is_single_dish", [(True), (False)])
def test_create_field_and_source_xds_with_n_fields(
    asdm_with_spw_simple, input_is_single_dish
):
    asdm_with_minimum_field = copy.deepcopy(asdm_with_spw_simple)

    field_row_0_xml = """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J0423-0120 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
  </row>
    """
    field_row_1_xml = """
  <row>
    <fieldId> Field_1 </fieldId>
    <fieldName> J0423-0121 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </delayDir>
    <phaseDir> 2 1 2 1.1487030439690096 -0.023431362760917465  </phaseDir>
    <referenceDir> 2 1 2 1.148703043969797 -0.02343136276090743  </referenceDir>
  </row>
    """
    field_table = asdm_with_minimum_field.getField()
    for row_xml in [field_row_0_xml, field_row_1_xml]:
        field_row = pyasdm.FieldRow(field_table)
        field_row.setFromXML(row_xml)
        field_table.add(field_row)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_minimum_field, {"fieldId": [0, 1]}, 0, input_is_single_dish
    )
    issues = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues
    expected_name_shape = (2,)
    assert field_and_source_xds.coords["field_name"].shape == expected_name_shape
    center_variable = (
        "FIELD_REFERENCE_CENTER_DIRECTION"
        if input_is_single_dish
        else "FIELD_PHASE_CENTER_DIRECTION"
    )
    expected_data_shape = expected_name_shape + (2,)
    assert field_and_source_xds.data_vars[center_variable].shape == expected_data_shape


@pytest.mark.parametrize("input_is_single_dish", [(True), (False)])
def test_create_field_and_source_xds_with_field_source(
    asdm_with_spw_simple, input_is_single_dish
):
    asdm_with_source_field = copy.deepcopy(asdm_with_spw_simple)

    # numLines, etc. optional attributes (line info) borrowed from a different row
    source_row_0_xml_main_chunk = """
    <sourceId> 0 </sourceId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 1.148703043969797 -0.02343136276090743  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J0423-0120 </sourceName>
    <directionCode>ICRS</directionCode>
    <numFreq> 8 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 8 6.246009851630612E11 6.351401391376388E11 6.227258305018281E11 6.370152937988718E11 6.208506844426849E11 6.38890439858015E11 6.189755340824967E11 6.407655902182032E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <flux> 2 8 4 2.3005181262897874 0.0 0.0 0.0 2.287221128582244 0.0 0.0 0.0 2.3029156368538692 0.0 0.0 0.0 2.284886406783615 0.0 0.0 0.0 2.305322876552315 0.0 0.0 0.0 2.282560930875987 0.0 0.0 0.0 2.307739931084516 0.0 0.0 0.0 2.28024462134997 0.0 0.0 0.0  </flux>
    <size> 2 8 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </size>
    <dopplerVelocity> 1 1 0.0  </dopplerVelocity>
    <dopplerReferenceSystem>LSRK</dopplerReferenceSystem>
    <dopplerCalcType>OPTICAL</dopplerCalcType>
    <parallax> 1 1 0.0  </parallax>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
    """
    source_row_0_xml_without_lineinfo = f"""
  <row>
    {source_row_0_xml_main_chunk}
  </row>
    """
    source_table = asdm_with_source_field.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml_without_lineinfo)
    source_table.add(source_row_0)

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
    field_table = asdm_with_source_field.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_source_field, {"fieldId": [0]}, 0, input_is_single_dish
    )
    issues = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues

    if input_is_single_dish:
        phase_center_name = "FIELD_REFERENCE_CENTER_DIRECTION"
    else:
        phase_center_name = "FIELD_PHASE_CENTER_DIRECTION"
    assert phase_center_name in field_and_source_xds.data_vars
    expected_phase_center = xr.DataArray(
        data=[[1.14870304, -0.02343136]],
        dims=["field_name", "sky_dir_label"],
        coords={
            "field_name": ["J0423-0120_0"],
            "source_name": ("field_name", ["J0423-0120"]),
            "sky_dir_label": ["ra", "dec"],
        },
    )
    xr.testing.assert_allclose(
        field_and_source_xds[phase_center_name], expected_phase_center
    )
    assert field_and_source_xds[phase_center_name].attrs["frame"] == "icrs"

    # Now with lineinfo in Source
    asdm_with_source_field_with_lineinfo = copy.deepcopy(asdm_with_spw_simple)
    source_row_0_xml_with_lineinfo = f"""
  <row>
    {source_row_0_xml_main_chunk}
    <numLines> 1 </numLines>
    <transition> 1 1 &quot;cii_blue(ID=0)&quot;  </transition>
    <restFrequency> 1 1 1.8949423708552605E12  </restFrequency>
    <sysVel> 1 1 0.0  </sysVel>
    <frequencyRefCode>GEO</frequencyRefCode>
  </row>
    """
    source_table = asdm_with_source_field_with_lineinfo.getSource()
    source_row_0 = pyasdm.SourceRow(source_table)
    source_row_0.setFromXML(source_row_0_xml_with_lineinfo)
    source_table.add(source_row_0)

    field_table = asdm_with_source_field_with_lineinfo.getField()
    field_row_0 = pyasdm.FieldRow(field_table)
    field_row_0.setFromXML(field_row_0_xml)
    field_table.add(field_row_0)

    field_and_source_xds = create_field_and_source_xds(
        asdm_with_source_field_with_lineinfo, {"fieldId": [0]}, 0, input_is_single_dish
    )
    issues_field_source = check_dataset(field_and_source_xds, FieldSourceXds)
    assert not issues_field_source

    assert phase_center_name in field_and_source_xds.data_vars
    xr.testing.assert_allclose(
        field_and_source_xds[phase_center_name], expected_phase_center
    )
    assert field_and_source_xds[phase_center_name].attrs["frame"] == "icrs"

    assert "line_label" in field_and_source_xds.coords
    rest_freq_name = "LINE_REST_FREQUENCY"
    sys_vel_name = "LINE_SYSTEMIC_VELOCITY"
    for data_var in [rest_freq_name, sys_vel_name]:
        assert data_var in field_and_source_xds.data_vars
    expected_dims = ["field_name", "line_label"]
    expected_coords = {
        "field_name": ["J0423-0120_0"],
        "source_name": ("field_name", ["J0423-0120"]),
        "line_label": ["0"],
        "line_name": (["field_name", "line_label"], [["cii_blue(ID=0)"]]),
    }
    expected_rest_freq = xr.DataArray(
        data=[[1.894942370e12]], dims=expected_dims, coords=expected_coords
    )
    xr.testing.assert_allclose(
        field_and_source_xds.data_vars[rest_freq_name], expected_rest_freq
    )
    assert field_and_source_xds[rest_freq_name].attrs["observer"] == "gcrs"

    expected_sys_vel = xr.DataArray(
        data=[[0.0]], dims=expected_dims, coords=expected_coords
    )
    xr.testing.assert_allclose(
        field_and_source_xds.data_vars[sys_vel_name], expected_sys_vel
    )
    assert field_and_source_xds[sys_vel_name].attrs["units"] == "m/s"


reference_field_centers = [
    [4.004208, -0.425373],
    [4.004208, -0.425373],
    [4.509587, -0.470912],
    [4.764123, -0.305859],
    [4.714607, -0.420047],
]

reference_source_line_names = [
    ["H2CO_3_0_3__2_0_2_(ID=3974784)"],
    ["Continuum(ID=0)", "H2CO_3_2_2__2_2_1_(ID=3974785)"],
    [
        "Continuum(ID=0)",
        "H2CO_3_2_2__2_2_1_(ID=3974785)",
        "H2CO_3_2_1__2_2_0_(ID=3974786)",
        "H2CO_3_0_3__2_0_2_(ID=3974784)",
    ],
]

reference_source_rest_frequencies = [
    [2.1822192e11],
    [2.1800000e11, 2.1847563e11],
    [2.1800000e11, 2.1847563e11, 2.1876007e11, 2.1822219e11],
]


@pytest.mark.parametrize("input_is_single_dish", [(True), (False)])
@pytest.mark.parametrize(
    "input_field_ids, expected_field_name, expected_source_name, expected_center, expected_line_names, expected_rest_frequencies, expected_error",
    [
        (
            [0],
            ["J1517-2422_0"],
            ["J1517-2422"],
            [reference_field_centers[0]],
            [reference_source_line_names[0]],
            [reference_source_rest_frequencies[0]],
            no_raises(),
        ),
        (
            [1],
            ["J1517-2422_b_1"],
            ["J1517-2422"],
            [reference_field_centers[1]],
            [reference_source_line_names[0]],
            [reference_source_rest_frequencies[0]],
            no_raises(),
        ),
        (
            [2],
            ["J1713-2658_2"],
            ["J1517-2422b"],
            [reference_field_centers[2]],
            [reference_source_line_names[1]],
            [reference_source_rest_frequencies[1]],
            no_raises(),
        ),
        (
            [3],
            ["18089-1732_3"],
            ["J1517-2422c"],
            [reference_field_centers[3]],
            [reference_source_line_names[2]],
            [reference_source_rest_frequencies[2]],
            no_raises(),
        ),
        (
            [4],
            ["G5.89-0.39_4"],
            ["J1517-2422c"],
            [reference_field_centers[4]],
            [],
            [],
            pytest.raises(RuntimeError, match="Unexpected inconsistency"),
        ),
        (
            [0, 1],
            ["J1517-2422_0", "J1517-2422_b_1"],
            ["J1517-2422", "J1517-2422"],
            [reference_field_centers[0], reference_field_centers[1]],
            [reference_source_line_names[0], reference_source_line_names[0]],
            [
                reference_source_rest_frequencies[0],
                reference_source_rest_frequencies[0],
            ],
            no_raises(),
        ),
        (
            [2, 3],
            ["J1713-2658_2", "18089-1732_3"],
            ["J1517-2422b", "J1517-2422c"],
            [reference_field_centers[2], reference_field_centers[3]],
            [reference_source_line_names[1] + ["", ""], reference_source_line_names[2]],
            [
                reference_source_rest_frequencies[1] + [np.nan, np.nan],
                reference_source_rest_frequencies[2],
            ],
            no_raises(),
        ),
    ],
)
def test_create_field_and_source_xds_with_multi_n_field_n_source(
    asdm_with_spw_simple,
    input_is_single_dish,
    input_field_ids,
    expected_field_name,
    expected_source_name,
    expected_center,
    expected_line_names,
    expected_rest_frequencies,
    expected_error,
):
    asdm_with_source_field = copy.deepcopy(asdm_with_spw_simple)

    # Examples borrowed from uid___A002_Xc3412f_X2a7d
    source_rows_xml = [
        """
  <row>
    <sourceId> 0 </sourceId>
    <timeInterval> 7116244650251387903 4214254773206775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 4.0042080373642355 -0.4253729833818179  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J1517-2422 </sourceName>
    <directionCode>ICRS</directionCode>
    <numLines> 1 </numLines>
    <transition> 1 1 &quot;H2CO_3_0_3__2_0_2_(ID=3974784)&quot;  </transition>
    <restFrequency> 1 1 2.18222192E11  </restFrequency>
    <sysVel> 1 1 0.0  </sysVel>
    <numFreq> 5 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 5 2.181932254559272E11 2.198792606121772E11 2.1797102612834592E11 2.1844670316774753E11 2.1873088285524753E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <velRefCode>LSRK</velRefCode>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
        """,
        """
  <row>
    <sourceId> 1 </sourceId>
    <timeInterval> 7116244650251387903 4214254773206775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 4.0042080373642355 -0.4253729833818179  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J1517-2422b </sourceName>
    <directionCode>ICRS</directionCode>
    <numLines> 2 </numLines>
    <transition> 1 2 &quot;Continuum(ID=0)&quot; &quot;H2CO_3_2_2__2_2_1_(ID=3974785)&quot;  </transition>
    <restFrequency> 1 2 2.18E11 2.18475632E11  </restFrequency>
    <sysVel> 1 2 0.0 0.0  </sysVel>
    <numFreq> 5 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 5 2.181932254559272E11 2.198792606121772E11 2.1797102612834592E11 2.1844670316774753E11 2.1873088285524753E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <flux> 2 5 4 3.2866613225069625 0.0 0.0 0.0 3.2817622136727502 0.0 0.0 0.0 3.287310336055036 0.0 0.0 0.0 3.285921911861967 0.0 0.0 0.0 3.2850941584888718 0.0 0.0 0.0  </flux>
    <size> 2 5 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </size>
    <velRefCode>LSRK</velRefCode>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
        """,
        """
  <row>
    <sourceId> 2 </sourceId>
    <timeInterval> 7116244650251387903 4214254773206775807 </timeInterval>
    <code> none </code>
    <direction> 1 2 4.0042080373642355 -0.4253729833818179  </direction>
    <properMotion> 1 2 0.0 0.0  </properMotion>
    <sourceName> J1517-2422c </sourceName>
    <directionCode>ICRS</directionCode>
    <numLines> 4 </numLines>
    <transition> 1 4 &quot;Continuum(ID=0)&quot; &quot;H2CO_3_2_2__2_2_1_(ID=3974785)&quot; &quot;H2CO_3_2_1__2_2_0_(ID=3974786)&quot; &quot;H2CO_3_0_3__2_0_2_(ID=3974784)&quot;  </transition>
    <restFrequency> 1 4 2.18E11 2.18475632E11 2.18760066E11 2.18222192E11  </restFrequency>
    <sysVel> 1 4 0.0 0.0 0.0 0.0  </sysVel>
    <numFreq> 5 </numFreq>
    <numStokes> 4 </numStokes>
    <frequency> 1 5 2.181932254559272E11 2.198792606121772E11 2.1797102612834592E11 2.1844670316774753E11 2.1873088285524753E11  </frequency>
    <stokesParameter> 1 4 I Q U V</stokesParameter>
    <velRefCode>LSRK</velRefCode>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
        """,
    ]

    source_table = asdm_with_source_field.getSource()
    for row_xml in source_rows_xml:
        source_row = pyasdm.SourceRow(source_table)
        source_row.setFromXML(row_xml)
        source_table.add(source_row)

    field_rows_xml = [
        """
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> J1517-2422 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 4.004208037363708 -0.4253729833818717  </delayDir>
    <phaseDir> 2 1 2 4.004208037363708 -0.4253729833818717  </phaseDir>
    <referenceDir> 2 1 2 4.0042080373642355 -0.4253729833818179  </referenceDir>
    <time> 5009117382048000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 0 </sourceId>
  </row>
        """,
        """
  <row>
    <fieldId> Field_1 </fieldId>
    <fieldName> J1517-2422_b </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 4.004208037363708 -0.4253729833818717  </delayDir>
    <phaseDir> 2 1 2 4.004208037363708 -0.4253729833818717  </phaseDir>
    <referenceDir> 2 1 2 4.0042080373642355 -0.4253729833818179  </referenceDir>
    <time> 5009117382048000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 0 </sourceId>
  </row>
        """,
        """
  <row>
    <fieldId> Field_2 </fieldId>
    <fieldName> J1713-2658 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 4.509587217416476 -0.4709117799461657  </delayDir>
    <phaseDir> 2 1 2 4.509587217416476 -0.4709117799461657  </phaseDir>
    <referenceDir> 2 1 2 4.509587217417485 -0.4709117799458208  </referenceDir>
    <time> 5009118362784000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 1 </sourceId>
  </row>
        """,
        """
  <row>
    <fieldId> Field_3 </fieldId>
    <fieldName> 18089-1732 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 4.764123448294921 -0.3058592551387708  </delayDir>
    <phaseDir> 2 1 2 4.764123448294921 -0.3058592551387708  </phaseDir>
    <referenceDir> 2 1 2 4.764123448295888 -0.3058592551383841  </referenceDir>
    <time> 5009118476016000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 2 </sourceId>
  </row>
        """,
        """
  <row>
    <fieldId> Field_4 </fieldId>
    <fieldName> G5.89-0.39 </fieldName>
    <numPoly> 1 </numPoly>
    <delayDir> 2 1 2 4.714607002974764 -0.42004742145056834  </delayDir>
    <phaseDir> 2 1 2 4.714607002974764 -0.42004742145056834  </phaseDir>
    <referenceDir> 2 1 2 4.714607002975766 -0.4200474214501131  </referenceDir>
    <time> 5009118673440000000 </time>
    <code> none </code>
    <directionCode>ICRS</directionCode>
    <sourceId> 3 </sourceId>
  </row>
        """,
    ]
    field_table = asdm_with_source_field.getField()
    for row_xml in field_rows_xml:
        field_row = pyasdm.FieldRow(field_table)
        field_row.setFromXML(row_xml)
        field_table.add(field_row)

    with expected_error:
        field_and_source_xds = create_field_and_source_xds(
            asdm_with_source_field,
            {"fieldId": input_field_ids},
            0,
            input_is_single_dish,
        )
        issues_if = check_dataset(field_and_source_xds, FieldSourceXds)
        assert not issues_if

        if input_is_single_dish:
            phase_center_name = "FIELD_REFERENCE_CENTER_DIRECTION"
        else:
            phase_center_name = "FIELD_PHASE_CENTER_DIRECTION"
        assert phase_center_name in field_and_source_xds.data_vars
        expected_phase_center = xr.DataArray(
            data=expected_center,
            dims=["field_name", "sky_dir_label"],
            coords={
                "field_name": expected_field_name,
                "source_name": ("field_name", expected_source_name),
                "sky_dir_label": ["ra", "dec"],
            },
        )
        xr.testing.assert_allclose(
            field_and_source_xds[phase_center_name], expected_phase_center
        )
        assert field_and_source_xds[phase_center_name].attrs["frame"] == "icrs"

        assert "line_label" in field_and_source_xds.coords
        rest_freq_name = "LINE_REST_FREQUENCY"
        sys_vel_name = "LINE_SYSTEMIC_VELOCITY"
        for data_var in [rest_freq_name, sys_vel_name]:
            assert data_var in field_and_source_xds.data_vars
        expected_dims = ["field_name", "line_label"]
        max_lines = max([len(line_names) for line_names in expected_line_names])
        line_label = [str(line_idx) for line_idx in range(max_lines)]
        expected_coords = {
            "field_name": expected_field_name,
            "source_name": ("field_name", expected_source_name),
            "line_label": line_label,
            "line_name": (["field_name", "line_label"], expected_line_names),
        }
        expected_rest_freq = xr.DataArray(
            data=expected_rest_frequencies,
            dims=expected_dims,
            coords=expected_coords,
        )
        xr.testing.assert_allclose(
            field_and_source_xds.data_vars[rest_freq_name], expected_rest_freq
        )
        assert field_and_source_xds[rest_freq_name].attrs["observer"] == "lsrk"

        sys_vel_data = np.array([[0.0] * len(line_label)] * len(input_field_ids))
        # mirror nan positions (mismatches in len of line numbers between sources)
        sys_vel_data[np.isnan(expected_rest_frequencies)] = np.nan
        expected_sys_vel = xr.DataArray(
            data=sys_vel_data,
            dims=expected_dims,
            coords=expected_coords,
        )
        xr.testing.assert_allclose(
            field_and_source_xds.data_vars[sys_vel_name], expected_sys_vel
        )
        assert field_and_source_xds[sys_vel_name].attrs["units"] == "m/s"


@pytest.mark.parametrize(
    "input_is_single_dish",
    [
        (True),
        (False),
    ],
)
def test___make_field_center_direction(input_is_single_dish):
    from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
        _make_field_center_direction,
    )

    xds = xr.Dataset()
    field_df = pd.DataFrame.from_dict(
        {
            "referenceDir": [0.1, 0.2],
            "phaseDir": [1.11, 1.21],
        }
    )
    center_direction = _make_field_center_direction(xds, field_df, input_is_single_dish)
    assert isinstance(center_direction, np.ndarray)


def test_extract_source_info_empty():
    from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
        extract_source_info,
    )

    result = extract_source_info(
        xr.Dataset(), None, pd.DataFrame(columns=["sourceId"]), 0
    )
    assert "field_name" in result.dims
    assert "source_name" in result.coords
