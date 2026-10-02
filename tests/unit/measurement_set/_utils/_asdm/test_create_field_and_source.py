import copy
import dataclasses
import logging
from unittest import mock

import numpy as np
import pyasdm
import pytest
import xarray as xr

import xradio.measurement_set._utils._asdm.create_field_and_source_xds as fs_module
from xradio.measurement_set._utils._asdm._utils.calculate_uvw import (
    calculate_uvw,
    phase_center_frame_to_astropy,
)
from xradio.measurement_set._utils._asdm._utils.field_source import (
    ASDM_DIRECTION_CODE_TO_FRAME,
)
from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set._utils._asdm.open_asdm import open_asdm
from xradio.measurement_set.schema import FieldSourceXds
from xradio.schema.check import check_dataset

# Source.timeInterval (XML: midpoint, duration) covering everything
WIDE_INTERVAL = "7226686294548387903 3993371484612775807"


def uvw_supports_frame(frame: str) -> bool:
    """Whether calculate_uvw accepts phase centers in this frame."""
    try:
        phase_center_frame_to_astropy(frame)
    except NotImplementedError:
        return False
    return True


def _fmt(value: float) -> str:
    return repr(float(value))


def _poly_dir_xml(terms) -> str:
    """Field direction polynomial: terms is a list of (ra, dec) per term."""
    values = " ".join(f"{_fmt(ra)} {_fmt(dec)}" for ra, dec in terms)
    return f"2 {len(terms)} 2 {values}"


def add_field(
    asdm,
    name: str,
    phase_dir=(1.0, -0.5),
    reference_dir=None,
    source_id: int | None = 0,
    direction_code: str | None = "ICRS",
    phase_dir_terms=None,
    ephemeris_id: int | None = None,
):
    """Add a Field row. Returns the row (fieldId assigned sequentially)."""
    phase_terms = phase_dir_terms or [phase_dir]
    reference_terms = [reference_dir or phase_dir] + [
        (0.0, 0.0) for _ in phase_terms[1:]
    ]
    optional_xml = ""
    if direction_code is not None:
        optional_xml += f"<directionCode>{direction_code}</directionCode>"
    if ephemeris_id is not None:
        optional_xml += f"<ephemerisId> {ephemeris_id} </ephemerisId>"
    if source_id is not None:
        optional_xml += f"<sourceId> {source_id} </sourceId>"
    xml = f"""
  <row>
    <fieldId> Field_0 </fieldId>
    <fieldName> {name} </fieldName>
    <numPoly> {len(phase_terms)} </numPoly>
    <delayDir> {_poly_dir_xml(phase_terms)} </delayDir>
    <phaseDir> {_poly_dir_xml(phase_terms)} </phaseDir>
    <referenceDir> {_poly_dir_xml(reference_terms)} </referenceDir>
    <time> 5230000639104000000 </time>
    <code> none </code>
    {optional_xml}
  </row>
    """
    table = asdm.getField()
    row = pyasdm.FieldRow(table)
    row.setFromXML(xml)
    return table.add(row)


def add_source(
    asdm,
    name: str,
    spw_id: int = 0,
    direction=(1.0, -0.5),
    direction_code: str | None = "ICRS",
    time_interval: str = WIDE_INTERVAL,
    transitions=None,
    rest_frequencies=None,
    sys_vels=None,
    num_lines: int | None = None,
    frequency_ref_code: str | None = None,
):
    """Add a Source row. pyasdm assigns sourceId by sourceName."""
    optional_xml = ""
    if direction_code is not None:
        optional_xml += f"<directionCode>{direction_code}</directionCode>"
    if num_lines is not None:
        optional_xml += f"<numLines> {num_lines} </numLines>"
    if transitions is not None:
        names = " ".join(f"&quot;{name}&quot;" for name in transitions)
        optional_xml += f"<transition> 1 {len(transitions)} {names} </transition>"
    if rest_frequencies is not None:
        values = " ".join(_fmt(freq) for freq in rest_frequencies)
        optional_xml += (
            f"<restFrequency> 1 {len(rest_frequencies)} {values} </restFrequency>"
        )
    if sys_vels is not None:
        values = " ".join(_fmt(vel) for vel in sys_vels)
        optional_xml += f"<sysVel> 1 {len(sys_vels)} {values} </sysVel>"
    if frequency_ref_code is not None:
        optional_xml += f"<frequencyRefCode>{frequency_ref_code}</frequencyRefCode>"
    xml = f"""
  <row>
    <sourceId> 0 </sourceId>
    <timeInterval> {time_interval} </timeInterval>
    <code> none </code>
    <direction> 1 2 {_fmt(direction[0])} {_fmt(direction[1])} </direction>
    <properMotion> 1 2 0.0 0.0 </properMotion>
    <sourceName> {name} </sourceName>
    {optional_xml}
    <spectralWindowId> SpectralWindow_{spw_id} </spectralWindowId>
  </row>
    """
    table = asdm.getSource()
    row = pyasdm.SourceRow(table)
    row.setFromXML(xml)
    return table.add(row)


def warnings_logged(mock_logger) -> list[str]:
    return [str(call.args[0]) for call in mock_logger.warning.call_args_list]


@pytest.fixture
def mock_logger(monkeypatch):
    logger = mock.MagicMock()
    monkeypatch.setattr(fs_module, "xradio_logger", lambda: logger)
    return logger


def assert_no_schema_issues(xds):
    issues = check_dataset(xds, FieldSourceXds)
    assert not issues, str(issues)


def test_create_field_and_source_xds_empty():
    with pytest.raises(AttributeError, match="has no attribute"):
        create_field_and_source_xds(None, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_no_field_id(asdm_empty):
    with pytest.raises(RuntimeError, match="no fieldId"):
        create_field_and_source_xds(asdm_empty, {"fieldId": []}, 0, False)


def test_create_field_and_source_xds_with_asdm_empty(asdm_empty):
    with pytest.raises(RuntimeError, match="not found in the Field table"):
        create_field_and_source_xds(asdm_empty, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_asdm_default(asdm_with_spw_default):
    with pytest.raises(RuntimeError, match="not found in the Field table"):
        create_field_and_source_xds(asdm_with_spw_default, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_asdm_simple(asdm_with_spw_simple):
    with pytest.raises(RuntimeError, match="not found in the Field table"):
        create_field_and_source_xds(asdm_with_spw_simple, {"fieldId": [0]}, 0, False)


def test_create_field_and_source_xds_with_field_source(asdm_with_spw_simple):
    """Rows taken from a real ALMA ASDM (Field and Source in ICRS)."""
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

    phase_dir = [1.1487030439690096, -0.023431362760917465]
    reference_dir = [1.148703043969797, -0.02343136276090743]

    # IF
    if_xds = create_field_and_source_xds(
        asdm_with_source_field, {"fieldId": [0]}, 0, False
    )
    assert_no_schema_issues(if_xds)
    assert list(if_xds.field_name.values) == ["J0423-0120_0"]
    assert list(if_xds.source_name.values) == ["J0423-0120"]
    assert list(if_xds.sky_dir_label.values) == ["ra", "dec"]
    np.testing.assert_array_equal(
        if_xds.FIELD_PHASE_CENTER_DIRECTION.values, [phase_dir]
    )
    assert if_xds.FIELD_PHASE_CENTER_DIRECTION.attrs["frame"] == "icrs"
    assert "FIELD_REFERENCE_CENTER_DIRECTION" not in if_xds
    np.testing.assert_array_equal(if_xds.SOURCE_DIRECTION.values, [reference_dir])
    assert if_xds.SOURCE_DIRECTION.attrs["frame"] == "icrs"
    assert "line_name" not in if_xds.coords
    assert "LINE_REST_FREQUENCY" not in if_xds
    assert if_xds.attrs["is_ephemeris"] is False
    assert if_xds.attrs["type"] == "field_and_source"

    # SD
    sd_xds = create_field_and_source_xds(
        asdm_with_source_field, {"fieldId": [0]}, 0, True
    )
    assert_no_schema_issues(sd_xds)
    np.testing.assert_array_equal(
        sd_xds.FIELD_REFERENCE_CENTER_DIRECTION.values, [reference_dir]
    )
    assert sd_xds.FIELD_REFERENCE_CENTER_DIRECTION.attrs["frame"] == "icrs"
    assert "FIELD_PHASE_CENTER_DIRECTION" not in sd_xds

    # IF with lineinfo in Source
    asdm_with_source_field_with_lineinfo = copy.deepcopy(asdm_with_spw_simple)
    source_row_0_xml_with_lineinfo = f"""
  <row>
    {source_row_0_xml_main_chunk}
    <numLines> 1 </numLines>
    <transition> 1 1 &quot;cii_blue(ID=0)&quot;  </transition>
    <restFrequency> 1 1 1.8949423708552605E12  </restFrequency>
    <sysVel> 1 1 0.0  </sysVel>
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

    xds = create_field_and_source_xds(
        asdm_with_source_field_with_lineinfo, {"fieldId": [0]}, 0, False
    )
    assert_no_schema_issues(xds)
    assert list(xds.line_label.values) == ["0"]
    assert xds.line_name.dims == ("field_name", "line_label")
    assert xds.line_name.values.tolist() == [["cii_blue(ID=0)"]]
    np.testing.assert_array_equal(
        xds.LINE_REST_FREQUENCY.values, [[1.8949423708552605e12]]
    )
    assert xds.LINE_REST_FREQUENCY.attrs["observer"] == "lsrk"
    assert xds.LINE_REST_FREQUENCY.attrs["units"] == "Hz"
    np.testing.assert_array_equal(xds.LINE_SYSTEMIC_VELOCITY.values, [[0.0]])
    assert xds.LINE_SYSTEMIC_VELOCITY.attrs["units"] == "m/s"


def test_phase_center_from_phase_dir_and_reference_center_from_reference_dir():
    """F30: interferometric FIELD_PHASE_CENTER_DIRECTION from Field.phaseDir,
    single-dish FIELD_REFERENCE_CENTER_DIRECTION from Field.referenceDir."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "M100", phase_dir=(1.0, -0.5), reference_dir=(1.1, -0.4))
    add_source(asdm, "M100", direction=(1.05, -0.45))

    if_xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(if_xds)
    np.testing.assert_array_equal(
        if_xds.FIELD_PHASE_CENTER_DIRECTION.sel(field_name="M100_0").values,
        [1.0, -0.5],
    )
    assert "FIELD_REFERENCE_CENTER_DIRECTION" not in if_xds
    np.testing.assert_array_equal(
        if_xds.SOURCE_DIRECTION.sel(field_name="M100_0").values, [1.05, -0.45]
    )

    sd_xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, True)
    assert_no_schema_issues(sd_xds)
    np.testing.assert_array_equal(
        sd_xds.FIELD_REFERENCE_CENTER_DIRECTION.sel(field_name="M100_0").values,
        [1.1, -0.4],
    )
    assert "FIELD_PHASE_CENTER_DIRECTION" not in sd_xds


def make_mosaic_asdm():
    """Calibrator field 0 and two mosaic pointings (fields 1, 2) sharing the
    name M100 and source 1. Source 1 has two lines in SPW 0."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "J0423-0120", phase_dir=(1.15, -0.02), source_id=0)
    add_field(asdm, "M100", phase_dir=(3.2, 0.27), source_id=1)
    add_field(asdm, "M100", phase_dir=(3.201, 0.2705), source_id=1)
    add_source(asdm, "J0423-0120", direction=(1.15, -0.02))
    add_source(
        asdm,
        "M100",
        direction=(3.2005, 0.2702),
        transitions=["CO_v_0_2_1(ID=0)", "CS_v_0_5_4(ID=1)"],
        rest_frequencies=[2.30538e11, 2.44935643e11],
        sys_vels=[1571.0, 1572.0],
        num_lines=2,
    )
    return asdm


def test_multi_field_partition():
    """F31/K9: several fields in one partition (no fieldId partition axis): one
    entry per field, ascending fieldId, unique names (F29), per-field source
    and line information."""
    asdm = make_mosaic_asdm()
    xds = create_field_and_source_xds(asdm, {"fieldId": [2, 0, 1, 1]}, 0, False)
    assert_no_schema_issues(xds)

    assert list(xds.field_name.values) == ["J0423-0120_0", "M100_1", "M100_2"]
    assert list(xds.source_name.values) == ["J0423-0120", "M100", "M100"]
    assert xds.FIELD_PHASE_CENTER_DIRECTION.dims == ("field_name", "sky_dir_label")
    np.testing.assert_array_equal(
        xds.FIELD_PHASE_CENTER_DIRECTION.values,
        [[1.15, -0.02], [3.2, 0.27], [3.201, 0.2705]],
    )
    np.testing.assert_array_equal(
        xds.SOURCE_DIRECTION.values,
        [[1.15, -0.02], [3.2005, 0.2702], [3.2005, 0.2702]],
    )

    # lines: '0', '1' labels, (field_name, line_label) names, NaN for the
    # calibrator without line information
    assert list(xds.line_label.values) == ["0", "1"]
    assert xds.line_name.dims == ("field_name", "line_label")
    assert xds.line_name.values.tolist() == [
        ["", ""],
        ["CO_v_0_2_1(ID=0)", "CS_v_0_5_4(ID=1)"],
        ["CO_v_0_2_1(ID=0)", "CS_v_0_5_4(ID=1)"],
    ]
    np.testing.assert_array_equal(
        xds.LINE_REST_FREQUENCY.values,
        [[np.nan, np.nan], [2.30538e11, 2.44935643e11], [2.30538e11, 2.44935643e11]],
    )
    np.testing.assert_array_equal(
        xds.LINE_SYSTEMIC_VELOCITY.values,
        [[np.nan, np.nan], [1571.0, 1572.0], [1571.0, 1572.0]],
    )


def test_mosaic_pointings_keep_unique_names():
    """F29: pointings that share a fieldName get distinct field_name values."""
    asdm = make_mosaic_asdm()
    names = [
        create_field_and_source_xds(
            asdm, {"fieldId": [fid]}, 0, False
        ).field_name.values[0]
        for fid in range(3)
    ]
    assert names == ["J0423-0120_0", "M100_1", "M100_2"]


def test_line_info_read_per_source_row(mock_logger):
    """F32: rows of other sources / SPWs without line information do not drop
    the line information of the partition's own Source row."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    add_field(asdm, "Calibrator", phase_dir=(2.0, 0.1), source_id=1)
    add_source(
        asdm,
        "Target",
        spw_id=0,
        transitions=["CO_v_0_2_1(ID=0)"],
        rest_frequencies=[2.30538e11],
        sys_vels=[1000.0],
        num_lines=1,
    )
    add_source(asdm, "Calibrator", spw_id=0, direction=(2.0, 0.1))  # no line info
    # no line info for SPW 1 (added last: pyasdm SourceTable.add fails when a new
    # source is added to a SPW after another SPW was added)
    add_source(asdm, "Target", spw_id=1)

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert xds.line_name.values.tolist() == [["CO_v_0_2_1(ID=0)"]]
    np.testing.assert_array_equal(xds.LINE_REST_FREQUENCY.values, [[2.30538e11]])
    np.testing.assert_array_equal(xds.LINE_SYSTEMIC_VELOCITY.values, [[1000.0]])
    assert not warnings_logged(mock_logger)

    # the same source in SPW 1, without line info: no line coords/vars
    xds_spw1 = create_field_and_source_xds(asdm, {"fieldId": [0]}, 1, False)
    assert_no_schema_issues(xds_spw1)
    assert "line_label" not in xds_spw1.coords
    assert "LINE_REST_FREQUENCY" not in xds_spw1
    assert "LINE_SYSTEMIC_VELOCITY" not in xds_spw1

    # both fields in one partition: the calibrator gets NaN, with a warning
    xds_both = create_field_and_source_xds(asdm, {"fieldId": [0, 1]}, 0, False)
    assert_no_schema_issues(xds_both)
    np.testing.assert_array_equal(
        xds_both.LINE_REST_FREQUENCY.values, [[2.30538e11], [np.nan]]
    )
    assert any("Calibrator_1" in msg for msg in warnings_logged(mock_logger))


def test_line_info_partial_attributes(mock_logger):
    """numLines/transition without restFrequency/sysVel: names kept, no
    LINE_REST_FREQUENCY/LINE_SYSTEMIC_VELOCITY variables, warning logged."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    add_source(asdm, "Target", transitions=["H2O"], num_lines=1)

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert xds.line_name.values.tolist() == [["H2O"]]
    assert "LINE_REST_FREQUENCY" not in xds
    assert "LINE_SYSTEMIC_VELOCITY" not in xds
    assert any("incomplete" in msg for msg in warnings_logged(mock_logger))


@pytest.mark.parametrize(
    "frequency_ref_code, expected_observer",
    [(None, "lsrk"), ("LSRK", "lsrk"), ("TOPO", "TOPO"), ("BARY", "BARY")],
)
def test_rest_frequency_observer(frequency_ref_code, expected_observer):
    """F96: rest frequency observer from Source.frequencyRefCode, 'lsrk' (as in
    MSv2-converted datasets) when absent."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    add_source(
        asdm,
        "Target",
        transitions=["CO", "CS"],
        rest_frequencies=[2.30538e11, 2.44935643e11],
        sys_vels=[10.0, 20.0],
        frequency_ref_code=frequency_ref_code,
    )
    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert xds.LINE_REST_FREQUENCY.attrs["observer"] == expected_observer
    # numLines absent: number of lines from the per-line attributes
    assert list(xds.line_label.values) == ["0", "1"]
    assert xds.line_name.values.tolist() == [["CO", "CS"]]
    assert xds.line_name.sel(field_name="Target_0", line_label="1").item() == "CS"


def test_no_source_row_for_spw(mock_logger):
    """F70: no Source row for (source, spw): 'Unknown' source, no error."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    add_source(asdm, "Target", spw_id=1)

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert list(xds.source_name.values) == ["Unknown"]
    assert "SOURCE_DIRECTION" not in xds
    np.testing.assert_array_equal(
        xds.FIELD_PHASE_CENTER_DIRECTION.values, [[1.0, -0.5]]
    )
    assert any("No Source row found" in msg for msg in warnings_logged(mock_logger))


def test_several_source_rows_for_spw(mock_logger):
    """F70: several Source rows (time intervals) for (source, spw): the
    earliest is used, with a warning."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    later = add_source(
        asdm,
        "Target",
        direction=(1.2, -0.6),
        time_interval="5230000652242000000 2000000000",
    )
    earlier = add_source(
        asdm,
        "Target",
        direction=(1.1, -0.55),
        time_interval="5230000552242000000 2000000000",
    )
    assert later.getSourceId() == earlier.getSourceId() == 0
    assert asdm.getSource().size() == 2

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert list(xds.source_name.values) == ["Target"]
    np.testing.assert_array_equal(xds.SOURCE_DIRECTION.values, [[1.1, -0.55]])
    assert any("Only the first" in msg for msg in warnings_logged(mock_logger))


def test_field_without_source_id(mock_logger):
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=None)
    add_field(asdm, "Other", phase_dir=(2.0, 0.0), source_id=0)
    add_source(asdm, "Other", direction=(2.0, 0.0))

    xds = create_field_and_source_xds(asdm, {"fieldId": [0, 1]}, 0, False)
    assert_no_schema_issues(xds)
    assert list(xds.source_name.values) == ["Unknown", "Other"]
    np.testing.assert_array_equal(
        xds.SOURCE_DIRECTION.values, [[np.nan, np.nan], [2.0, 0.0]]
    )
    assert any("has no sourceId" in msg for msg in warnings_logged(mock_logger))


@pytest.mark.parametrize(
    "direction_code, expected_frame",
    [
        ("ICRS", "icrs"),
        ("J2000", "fk5"),
        (None, "fk5"),
        ("B1950", "fk4"),
        ("GALACTIC", "galactic"),
        ("SUPERGAL", "supergalactic"),
    ],
)
def test_field_direction_frame(direction_code, expected_frame):
    """F69: frame from Field.directionCode (J2000 when absent). These celestial
    frames are accepted for interferometric phase centres (UVW) too."""
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0, direction_code=direction_code)
    add_source(asdm, "Target", direction_code="ICRS")

    for is_single_dish, var in [
        (False, "FIELD_PHASE_CENTER_DIRECTION"),
        (True, "FIELD_REFERENCE_CENTER_DIRECTION"),
    ]:
        xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, is_single_dish)
        assert_no_schema_issues(xds)
        assert xds[var].attrs["frame"] == expected_frame
        assert xds[var].attrs["units"] == "rad"
        assert xds.SOURCE_DIRECTION.attrs["frame"] == "icrs"


@pytest.mark.parametrize(
    "direction_code, expected_frame",
    [
        ("HADEC", "hadec"),
        ("AZELNE", "altaz"),
        ("AZELNEGEO", "altaz"),
        ("ITRF", "itrs"),
    ],
)
def test_phase_center_frame_without_uvw_support_rejected(
    direction_code, expected_frame
):
    """Phase centres in frames that the UVW calculation cannot use (they would
    need an observation time and location) fail when the partition is opened,
    not later when the lazy UVW is computed. Single dish (no UVW) keeps them."""
    assert not uvw_supports_frame(expected_frame)
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0, direction_code=direction_code)
    add_source(asdm, "Target")

    with pytest.raises(ValueError, match="UVW cannot be calculated") as exc_info:
        create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    message = str(exc_info.value)
    assert f"frame '{expected_frame}'" in message
    assert f"directionCode {direction_code}" in message
    assert "Target_0" in message

    sd_xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, True)
    assert_no_schema_issues(sd_xds)
    assert sd_xds.FIELD_REFERENCE_CENTER_DIRECTION.attrs["frame"] == expected_frame
    np.testing.assert_array_equal(
        sd_xds.FIELD_REFERENCE_CENTER_DIRECTION.values, [[1.0, -0.5]]
    )


ALMA_CENTER_ITRF = np.array([2225142.180268967, -5440307.370348562, -2481029.851873547])


def _uvw_from_phase_center(
    phase_center_direction: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """UVW of 3 ALMA baselines at 2 times for a (field_name, sky_dir_label)
    phase center, with calculate_uvw (as the lazy UVW array does)."""
    positions = np.array(
        [
            ALMA_CENTER_ITRF,
            ALMA_CENTER_ITRF + [900.0, 1200.0, -500.0],
            ALMA_CENTER_ITRF + [-300.0, 800.0, 100.0],
        ]
    )
    time = xr.DataArray(
        np.array([1709272763.0, 1709283600.0]),
        dims="time",
        attrs={"type": "time", "units": "s", "format": "unix", "scale": "utc"},
    )
    antenna_position = xr.DataArray(
        positions,
        dims=["antenna_name", "cartesian_pos_label"],
        coords={
            "antenna_name": ["A", "B", "C"],
            "cartesian_pos_label": ["x", "y", "z"],
        },
        attrs={"type": "location", "units": "m", "frame": "ITRS"},
    )
    antenna1 = xr.DataArray(["A", "A", "B"], dims="baseline_id")
    antenna2 = xr.DataArray(["B", "C", "C"], dims="baseline_id")
    uvw = calculate_uvw(
        None, time, antenna1, antenna2, antenna_position, phase_center_direction
    )
    baseline_lengths = np.linalg.norm(
        positions[[0, 0, 1]] - positions[[1, 2, 2]], axis=-1
    )
    return uvw, baseline_lengths


@pytest.mark.parametrize("direction_code", sorted(ASDM_DIRECTION_CODE_TO_FRAME))
def test_accepted_phase_center_frames_give_uvw(direction_code):
    """Every phase centre accepted when a partition is opened can be used by
    the UVW calculation: the field direction frames and the UVW frames agree
    (rejected frames are the ones calculate_uvw does not support)."""
    frame = ASDM_DIRECTION_CODE_TO_FRAME[direction_code]
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0, direction_code=direction_code)
    add_source(asdm, "Target")

    if not uvw_supports_frame(frame):
        with pytest.raises(ValueError, match="UVW cannot be calculated"):
            create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
        return

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    phase_center = xds.FIELD_PHASE_CENTER_DIRECTION
    assert phase_center.attrs["frame"] == frame
    uvw, baseline_lengths = _uvw_from_phase_center(phase_center)
    assert uvw.shape == (2, 3, 3)
    assert np.all(np.isfinite(uvw))
    # UVW is a rotation of the baseline vector
    np.testing.assert_allclose(
        np.linalg.norm(uvw, axis=-1),
        np.broadcast_to(baseline_lengths, (2, 3)),
        rtol=1e-9,
    )


def _open_asdm_with_field_codes(synthetic_asdm_module, make_synthetic_asdm, codes):
    """Write the synthetic mosaic ASDM (fields 0, 1, 2, one partition per
    field) with the given Field directionCode of every field and open it."""
    spec = synthetic_asdm_module.mosaic_spec(
        name="uid___A002_X1234_X" + "".join(code.lower() for code in codes)
    )
    spec.fields = [
        dataclasses.replace(field, direction_code=code)
        for field, code in zip(spec.fields, codes, strict=True)
    ]
    truth = make_synthetic_asdm(spec)
    return open_asdm(truth.path)


def test_partition_with_unsupported_phase_center_frame_skipped_at_open(
    synthetic_asdm_module, make_synthetic_asdm, caplog
):
    """K13: a partition whose phase centre frame has no UVW support is skipped
    (and logged) when the processing set is opened. The other partitions open
    and their UVW can be computed (before, the partition opened and its UVW
    raised NotImplementedError at compute time)."""
    with caplog.at_level(logging.WARNING):
        ps_xdt = _open_asdm_with_field_codes(
            synthetic_asdm_module, make_synthetic_asdm, ["ICRS", "HADEC", "ICRS"]
        )

    assert "UVW cannot be calculated" in caplog.text
    assert "1 of 3 partitions" in caplog.text
    field_names = set()
    for node in ps_xdt.children.values():
        field_names.update(map(str, np.unique(node.ds.field_name.values)))
        phase_center = node["field_and_source_base_xds"].ds.FIELD_PHASE_CENTER_DIRECTION
        assert phase_center.attrs["frame"] == "icrs"
        uvw = node.ds.UVW.values
        assert uvw.shape == (node.ds.sizes["time"], node.ds.sizes["baseline_id"], 3)
        assert np.all(np.isfinite(uvw))
    assert field_names == {"J1229+0203_0", "M100_2"}


def test_all_partitions_with_unsupported_phase_center_frame(
    synthetic_asdm_module, make_synthetic_asdm
):
    """K13: when no partition has a phase centre usable for UVW, opening fails
    (RuntimeError) instead of returning partitions whose UVW cannot be
    computed."""
    with pytest.raises(RuntimeError, match="None of the 3 partitions"):
        _open_asdm_with_field_codes(
            synthetic_asdm_module, make_synthetic_asdm, ["ITRF", "AZELNE", "HADEC"]
        )


def test_field_direction_frame_unsupported():
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0, direction_code="APP")
    add_source(asdm, "Target")
    with pytest.raises(ValueError, match="APP is not supported"):
        create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)


def test_fields_with_different_frames_rejected():
    asdm = pyasdm.ASDM()
    add_field(asdm, "A", source_id=0, direction_code="ICRS")
    add_field(asdm, "B", phase_dir=(2.0, 0.0), source_id=0, direction_code="J2000")
    add_source(asdm, "A")
    with pytest.raises(ValueError, match="different direction frames"):
        create_field_and_source_xds(asdm, {"fieldId": [0, 1]}, 0, False)


def test_source_direction_unsupported_frame_is_omitted(mock_logger):
    asdm = pyasdm.ASDM()
    add_field(asdm, "Target", source_id=0)
    add_source(asdm, "Target", direction_code="APP")

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    assert "SOURCE_DIRECTION" not in xds
    assert list(xds.source_name.values) == ["Target"]
    assert any(
        "SOURCE_DIRECTION not included" in m for m in warnings_logged(mock_logger)
    )


def test_polynomial_and_ephemeris_directions_warn(mock_logger):
    """F71: the zeroth-order term of the direction polynomial is used, with a
    warning for numPoly > 1 and for ephemerisId."""
    asdm = pyasdm.ASDM()
    add_field(
        asdm,
        "Mars",
        source_id=0,
        phase_dir_terms=[(1.0, -0.5), (1e-5, 2e-5)],
        ephemeris_id=0,
    )
    add_source(asdm, "Mars")

    xds = create_field_and_source_xds(asdm, {"fieldId": [0]}, 0, False)
    assert_no_schema_issues(xds)
    np.testing.assert_array_equal(
        xds.FIELD_PHASE_CENTER_DIRECTION.values, [[1.0, -0.5]]
    )
    assert xds.attrs["is_ephemeris"] is False
    messages = warnings_logged(mock_logger)
    assert any("numPoly=2" in msg for msg in messages)
    assert any("ephemerisId=0" in msg for msg in messages)
