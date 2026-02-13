import pytest

import numpy as np
import pandas as pd
import xarray as xr

import pyasdm

from xradio.measurement_set._utils._asdm.open_asdm import open_asdm


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


def add_config_description_table(asdm: pyasdm.ASDM):
    config_description_row_0_xml = """
  <row>
    <numAntenna> 2 </numAntenna>
    <numDataDescription> 4 </numDataDescription>
    <numFeed> 1 </numFeed>
    <correlationMode>AUTO_ONLY</correlationMode>
    <configDescriptionId> ConfigDescription_0 </configDescriptionId>
    <numAtmPhaseCorrection> 1 </numAtmPhaseCorrection>
    <atmPhaseCorrection> 1 1 AP_UNCORRECTED</atmPhaseCorrection>
    <processorType>RADIOMETER</processorType>
    <spectralType>BASEBAND_WIDE</spectralType>
    <antennaId> 1 12 Antenna_0 Antenna_1 Antenna_2 Antenna_3 Antenna_4 Antenna_5 Antenna_6 Antenna_7 Antenna_8 Antenna_9 Antenna_10 Antenna_11  </antennaId>
    <dataDescriptionId> 1 4 DataDescription_0 DataDescription_1 DataDescription_2 DataDescription_3  </dataDescriptionId>
    <feedId> 1 12 0 0 0 0 0 0 0 0 0 0 0 0  </feedId>
    <processorId> Processor_0 </processorId>
    <switchCycleId> 1 4 SwitchCycle_0 SwitchCycle_0 SwitchCycle_0 SwitchCycle_0  </switchCycleId>
  </row>
    """
    config_description_table = asdm.getConfigDescription()
    config_description_row_0 = pyasdm.ConfigDescriptionRow(config_description_table)
    config_description_row_0.setFromXML(config_description_row_0_xml)
    config_description_table.add(config_description_row_0)


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


def add_spw_table(asdm: pyasdm.ASDM):
    spw_row_0_xml = """
  <row>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
    <basebandName>BB_1</basebandName>
    <netSideband>LSB</netSideband>
    <numChan> 1 </numChan>
    <refFreq> 2.84021E11 </refFreq>
    <sidebandProcessingMode>NONE</sidebandProcessingMode>
    <totBandwidth> 2.0E9 </totBandwidth>
    <windowFunction>UNIFORM</windowFunction>
    <chanFreqArray> 1 1 2.83021E11  </chanFreqArray>
    <chanWidth> 2.0E9 </chanWidth>
    <effectiveBw> 2.0E9 </effectiveBw>
    <name> X662533328#ALMA_RB_07#BB_1#SQLD </name>
    <resolution> 2.0E9 </resolution>
    <numAssocValues> 1 </numAssocValues>
    <assocNature> 1 1 BASEBAND_WIDE</assocNature>
    <assocSpectralWindowId> 1 1 SpectralWindow_0  </assocSpectralWindowId>
  </row>
    """
    spw_table = asdm.getSpectralWindow()
    spw_row_0 = pyasdm.SpectralWindowRow(spw_table)
    spw_row_0.setFromXML(spw_row_0_xml)
    spw_table.add(spw_row_0)


def add_scan_table(asdm: pyasdm.ASDM):
    scan_row_0_xml = """
  <row>
    <scanNumber> 1 </scanNumber>
    <startTime> 5230000639104000000 </startTime>
    <endTime> 5230000885728000000 </endTime>
    <numIntent> 2 </numIntent>
    <numSubscan> 10 </numSubscan>
    <scanIntent> 1 2 CALIBRATE_POINTING CALIBRATE_WVR</scanIntent>
    <calDataType> 1 2 CHANNEL_AVERAGE_CROSS WVR</calDataType>
    <calibrationOnLine> 1 2 true true  </calibrationOnLine>
    <calibrationFunction> 1 2 UNSPECIFIED UNSPECIFIED</calibrationFunction>
    <calibrationSet> 1 2 NONE NONE</calibrationSet>
    <calPattern> 1 2 FIVE_POINTS NONE</calPattern>
    <numField> 1 </numField>
    <fieldName> 1 1 &quot;J0423-0120&quot;  </fieldName>
    <sourceName> J0423-0120 </sourceName>
    <execBlockId> ExecBlock_0 </execBlockId>
  </row>
    """
    scan_table = asdm.getScan()
    scan_row_0 = pyasdm.ScanRow(scan_table)
    scan_row_0.setFromXML(scan_row_0_xml)
    scan_table.add(scan_row_0)


def add_execblock_table(asdm: pyasdm.ASDM):
    execblock_row_0_xml = """
  <row>
    <execBlockId> ExecBlock_0 </execBlockId>
    <startTime> 5230000552242000000 </startTime>
    <endTime> 5230001040525000000 </endTime>
    <execBlockNum> 999 </execBlockNum>
    <execBlockUID>
      <EntityRef entityId="uid://A002/X11b94a6/X119b" partId="X00000000" entityTypeName="ASDM" documentVersion="1"/>
    </execBlockUID>
    <projectUID>
      <EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/>
    </projectUID>
    <configName> 7M </configName>
    <telescopeName> ALMA </telescopeName>
    <observerName> riechers </observerName>
    <numObservingLog> 0 </numObservingLog>
    <observingLog> 1 0  </observingLog>
    <sessionReference>
      <EntityRef entityId="uid://A002/X11b94a6/X119a" partId="X00000000" entityTypeName="Session" documentVersion="1"/>
    </sessionReference>
    <baseRangeMin> 0.0 </baseRangeMin>
    <baseRangeMax> 0.0 </baseRangeMax>
    <baseRmsMinor> 0.0 </baseRmsMinor>
    <baseRmsMajor> 0.0 </baseRmsMajor>
    <basePa> 0.0 </basePa>
    <aborted> false </aborted>
    <numAntenna> 2 </numAntenna>
    <siteAltitude> 0.0 </siteAltitude>
    <siteLongitude> 0.0 </siteLongitude>
    <siteLatitude> 0.0 </siteLatitude>
    <observingScript> StandardInterferometry.py </observingScript>
    <antennaId> 1 12 Antenna_0 Antenna_1 Antenna_2 Antenna_3 Antenna_4 Antenna_5 Antenna_6 Antenna_7 Antenna_8 Antenna_9 Antenna_10 Antenna_11  </antennaId>
    <sBSummaryId> SBSummary_0 </sBSummaryId>
  </row>
    """
    execblock_table = asdm.getExecBlock()
    execblock_row_0 = pyasdm.ExecBlockRow(execblock_table)
    execblock_row_0.setFromXML(execblock_row_0_xml)
    execblock_table.add(execblock_row_0)


def add_processor_table(asdm: pyasdm.ASDM):
    processor_row_0_xml = """
  <row>
    <processorId> Processor_0 </processorId>
    <modeId> SquareLawDetector_3 </modeId>
    <processorType>RADIOMETER</processorType>
    <processorSubType>SQUARE_LAW_DETECTOR</processorSubType>
  </row>
    """
    processor_table = asdm.getProcessor()
    processor_row_0 = pyasdm.ProcessorRow(processor_table)
    processor_row_0.setFromXML(processor_row_0_xml)
    processor_table.add(processor_row_0)


def add_sbsummary_table(asdm: pyasdm.ASDM):
    sbsummary_row_0_xml = """
  <row>
    <sBSummaryId> SBSummary_0 </sBSummaryId>
    <sbSummaryUID>
      <EntityRef entityId="uid://A001/X362e/X332" partId="X00000000" entityTypeName="SchedBlock" documentVersion="1"/>
    </sbSummaryUID>
    <projectUID>
      <EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/>
    </projectUID>
    <obsUnitSetUID>
      <EntityRef entityId="uid://A001/X35fd/X21f" partId="X00000000" entityTypeName="ObsProject" documentVersion="1"/>
    </obsUnitSetUID>
    <frequency> 636.9642143006589 </frequency>
    <frequencyBand>ALMA_RB_09</frequencyBand>
    <sbType>OBSERVER</sbType>
    <sbDuration> 7200000000000 </sbDuration>
    <numObservingMode> 1 </numObservingMode>
    <observingMode> 1 1 &quot;Standard Interferometry&quot;  </observingMode>
    <numberRepeats> 1 </numberRepeats>
    <numScienceGoal> 11 </numScienceGoal>
    <scienceGoal> 1 11 &quot;representativeFrequency = 636.9642143006589 GHz&quot; &quot;minAcceptableAngResolution = 0.0 arcsec&quot; &quot;maxAcceptableAngResolution = 0.0 arcsec&quot; &quot;dynamicRange = 47.99883524702861&quot; &quot;representativeBandwidth = 2167.177596531372 MHz&quot; &quot;representativeSource = hers7&quot; &quot;sensitivityGoal = 5.5 mJy&quot; &quot;SBName = hers7_a_09_7M&quot; &quot;representativeWindow = X662533328#ALMA_RB_09#BB_2#SW-02#FULL_RES&quot; &quot;maxAllowedBeamAxialRatio = 999.999&quot; &quot;spectralDynamicRangeBandWidth = 2167.177596531372 MHz&quot;  </scienceGoal>
    <numWeatherConstraint> 3 </numWeatherConstraint>
    <weatherConstraint> 1 3 &quot;maxPWVC = 0.472 mm&quot; &quot;seeing = 0.0 arcsec&quot; &quot;phaseStability = 1.0 rad&quot;  </weatherConstraint>
    <centerDirection> 1 2 15.390375 0.5315833333333333  </centerDirection>
    <centerDirectionCode>ICRS</centerDirectionCode>
    <centerDirectionEquinox> 4453444735816000000 </centerDirectionEquinox>
  </row>
    """
    sbsummary_table = asdm.getSBSummary()
    sbsummary_row_0 = pyasdm.SBSummaryRow(sbsummary_table)
    sbsummary_row_0.setFromXML(sbsummary_row_0_xml)
    sbsummary_table.add(sbsummary_row_0)


def mock_asdm_set_from_file(self, _directory: str):
    add_main_table(self)
    add_config_description_table(self)
    add_data_description_table(self)
    add_polarization_table(self)
    add_field_table(self)
    add_spw_table(self)
    add_scan_table(self)
    add_execblock_table(self)
    add_processor_table(self)
    add_sbsummary_table(self)


def test_open_asdm_none():
    with pytest.raises(TypeError, match="expected"):
        open_asdm(None, ["fieldId"])


def test_open_asdm_empty(asdm_empty, monkeypatch):
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        lambda self, asdm_path: None,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="No partitions left"):
        open_asdm("/unused_path/foo", ["fieldId"])


def test_open_asdm_with_spw_default(asdm_with_spw_default, monkeypatch):

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    with pytest.raises(RuntimeError, match="No partitions left"):
        open_asdm("/unused_path/foo", [])


def test_open_asdm_with_spw_simple(asdm_with_spw_simple, monkeypatch):

    def mock_get_times_from_bdfs(
        bdf_paths: list[str], scans_metadata: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.1]), np.array([1.0]), np.array([0.101]), np.array([1.0])

    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_asdm.pyasdm.ASDM.setFromFile",
        mock_asdm_set_from_file,
    )
    monkeypatch.setattr(
        "pyasdm.MainRow.getBDFPath", lambda bdf_paths: "/monkypatched_path/foo"
    )
    monkeypatch.setattr(
        "xradio.measurement_set._utils._asdm.open_partition.get_times_from_bdfs",
        mock_get_times_from_bdfs,
    )

    ps_xdt = open_asdm(
        "/unused_path/foo",
        ["dataDescriptionId", "execBlockId", "fieldId", "scanIntent"],
        include_processor_types=["CORRELATOR", "SPECTROMETER", "RADIOMETER"],
    )
    assert isinstance(ps_xdt, xr.DataTree)
    assert ps_xdt.type == "processing_set"
    for _msv4_name, msv4_xdt in enumerate(ps_xdt):
        assert isinstance(msv4_xdt, str)
