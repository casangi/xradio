import pytest

import pyasdm


def make_asdm_empty():
    test_asdm = pyasdm.ASDM()
    return test_asdm


@pytest.fixture(scope="session")
def asdm_empty():
    return make_asdm_empty()


def make_asdm_with_spw_default():
    test_asdm = pyasdm.ASDM()
    test_spw_table = test_asdm.getSpectralWindow()
    row_default = test_spw_table.newRowDefault()
    test_spw_table.add(row_default)
    assert test_spw_table.size() == 1
    return test_asdm


@pytest.fixture(scope="session")
def asdm_with_spw_default():
    return make_asdm_with_spw_default()


def make_asdm_with_spw_simple():
    spw_row_spec_0 = """
  <row>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
    <basebandName>BB_1</basebandName>
    <netSideband>LSB</netSideband>
    <numChan> 1 </numChan>
    <refFreq> 8.6021E10 </refFreq>
    <sidebandProcessingMode>NONE</sidebandProcessingMode>
    <totBandwidth> 2.0E9 </totBandwidth>
    <windowFunction>UNIFORM</windowFunction>
    <chanFreqArray> 1 1 8.5021E10  </chanFreqArray>
    <chanWidth> 2.0E9 </chanWidth>
    <effectiveBw> 2.0E9 </effectiveBw>
    <name> X0000000000#ALMA_RB_03#BB_1#SQLD </name>
    <resolution> 2.0E9 </resolution>
    <numAssocValues> 1 </numAssocValues>
    <assocNature> 1 1 BASEBAND_WIDE</assocNature>
    <assocSpectralWindowId> 1 1 SpectralWindow_0  </assocSpectralWindowId>
  </row>
    """
    spw_row_spec_8 = """
  <row>
    <spectralWindowId> SpectralWindow_8 </spectralWindowId>
    <basebandName>BB_1</basebandName>
    <netSideband>LSB</netSideband>
    <numChan> 128 </numChan>
    <numBin> 1 </numBin>
    <refFreq> 9.702100190734863E10 </refFreq>
    <measFreqRef> GALACTO </measFreqRef>
    <sidebandProcessingMode>NONE</sidebandProcessingMode>
    <totBandwidth> 2.0E9 </totBandwidth>
    <windowFunction>HANNING</windowFunction>
    <chanFreqStart> 9.701318940734863E10 </chanFreqStart>
    <chanFreqStep> -1.5625E7 </chanFreqStep>
    <chanWidth> 1.5625E7 </chanWidth>
    <effectiveBw> 4.1671875E7 </effectiveBw>
    <name> X0000000000#ALMA_RB_03#BB_1#SW-01#FULL_RES </name>
    <quantization> false </quantization>
    <refChan> -0.5 </refChan>
    <resolution> 3.125E7 </resolution>
    <numAssocValues> 4 </numAssocValues>
    <assocNature> 1 4 BASEBAND_WIDE BASEBAND_WIDE BASEBAND_WIDE CHANNEL_AVERAGE</assocNature>
    <assocSpectralWindowId> 1 4 SpectralWindow_5 SpectralWindow_6 SpectralWindow_7 SpectralWindow_9  </assocSpectralWindowId>
  </row>
    """
    test_asdm = pyasdm.ASDM()
    test_spw_table = test_asdm.getSpectralWindow()
    spw_row_0 = pyasdm.SpectralWindowRow(test_spw_table)
    spw_row_0.setFromXML(spw_row_spec_0)
    test_spw_table.add(spw_row_0)
    spw_row_8 = pyasdm.SpectralWindowRow(test_spw_table)
    spw_row_8.setFromXML(spw_row_spec_8)
    test_spw_table.add(spw_row_8)
    assert test_spw_table.size() == 2
    return test_asdm


@pytest.fixture(scope="session")
def asdm_with_spw_simple():
    """
    Meant for tests that do not need more than this minimal set of
    tables.
    """
    asdm = make_asdm_with_spw_simple()
    # Needed for test_open_partition / TODO: split to another one
    add_sbsummary_table(asdm)
    add_processor_table(asdm)
    return asdm


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


def make_asdm_with_main_execblock_config_processor_sbsummary():
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    add_execblock_table(asdm)
    add_config_description_table(asdm)
    add_processor_table(asdm)
    add_sbsummary_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main_execblock_config_processor_sbsummary():
    """
    Meant for tests of the info dicts (processor info, etc.)
    """
    return make_asdm_with_main_execblock_config_processor_sbsummary()
