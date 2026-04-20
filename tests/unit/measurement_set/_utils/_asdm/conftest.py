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
    return asdm


@pytest.fixture(scope="session")
def asdm_with_execblock_spw_simple():
    asdm = make_asdm_with_spw_simple()
    # Needed for test_open_partition / TODO: split to another one
    add_sbsummary_table(asdm)
    add_processor_table(asdm)
    add_execblock_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_execblock_processor_sbsummary_spw_simple():
    asdm = make_asdm_with_spw_simple()
    # Needed for test_open_partition / TODO: split to another one
    add_sbsummary_table(asdm)
    add_processor_table(asdm)
    add_execblock_table(asdm)
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


# Rename?: min_for_open_partition_to_work
def make_asdm_with_main_execblock_config_processor_sbsummary():
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    add_execblock_table(asdm)
    add_config_description_table(asdm)
    add_processor_table(asdm)
    add_sbsummary_table(asdm)
    add_antenna_station_tables(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main_execblock_config_processor_sbsummary():
    """
    Meant for tests of the info dicts (processor info, etc.)
    """
    return make_asdm_with_main_execblock_config_processor_sbsummary()


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


def make_asdm_with_main_data_description_config_description_polarization():
    """
    Produces an ASDM with a minimum set of tables for create_coordinates.
    """
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    add_data_description_table(asdm)
    add_config_description_table(asdm)
    add_polarization_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main_data_description_config_description_polarization():
    return make_asdm_with_main_data_description_config_description_polarization()


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


def make_asdm_with_main_etc_data_description_polarization_field_source():
    """
    Produces an ASDM with a minimum set of tables for open_partition.
    """
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    add_antenna_station_tables(asdm)
    add_execblock_table(asdm)
    add_processor_table(asdm)
    add_sbsummary_table(asdm)
    add_data_description_table(asdm)
    add_config_description_table(asdm)
    add_polarization_table(asdm)
    add_field_table(asdm)
    add_source_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main_etc_data_description_polarization_field_source():
    """
    Produces an ASDM with a minimum set of tables for create_coordinates.
    """
    return make_asdm_with_main_etc_data_description_polarization_field_source()


def add_feed_table(asdm: pyasdm.ASDM):
    feed_row_0_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_0 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_row_1_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_1 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_table = asdm.getFeed()
    feed_row_0 = pyasdm.FeedRow(feed_table)
    feed_row_0.setFromXML(feed_row_0_xml)
    feed_table.add(feed_row_0)
    feed_row_1 = pyasdm.FeedRow(feed_table)
    feed_row_1.setFromXML(feed_row_1_xml)
    feed_table.add(feed_row_1)


def add_antenna_station_tables(asdm: pyasdm.ASDM):
    antenna_row_0_xml = """
  <row>
    <antennaId> Antenna_0 </antennaId>
    <name> CM01 </name>
    <antennaMake>MITSUBISHI_7</antennaMake>
    <antennaType>GROUND_BASED</antennaType>
    <dishDiameter> 7.0 </dishDiameter>
    <position> 1 3 -0.002052 -2.32E-4 7.502983  </position>
    <offset> 1 3 0.0 0.0 0.0  </offset>
    <time> 5230000552242000000 </time>
    <stationId> Station_0 </stationId>
  </row>
    """
    antenna_row_1_xml = """
  <row>
    <antennaId> Antenna_1 </antennaId>
    <name> CM03 </name>
    <antennaMake>MITSUBISHI_7</antennaMake>
    <antennaType>GROUND_BASED</antennaType>
    <dishDiameter> 7.0 </dishDiameter>
    <position> 1 3 -0.001862 0.001218 7.49941  </position>
    <offset> 1 3 0.0 0.0 0.0  </offset>
    <time> 5230000552242000000 </time>
    <stationId> Station_1 </stationId>
  </row>
    """
    antenna_table = asdm.getAntenna()
    antenna_row_0 = pyasdm.AntennaRow(antenna_table)
    antenna_row_0.setFromXML(antenna_row_0_xml)
    antenna_table.add(antenna_row_0)
    antenna_row_1 = pyasdm.AntennaRow(antenna_table)
    antenna_row_1.setFromXML(antenna_row_1_xml)
    antenna_table.add(antenna_row_1)

    station_row_0_xml = """
  <row>
    <stationId> Station_0 </stationId>
    <name> N602 </name>
    <position> 1 3 2225077.740418 -5440126.559719 -2481521.871323  </position>
    <type>ANTENNA_PAD</type>
  </row>
"""
    station_row_1_xml = """
  <row>
    <stationId> Station_1 </stationId>
    <name> J503 </name>
    <position> 1 3 2225074.121158 -5440116.537775 -2481546.908647  </position>
    <type>ANTENNA_PAD</type>
  </row>

    """
    station_table = asdm.getStation()
    station_row_0 = pyasdm.StationRow(station_table)
    station_row_0.setFromXML(station_row_0_xml)
    station_table.add(station_row_0)
    station_row_1 = pyasdm.StationRow(station_table)
    station_row_1.setFromXML(station_row_1_xml)
    station_table.add(station_row_1)


@pytest.fixture(scope="session")
def asdm_with_execblock_antenna_station_feed():
    """
    Meant for create_antenna_xds tests
    """
    asdm = make_asdm_with_spw_simple()
    add_execblock_table(asdm)
    add_antenna_station_tables(asdm)
    add_feed_table(asdm)
    return asdm


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


@pytest.fixture(scope="session")
def asdm_with_polarization(asdm_with_spw_simple):
    """
    Meant for tests on arrays of arrays present for example in the
    polarization table (the 'corrProduct')
    """
    asdm = make_asdm_with_spw_simple()
    add_polarization_table(asdm)
    return asdm


def add_simple_feed_table(asdm: pyasdm.ASDM):
    feed_row_0_xml = """
  <row>
    <feedId> 0 </feedId>
    <timeInterval> 7226686294548387903 3993371484612775807 </timeInterval>
    <numReceptor> 2 </numReceptor>
    <beamOffset> 2 2 2 0.0 0.0 0.0 0.0  </beamOffset>
    <focusReference> 2 2 3 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0 -99999.0  </focusReference>
    <polarizationTypes> 1 2 X Y</polarizationTypes>
    <polResponse> 2 2 2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  </polResponse>
    <receptorAngle> 1 2 -0.9346238144 0.6361725124  </receptorAngle>
    <antennaId> Antenna_0 </antennaId>
    <receiverId> 1 2 0 0  </receiverId>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
  </row>
    """
    feed_table = asdm.getFeed()
    feed_row_0 = pyasdm.FeedRow(feed_table)
    feed_row_0.setFromXML(feed_row_0_xml)
    feed_table.add(feed_row_0)


@pytest.fixture(scope="session")
def asdm_with_simple_feed(asdm_with_spw_simple):
    asdm = make_asdm_with_spw_simple()
    add_simple_feed_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main(asdm_with_spw_simple):
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    return asdm


@pytest.fixture(scope="session")
def asdm_with_main_config(asdm_with_spw_simple):
    asdm = make_asdm_with_spw_simple()
    add_main_table(asdm)
    add_config_description_table(asdm)
    return asdm


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


@pytest.fixture(scope="session")
def mock_asdm_set_from_file():
    """
    Meant to mock the asdm.setFromFile() function of pyasdm, for open_asdm to be able to run completely but
    without using any real I/O.
    """

    def _function_mock_asdm_set_from_file(self, _directory: str):
        """
        This function is meant to mock asdm.setFromFile(), and needs to operate on an asdm object previously
        instantiated.
        """
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

    return _function_mock_asdm_set_from_file
