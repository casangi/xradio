import pytest

import pyasdm
#from pyasdm import ASDM
#from pyasdm import SpectralWindowTable
#from pyasdm import SpectralWindowRow


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
    return make_asdm_with_spw_simple()
