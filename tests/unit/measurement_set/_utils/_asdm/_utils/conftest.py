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
    spw_row_spec = """
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
    test_asdm = pyasdm.ASDM()
    test_spw_table = test_asdm.getSpectralWindow()
    spw_row = pyasdm.SpectralWindowRow(test_spw_table)
    spw_row.setFromXML(spw_row_spec)
    test_spw_table.add(spw_row)
    assert test_spw_table.size() == 1
    return test_asdm
    
@pytest.fixture(scope="session")
def asdm_with_spw_simple():
    return make_asdm_with_spw_simple()
