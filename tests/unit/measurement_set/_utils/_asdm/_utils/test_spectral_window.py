import numpy as np
import pyasdm
import pytest


def make_asdm_with_one_spw(num_chan: int, chan_freq_xml: str) -> pyasdm.ASDM:
    """ASDM with a single SpectralWindow row (SpectralWindow_0).

    chan_freq_xml gives the chanFreqStart/chanFreqStep/chanFreqArray elements.
    """
    spw_row_xml = f"""
  <row>
    <spectralWindowId> SpectralWindow_0 </spectralWindowId>
    <basebandName>BB_1</basebandName>
    <netSideband>USB</netSideband>
    <numChan> {num_chan} </numChan>
    <refFreq> 1.0E11 </refFreq>
    <sidebandProcessingMode>NONE</sidebandProcessingMode>
    <totBandwidth> 2.0E9 </totBandwidth>
    <windowFunction>HANNING</windowFunction>
    {chan_freq_xml}
    <chanWidth> 1.5625E7 </chanWidth>
    <numAssocValues> 0 </numAssocValues>
  </row>"""
    asdm = pyasdm.ASDM()
    spw_table = asdm.getSpectralWindow()
    spw_row = pyasdm.SpectralWindowRow(spw_table)
    spw_row.setFromXML(spw_row_xml)
    spw_table.add(spw_row)
    return asdm


def test_ensure_spw_name_conforms(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        ensure_spw_name_conforms,
    )

    spw_id = 2
    spw_name = ensure_spw_name_conforms("", spw_id)
    assert spw_name == f"spw_{spw_id}"

    name_prefix = "Test_SPW_Name"
    spw_name = ensure_spw_name_conforms(name_prefix, spw_id)
    assert spw_name == f"{name_prefix}_{spw_id}"


def test_get_spw_name_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_name

    with pytest.raises(AttributeError, match="has no attribute"):
        get_spw_name(asdm_empty, 1)


def test_get_spw_name_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_name

    name = get_spw_name(asdm_with_spw_default, 0)
    assert name is None


def test_get_spw_name_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_name

    name = get_spw_name(asdm_with_spw_simple, 0)
    assert name == "X0000000000#ALMA_RB_03#BB_1#SQLD"


def test_get_spw_frequency_centers_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    with pytest.raises(AttributeError, match="has no attribute"):
        get_spw_frequency_centers(asdm_empty, 0, 64)


def test_get_spw_frequency_centers_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    with pytest.raises(ValueError, match="chanFreqArray"):
        get_spw_frequency_centers(asdm_with_spw_default, 0, 1)


def test_get_spw_frequency_centers_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    # chanFreqArray
    centers_0 = get_spw_frequency_centers(asdm_with_spw_simple, 0, 1)
    assert isinstance(centers_0, np.ndarray)
    assert centers_0.dtype == np.float64
    np.testing.assert_array_equal(centers_0, [85021000000.0])
    with pytest.raises(RuntimeError, match="channels"):
        get_spw_frequency_centers(asdm_with_spw_simple, 0, 64)

    # chanFreqStart = 9.701318940734863E10, chanFreqStep = -1.5625E7 (LSB)
    centers_1 = get_spw_frequency_centers(asdm_with_spw_simple, 1, 128)
    assert isinstance(centers_1, np.ndarray)
    assert centers_1.dtype == np.float64
    assert len(centers_1) == 128
    assert centers_1[0] == 97013189407.34863
    assert centers_1.max() == 97013189407.34863
    assert centers_1.min() == 95028814407.34863
    np.testing.assert_array_equal(
        centers_1, 97013189407.34863 - 1.5625e7 * np.arange(128)
    )
    # numChan of the SPW is 128
    with pytest.raises(RuntimeError, match="numChan is 128"):
        get_spw_frequency_centers(asdm_with_spw_simple, 1, 127)
    with pytest.raises(AttributeError, match="has no attribute"):
        get_spw_frequency_centers(asdm_with_spw_simple, 8, 128)


@pytest.mark.parametrize(
    "num_chan, freq_start, freq_step",
    [
        # With a float np.arange(start, start + n * step, step) these gave n + 1
        # frequencies (positive steps across a 2^k Hz boundary)
        (128, 135495955285.89311, 15625000.0),
        (64, 135495955285.89311, 31250000.0),
        (8192, 182505152652.58627, 15625000.0),
        (4080, 430522442376.83344, 31250000.0),
        # negative step (LSB)
        (128, 97013189407.34863, -15625000.0),
        (1, 85021000000.0, 2.0e9),
    ],
)
def test_get_spw_frequency_centers_start_step_exact_num_chan(
    num_chan, freq_start, freq_step
):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    asdm = make_asdm_with_one_spw(
        num_chan,
        f"<chanFreqStart> {freq_start!r} </chanFreqStart>"
        f"<chanFreqStep> {freq_step!r} </chanFreqStep>",
    )
    centers = get_spw_frequency_centers(asdm, 0, num_chan)

    assert isinstance(centers, np.ndarray)
    assert centers.dtype == np.float64
    assert centers.shape == (num_chan,)
    # channel i is at chanFreqStart + i * chanFreqStep
    np.testing.assert_array_equal(
        centers, [freq_start + freq_step * chan for chan in range(num_chan)]
    )
    assert centers[0] == freq_start
    np.testing.assert_allclose(
        centers[-1], freq_start + (num_chan - 1) * freq_step, rtol=0, atol=1e-3
    )
    if num_chan > 1:
        np.testing.assert_allclose(np.diff(centers), freq_step, rtol=0, atol=1e-3)

    with pytest.raises(RuntimeError, match=f"numChan is {num_chan}"):
        get_spw_frequency_centers(asdm, 0, num_chan + 1)


def test_get_spw_frequency_centers_chan_freq_array():
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    freqs = [1.0e11, 1.0e11 + 3.0e6, 1.0e11 + 7.5e6, 1.0e11 + 1.2e7]
    asdm = make_asdm_with_one_spw(
        4,
        "<chanFreqArray> 1 4 "
        + " ".join(repr(freq) for freq in freqs)
        + " </chanFreqArray>",
    )
    centers = get_spw_frequency_centers(asdm, 0, 4)
    assert isinstance(centers, np.ndarray)
    assert centers.dtype == np.float64
    np.testing.assert_array_equal(centers, freqs)

    with pytest.raises(RuntimeError, match="channels"):
        get_spw_frequency_centers(asdm, 0, 3)


def test_get_spw_frequency_centers_chan_freq_array_length_mismatch():
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    # numChan says 4, but chanFreqArray has 3 frequencies
    asdm = make_asdm_with_one_spw(
        4, "<chanFreqArray> 1 3 1.0E11 1.1E11 1.2E11 </chanFreqArray>"
    )
    with pytest.raises(RuntimeError, match="3 channel frequencies"):
        get_spw_frequency_centers(asdm, 0, 4)


def test_get_spw_frequency_centers_start_without_step():
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    # chanFreqStart without chanFreqStep, and no chanFreqArray
    asdm = make_asdm_with_one_spw(
        128, "<chanFreqStart> 135495955285.89311 </chanFreqStart>"
    )
    with pytest.raises(ValueError, match="chanFreqStep present: False"):
        get_spw_frequency_centers(asdm, 0, 128)

    # chanFreqStart without chanFreqStep, falls back to chanFreqArray
    asdm = make_asdm_with_one_spw(
        2,
        "<chanFreqStart> 1.0E11 </chanFreqStart>"
        "<chanFreqArray> 1 2 1.0E11 1.5E11 </chanFreqArray>",
    )
    centers = get_spw_frequency_centers(asdm, 0, 2)
    np.testing.assert_array_equal(centers, [1.0e11, 1.5e11])


def test_get_spw_frequency_centers_start_step_preferred_over_array():
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_spw_frequency_centers,
    )

    asdm = make_asdm_with_one_spw(
        2,
        "<chanFreqStart> 1.0E11 </chanFreqStart>"
        "<chanFreqStep> 1.0E6 </chanFreqStep>"
        "<chanFreqArray> 1 2 1.0E11 1.000001E11 </chanFreqArray>",
    )
    centers = get_spw_frequency_centers(asdm, 0, 2)
    np.testing.assert_array_equal(centers, [1.0e11, 1.0e11 + 1.0e6])


def test_get_chan_width_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_chan_width,
    )

    with pytest.raises(AttributeError, match="has no attribute"):
        get_chan_width(asdm_empty, 0)


def test_get_chan_width_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_chan_width,
    )

    with pytest.raises(ValueError, match="chanWidthArray"):
        get_chan_width(asdm_with_spw_default, 0)


def test_get_chan_width_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_chan_width,
    )

    chan_width = get_chan_width(asdm_with_spw_simple, 0)
    assert chan_width == 2000000000.0


def test_get_reference_frame_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_reference_frame,
    )

    with pytest.raises(AttributeError, match="has no attribute"):
        get_reference_frame(asdm_empty, 0)


def test_get_reference_frame_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_reference_frame,
    )

    ref_frame = get_reference_frame(asdm_with_spw_default, 0)
    assert ref_frame == "TOPO"


def test_get_reference_frame_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import (
        get_reference_frame,
    )

    ref_frame = get_reference_frame(asdm_with_spw_simple, 0)
    assert ref_frame == "TOPO"
    ref_frame = get_reference_frame(asdm_with_spw_simple, 1)
    assert ref_frame == "GALACTO"
