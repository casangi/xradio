import pytest


def test_ensure_spw_name_conforms(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import ensure_spw_name_conforms

    spw_id = 2
    spw_name = ensure_spw_name_conforms("", spw_id)
    assert spw_name == f"spw_{spw_id}"

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
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_frequency_centers

    with pytest.raises(AttributeError, match="has no attribute"):
        centers = get_spw_frequency_centers(asdm_empty, 0, 64)

def test_get_spw_frequency_centers_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_frequency_centers

    with pytest.raises(NameError, match="chanFreqArray"):
        centers = get_spw_frequency_centers(asdm_with_spw_default, 0, 1)

def test_get_spw_frequency_centers_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_spw_frequency_centers

    centers_0 = get_spw_frequency_centers(asdm_with_spw_simple, 0, 1)
    assert centers_0 == [85021000000.0]
    centers_1 = get_spw_frequency_centers(asdm_with_spw_simple, 1, 128)
    assert len(centers_1) == 128
    assert centers_1.max() == 97013189407.34863
    assert centers_1.min() == 95028814407.34863
    with pytest.raises(AttributeError, match="has no attribute"):
        get_spw_frequency_centers(asdm_with_spw_simple, 8, 128)

def test_get_chan_width_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_chan_width

    with pytest.raises(AttributeError, match="has no attribute"):
        chan_width = get_chan_width(asdm_empty, 0)

def test_get_chan_width_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_chan_width

    with pytest.raises(NameError, match="chanWidthArray"):
        chan_width = get_chan_width(asdm_with_spw_default, 0)

def test_get_chan_width_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_chan_width

    chan_width = get_chan_width(asdm_with_spw_simple, 0)
    assert chan_width == 2000000000.0

def test_get_reference_frame_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_reference_frame

    with pytest.raises(AttributeError, match="has no attribute"):
        ref_frame = get_reference_frame(asdm_empty, 0)

def test_get_reference_frame_default(asdm_with_spw_default):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_reference_frame

    ref_frame = get_reference_frame(asdm_with_spw_default, 0)
    assert ref_frame == "TOPO"

def test_get_reference_frame_simple(asdm_with_spw_simple):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_reference_frame

    ref_frame = get_reference_frame(asdm_with_spw_simple, 0)
    assert ref_frame == "TOPO"
