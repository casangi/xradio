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

def test_get_chan_width_empty(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_chan_width

    with pytest.raises(AttributeError, match="has no attribute"):
        chan_width = get_chan_width(asdm_empty, 0)

def test_get_reference_frame(asdm_empty):
    from xradio.measurement_set._utils._asdm._utils.spectral_window import get_reference_frame

    with pytest.raises(AttributeError, match="has no attribute"):
        ref_frame = get_reference_frame(asdm_empty, 0)
