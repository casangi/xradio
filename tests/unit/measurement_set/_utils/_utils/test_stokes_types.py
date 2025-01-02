from xradio.measurement_set._utils._utils.stokes_types import stokes_types


def test_stokes_types():

    for key in range(0, 33):
        assert key in stokes_types
