from contextlib import nullcontext as no_raises

import numpy as np

import pytest

import pyasdm


@pytest.mark.parametrize(
    "input_shape, input_dtype, input_key, expected_error",
    [
        (None, None, None, pytest.raises(NotImplementedError, match="^$")),
        (
            (2, 2),
            np.float64,
            (slice(0, 1), slice(0, 2)),
            pytest.raises(NotImplementedError, match="^$"),
        ),
    ],
)
def test_ASDMBackendArray__raw_indexing_method(
    input_shape, input_dtype, input_key, expected_error
):
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import ASDMBackendArray

    backend_array = ASDMBackendArray(input_shape, input_dtype)
    assert backend_array.shape == input_shape
    assert backend_array.dtype == input_dtype
    with expected_error:
        indexed_array = backend_array._raw_indexing_method(input_key)
        assert isinstance(indexed_array, np.ndarray)
        if input_shape:
            assert indexed_array.shape == input_key


@pytest.mark.parametrize(
    "input_shape, input_bdf_paths, input_bdf_spw_id, input_time_indices, input_key, expected_error",
    [
        (None, [], 0, {}, None, pytest.raises(ValueError, match="need at least one")),
        (
            (2, 3, 6, 2),
            ["/foo/bdf1", "/foo/bdf2"],
            1,
            {"bdf_names": ["/foo/bdf1", "/foo/bdf2"], "bdf_start": [0, 2, 4]},
            (slice(0, 1), None, None, None),
            pytest.raises(
                pyasdm.exceptions.BDFReaderException, match="Error while opening"
            ),
        ),
    ],
)
def test_VisibilityArray__raw_indexing_method(
    input_shape,
    input_bdf_paths,
    input_bdf_spw_id,
    input_time_indices,
    input_key,
    expected_error,
):
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import VisibilityArray

    vis = VisibilityArray(
        input_shape, input_bdf_paths, input_bdf_spw_id, input_time_indices
    )
    assert vis.shape == input_shape
    assert vis.dtype == np.dtype("complex128")
    with expected_error:
        vis._raw_indexing_method(input_key)
        # No true indexing possible here without true BDFs / or
        # overly mocked


@pytest.mark.parametrize(
    "input_shape, expected_output_shape, expected_error",
    [
        (None, None, pytest.raises(TypeError, match="as shape arguments")),
        ((1, 5, 2, 1), None, no_raises()),
    ],
)
def test_WeightArray__raw_indexing_method(
    input_shape, expected_output_shape, expected_error
):
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import WeightArray

    weight = WeightArray(input_shape)
    assert weight.shape == input_shape
    assert weight.dtype == np.dtype("float64")
    with expected_error:
        key = (0, 0, 0, slice(0, 1))
        weight_selected = weight._raw_indexing_method(key)
        assert isinstance(weight_selected, np.ndarray)
        assert weight_selected.dtype == np.float64
        assert weight_selected.shape == (1,)  #  expected_output_shape
        # No true indexing possible here without true BDFs / or
        # overly mocked
        # weight_selected = weight[0, 0, 0, 0:1]


@pytest.mark.parametrize(
    "input_shape, input_bdf_paths, input_bdf_spw_id, input_time_indices, input_key, expected_error",
    [
        (None, [], 0, {}, None, pytest.raises(ValueError, match="need at least one")),
        (
            (10, 3, 8, 2),
            ["/foo/bdf1", "/foo/bdf2"],
            1,
            {"bdf_names": ["/foo/bdf1", "/foo/bdf2"], "bdf_start": [0, 2, 4]},
            (slice(0, 1), None, None, None),
            pytest.raises(
                pyasdm.exceptions.BDFReaderException, match="Error while opening"
            ),
        ),
    ],
)
def test_FlagArray__raw_indexing_method(
    input_shape,
    input_bdf_paths,
    input_bdf_spw_id,
    input_time_indices,
    input_key,
    expected_error,
):
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import FlagArray

    flag = FlagArray(input_shape, input_bdf_paths, input_bdf_spw_id, input_time_indices)
    assert flag.shape == input_shape
    assert flag.dtype == np.dtype("bool")
    with expected_error:
        flag._raw_indexing_method(input_key)


def test_UVWArray__raw_indexing_method():
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import UVWArray

    uvw = UVWArray(None, None, None, None, None, None)
    with pytest.raises(AttributeError, match="NoneType"):
        uvw._raw_indexing_method(None)
