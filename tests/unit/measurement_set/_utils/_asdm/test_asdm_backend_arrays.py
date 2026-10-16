from contextlib import nullcontext as no_raises
from unittest import mock

import numpy as np
import pyasdm
import pytest


@pytest.mark.parametrize(
    "input_shape, input_dtype, input_key, expected_error",
    [
        (None, None, None, pytest.raises(NotImplementedError, match="^$")),
        (
            (2, 2),
            np.float32,
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
    "dim_indexers",
    [
        {"time": 0},
        {"time": -1, "polarization": 1},
        {"time": slice(None, None, 2)},
        {"frequency": slice(1, 8, 3), "baseline_id": 4},
        {"baseline_id": slice(5, 0, -2)},
        {"time": slice(2, 2, 2)},
    ],
)
def test_ASDMBackendArray_basic_indexing(dim_indexers):
    import xarray as xr

    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import (
        ASDMBackendArray,
    )

    data = np.arange(7 * 6 * 8 * 2, dtype="float64").reshape(7, 6, 8, 2)
    raw_keys = []

    class SimpleNumpyBackendArray(ASDMBackendArray):
        def _raw_indexing_method(self, key):
            raw_keys.append(key)
            return data[key]

    dims = ("time", "baseline_id", "frequency", "polarization")
    array = xr.DataArray(
        xr.core.indexing.LazilyIndexedArray(
            SimpleNumpyBackendArray(data.shape, data.dtype)
        ),
        dims=dims,
    )
    expected = data[tuple(dim_indexers.get(dim, slice(None)) for dim in dims)]
    np.testing.assert_array_equal(array.isel(dim_indexers).values, expected)
    assert all(
        isinstance(dim_key, slice) and dim_key.step in (None, 1)
        for key in raw_keys
        for dim_key in key
    )


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
        ((None,) * 4, (2, 1), pytest.raises(TypeError, match="not supported")),
        ((1, 5, 2, 1), (2, 1), no_raises()),
        ((2, 10, 16, 2), (2, 1), no_raises()),
        ((2, 3, 16, 2), (1, 1), no_raises()),
        ((2, 2, 16, 2), (0, 1), no_raises()),
        ((2, 1, 16, 2), (0, 1), no_raises()),
    ],
)
def test_WeightArray__basic_indexing_method(
    input_shape, expected_output_shape, expected_error
):
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import WeightArray

    weight = WeightArray(input_shape)
    assert weight.shape == input_shape
    assert weight.dtype == np.dtype("float32")
    with expected_error:
        key = (0, slice(2, 4), 0, slice(0, 1))
        weight_selected = weight._basic_indexing_method(key)
        assert isinstance(weight_selected, np.ndarray)
        assert weight_selected.dtype == np.float32
        assert weight_selected.shape == expected_output_shape
        # No true indexing possible here without true BDFs / or
        # overly mocked
        # weight_selected = weight[0, 0, 0, 0:1]
        # Or could do weight[xr.core.indexing.BasicIndexer(...)]


def test_WeightArray_selection_basic_and_raw_methods():
    from xradio.measurement_set._utils._asdm.asdm_backend_arrays import WeightArray

    weight = WeightArray((20, 10, 30, 4))
    with mock.patch(
        "xradio.measurement_set._utils._asdm.asdm_backend_arrays.np.broadcast_to",
        wraps=np.broadcast_to,
    ) as mock_broadcast_to:
        selected_basic = weight._basic_indexing_method((5, slice(0, 2), 3, slice(None)))
        selected_raw = weight._raw_indexing_method((5, slice(0, 2), 3, slice(None)))

    mock_broadcast_to.assert_called_with(
        np.ones((), dtype=np.dtype("float32")), shape=(2, 4)
    )
    assert selected_basic.shape == (2, 4)
    assert (selected_basic == 1).all()
    assert selected_raw.shape == (2, 4)
    assert (selected_raw == 1).all()


@pytest.mark.parametrize(
    "input_shape, input_bdf_paths, input_bdf_spw_id, input_time_indices, input_key, expected_error",
    [
        (None, [], 0, {}, None, pytest.raises(KeyError, match="bdf_names")),
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
