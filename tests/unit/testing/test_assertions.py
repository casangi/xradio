import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xradio.testing import assert_xarray_datasets_equal


def _make_dataset():
    data = np.arange(6, dtype=float).reshape(2, 3)
    ds = xr.Dataset(
        data_vars={"var": (("x", "y"), data)},
        coords={"x": [10, 20], "y": [1, 2, 3]},
        attrs={"meta": {"source": "test", "version": 1}},
    )
    ds["var"].attrs["units"] = "K"
    return ds


def test_assert_xarray_datasets_equal_passes():
    test = _make_dataset()
    true = _make_dataset()
    assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_dim_mismatch():
    test = _make_dataset()
    true = _make_dataset().isel(y=slice(0, 2))
    with pytest.raises(AssertionError, match="Dimension check failed"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_dim_name_mismatch():
    test = _make_dataset()
    true = _make_dataset().rename({"y": "z"})
    with pytest.raises(AssertionError, match="Dimension check failed"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_value_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true = true.assign_coords(x=[11, 20])
    with pytest.raises(AssertionError, match="coord 'x' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_name_mismatch():
    test = _make_dataset()
    true = _make_dataset().rename({"x": "x2"})
    with pytest.raises(AssertionError, match="Dimension check failed"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_attr_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true.attrs["meta"]["version"] = 2
    with pytest.raises(
        AssertionError, match="dataset attrs\\['meta'\\]\\['version'\\]"
    ):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_attr_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true.coords["x"].attrs["units"] = "deg"
    with pytest.raises(AssertionError, match="coord 'x' attrs keys mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_attr_value_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test.coords["x"].attrs["units"] = "rad"
    true.coords["x"].attrs["units"] = "deg"
    with pytest.raises(AssertionError, match="coord 'x' attrs\\['units'\\]"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_value_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true["var"] = true["var"] + 1
    with pytest.raises(AssertionError, match="data_var 'var' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_key_mismatch():
    test = _make_dataset()
    true = _make_dataset().rename({"var": "var2"})
    with pytest.raises(AssertionError, match="Data variable keys mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_attr_key_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true["var"].attrs["long_name"] = "temperature"
    with pytest.raises(AssertionError, match="data_var 'var' attrs keys mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_attr_value_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["var"].attrs["units"] = "K"
    true["var"].attrs["units"] = "m"
    with pytest.raises(AssertionError, match="data_var 'var' attrs\\['units'\\]"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_dtype_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true["var"] = true["var"].astype(np.int64)
    with pytest.raises(AssertionError, match="data_var 'var' dtype mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_shape_mismatch():
    test = _make_dataset()
    true = _make_dataset().isel(y=slice(0, 2))
    with pytest.raises(AssertionError, match="Dimension check failed"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_data_var_dim_order_mismatch():
    test = _make_dataset()
    true = _make_dataset().transpose("y", "x")
    with pytest.raises(AssertionError, match="data_var 'var' dims mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_dtype_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    true = true.assign_coords(x=true.coords["x"].astype(np.float64))
    with pytest.raises(AssertionError, match="coord 'x' dtype mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_coord_shape_mismatch():
    test = _make_dataset()
    true = _make_dataset().isel(x=slice(0, 1))
    with pytest.raises(AssertionError, match="Dimension check failed"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_attr_type_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test.attrs["meta"]["version"] = 1
    true.attrs["meta"]["version"] = "1"
    with pytest.raises(
        AssertionError, match="dataset attrs\\['meta'\\]\\['version'\\]"
    ):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_attr_list_array_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test.attrs["meta"]["values"] = [1, 2, 3]
    true.attrs["meta"]["values"] = np.array([1, 2, 4])
    with pytest.raises(AssertionError, match="dataset attrs\\['meta'\\]\\['values'\\]"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_attr_numeric_tolerance():
    test = _make_dataset()
    true = _make_dataset()
    test.attrs["meta"]["offset"] = 1.0
    true.attrs["meta"]["offset"] = 1.0000005
    assert_xarray_datasets_equal(test, true, rtol=1e-5, atol=0.0)
    with pytest.raises(AssertionError, match="dataset attrs\\['meta'\\]\\['offset'\\]"):
        assert_xarray_datasets_equal(test, true, rtol=1e-9, atol=0.0)


def test_assert_xarray_datasets_equal_check_attrs_false():
    test = _make_dataset()
    true = _make_dataset()
    true.attrs["meta"]["version"] = 2
    assert_xarray_datasets_equal(test, true, check_attrs=False)


def test_assert_xarray_datasets_equal_non_numeric_data_var():
    test = _make_dataset()
    true = _make_dataset()
    test["label"] = xr.DataArray(
        np.array([["a", "b", "c"], ["d", "e", "f"]]), dims=("x", "y")
    )
    true["label"] = xr.DataArray(
        np.array([["a", "b", "c"], ["d", "e", "g"]]), dims=("x", "y")
    )
    with pytest.raises(AssertionError, match="data_var 'label' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_nan_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["var"] = test["var"].copy()
    true["var"] = true["var"].copy()
    test["var"][0, 0] = np.nan
    true["var"][0, 0] = 0.0
    with pytest.raises(AssertionError, match="data_var 'var' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_inf_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["var"] = test["var"].copy()
    true["var"] = true["var"].copy()
    test["var"][0, 0] = 1.0
    true["var"][0, 0] = np.inf
    with pytest.raises(AssertionError, match="data_var 'var' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_complex_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["var"] = test["var"].astype(np.complex128)
    true["var"] = true["var"].astype(np.complex128)
    true["var"][0, 0] = 1.0 + 2.0j
    test["var"][0, 0] = 1.0 + 3.0j
    with pytest.raises(AssertionError, match="data_var 'var' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_datetime_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["time"] = xr.DataArray(
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        dims=("x",),
    )
    true["time"] = xr.DataArray(
        np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]"),
        dims=("x",),
    )
    with pytest.raises(AssertionError, match="data_var 'time' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_object_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["obj"] = xr.DataArray(np.array(["a", "b"], dtype=object), dims=("x",))
    true["obj"] = xr.DataArray(np.array(["a", "c"], dtype=object), dims=("x",))
    with pytest.raises(AssertionError, match="data_var 'obj' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_index_type_mismatch():
    test = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.RangeIndex(2)},
    )
    true = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.Index([0, 1])},
    )
    with pytest.raises(AssertionError, match="coord 'x' index type mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_index_value_mismatch():
    test = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.Index([0, 1])},
    )
    true = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.Index([0, 2])},
    )
    with pytest.raises(AssertionError, match="coord 'x' values mismatch"):
        assert_xarray_datasets_equal(test, true)


def test_assert_xarray_datasets_equal_index_numeric_tolerance():
    test = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.Index([1.0, 2.0])},
    )
    true = xr.Dataset(
        data_vars={"var": (("x",), np.array([1.0, 2.0]))},
        coords={"x": pd.Index([1.0 + 1e-7, 2.0 - 1e-7])},
    )
    assert_xarray_datasets_equal(test, true, rtol=1e-6, atol=0.0)
    with pytest.raises(AssertionError, match="coord 'x' values mismatch"):
        assert_xarray_datasets_equal(test, true, rtol=1e-9, atol=0.0)


def test_assert_xarray_datasets_equal_encoding_mismatch():
    test = _make_dataset()
    true = _make_dataset()
    test["var"].encoding["dtype"] = np.float32
    true["var"].encoding["dtype"] = np.float64
    with pytest.raises(AssertionError, match="data_var 'var' encoding\\['dtype'\\]"):
        assert_xarray_datasets_equal(test, true, check_encoding=True)
