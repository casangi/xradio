"""Assertion helpers for xarray objects in tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import xarray as xr


def assert_xarray_datasets_equal(
    test: xr.Dataset,
    true: xr.Dataset,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    check_attrs: bool = True,
    check_encoding: bool = False,
) -> None:
    """Assert two xarray Datasets match in structure, data, and metadata."""
    if not isinstance(test, xr.Dataset):
        raise AssertionError(f"Expected test to be xarray.Dataset, got {type(test)}")
    if not isinstance(true, xr.Dataset):
        raise AssertionError(f"Expected true to be xarray.Dataset, got {type(true)}")

    _check_dims(test, true)
    _check_coords(test, true, rtol=rtol, atol=atol)
    _check_indexes(test, true, rtol=rtol, atol=atol)
    _check_data_vars(test, true, rtol=rtol, atol=atol)
    if check_attrs:
        _check_attrs(test, true, rtol=rtol, atol=atol)
    if check_encoding:
        _check_encoding(test, true, rtol=rtol, atol=atol)


def _check_dims(test: xr.Dataset, true: xr.Dataset) -> None:
    test_dims = dict(test.sizes)
    true_dims = dict(true.sizes)

    missing = sorted(set(true_dims) - set(test_dims))
    extra = sorted(set(test_dims) - set(true_dims))
    size_mismatch = [
        (name, test_dims[name], true_dims[name])
        for name in sorted(set(test_dims) & set(true_dims))
        if test_dims[name] != true_dims[name]
    ]

    if missing or extra or size_mismatch:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if size_mismatch:
            details.append(
                "size_mismatch="
                + ", ".join(f"{n}: test={t} true={v}" for n, t, v in size_mismatch)
            )
        raise AssertionError(f"Dimension check failed: {'; '.join(details)}")


def _check_coords(
    test: xr.Dataset, true: xr.Dataset, *, rtol: float, atol: float
) -> None:
    test_coords = set(test.coords)
    true_coords = set(true.coords)

    missing = sorted(true_coords - test_coords)
    extra = sorted(test_coords - true_coords)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AssertionError(f"Coordinate keys mismatch: {'; '.join(details)}")

    for name in sorted(test_coords):
        _check_dataarray(
            test.coords[name],
            true.coords[name],
            context=f"coord '{name}'",
            rtol=rtol,
            atol=atol,
        )


def _check_data_vars(
    test: xr.Dataset, true: xr.Dataset, *, rtol: float, atol: float
) -> None:
    test_vars = set(test.data_vars)
    true_vars = set(true.data_vars)

    missing = sorted(true_vars - test_vars)
    extra = sorted(test_vars - true_vars)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AssertionError(f"Data variable keys mismatch: {'; '.join(details)}")

    for name in sorted(test_vars):
        _check_dataarray(
            test.data_vars[name],
            true.data_vars[name],
            context=f"data_var '{name}'",
            rtol=rtol,
            atol=atol,
        )


def _check_indexes(
    test: xr.Dataset, true: xr.Dataset, *, rtol: float, atol: float
) -> None:
    test_indexes = set(test.indexes)
    true_indexes = set(true.indexes)

    missing = sorted(true_indexes - test_indexes)
    extra = sorted(test_indexes - true_indexes)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AssertionError(f"Index keys mismatch: {'; '.join(details)}")

    for name in sorted(test_indexes):
        test_index = test.indexes[name]
        true_index = true.indexes[name]
        if type(test_index) is not type(true_index):
            raise AssertionError(
                "coord '{}' index type mismatch: test={} true={}".format(
                    name, type(test_index).__name__, type(true_index).__name__
                )
            )
        _compare_arrays(
            np.asarray(test_index.values),
            np.asarray(true_index.values),
            context=f"coord '{name}' index values",
            rtol=rtol,
            atol=atol,
        )


def _check_attrs(
    test: xr.Dataset, true: xr.Dataset, *, rtol: float, atol: float
) -> None:
    _compare_attrs_dict(
        test.attrs, true.attrs, context="dataset attrs", rtol=rtol, atol=atol
    )

    for name in sorted(set(test.coords) & set(true.coords)):
        _compare_attrs_dict(
            test.coords[name].attrs,
            true.coords[name].attrs,
            context=f"coord '{name}' attrs",
            rtol=rtol,
            atol=atol,
        )

    for name in sorted(set(test.data_vars) & set(true.data_vars)):
        _compare_attrs_dict(
            test.data_vars[name].attrs,
            true.data_vars[name].attrs,
            context=f"data_var '{name}' attrs",
            rtol=rtol,
            atol=atol,
        )


def _check_dataarray(
    test: xr.DataArray,
    true: xr.DataArray,
    *,
    context: str,
    rtol: float,
    atol: float,
) -> None:
    if test.dims != true.dims:
        raise AssertionError(
            f"{context} dims mismatch: test={test.dims} true={true.dims}"
        )
    if test.shape != true.shape:
        raise AssertionError(
            f"{context} shape mismatch: test={test.shape} true={true.shape}"
        )
    if test.dtype != true.dtype:
        raise AssertionError(
            f"{context} dtype mismatch: test={test.dtype} true={true.dtype}"
        )

    test_values = np.asarray(test.values)
    true_values = np.asarray(true.values)
    _compare_arrays(
        test_values, true_values, context=f"{context} values", rtol=rtol, atol=atol
    )


def _compare_arrays(
    test: np.ndarray,
    true: np.ndarray,
    *,
    context: str,
    rtol: float,
    atol: float,
) -> None:
    try:
        if _is_numeric_dtype(test.dtype):
            np.testing.assert_allclose(test, true, rtol=rtol, atol=atol, equal_nan=True)
        else:
            np.testing.assert_array_equal(test, true)
    except AssertionError as exc:
        raise AssertionError(f"{context} mismatch: {exc}") from exc


def _compare_attrs_dict(
    test_attrs: Mapping[str, Any],
    true_attrs: Mapping[str, Any],
    *,
    context: str,
    rtol: float,
    atol: float,
) -> None:
    test_keys = set(test_attrs)
    true_keys = set(true_attrs)
    missing = sorted(true_keys - test_keys)
    extra = sorted(test_keys - true_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AssertionError(f"{context} keys mismatch: {'; '.join(details)}")

    # for key in sorted(test_keys & true_keys):
    for key in sorted(test_keys):
        _compare_attr_value(
            test_attrs[key],
            true_attrs[key],
            context=_context_key(context, key),
            rtol=rtol,
            atol=atol,
        )


def _check_encoding(
    test: xr.Dataset, true: xr.Dataset, *, rtol: float, atol: float
) -> None:
    _compare_attrs_dict(
        test.encoding,
        true.encoding,
        context="dataset encoding",
        rtol=rtol,
        atol=atol,
    )

    for name in sorted(set(test.coords) & set(true.coords)):
        _compare_attrs_dict(
            test.coords[name].encoding,
            true.coords[name].encoding,
            context=f"coord '{name}' encoding",
            rtol=rtol,
            atol=atol,
        )

    for name in sorted(set(test.data_vars) & set(true.data_vars)):
        _compare_attrs_dict(
            test.data_vars[name].encoding,
            true.data_vars[name].encoding,
            context=f"data_var '{name}' encoding",
            rtol=rtol,
            atol=atol,
        )


def _compare_attr_value(
    test: Any,
    true: Any,
    *,
    context: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(test, Mapping) and isinstance(true, Mapping):
        _compare_attrs_dict(test, true, context=context, rtol=rtol, atol=atol)
        return

    if isinstance(test, np.ndarray) or isinstance(true, np.ndarray):
        _compare_arrays(
            np.asarray(test), np.asarray(true), context=context, rtol=rtol, atol=atol
        )
        return

    if _is_sequence(test) and _is_sequence(true):
        if len(test) != len(true):
            raise AssertionError(
                f"{context} length mismatch: test={len(test)} true={len(true)}"
            )
        for idx, (t_item, v_item) in enumerate(zip(test, true, strict=True)):
            _compare_attr_value(
                t_item,
                v_item,
                context=f"{context}[{idx}]",
                rtol=rtol,
                atol=atol,
            )
        return

    if _is_numeric_scalar(test) and _is_numeric_scalar(true):
        if not math.isclose(test, true, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{context} mismatch: test={test} true={true}")
        return

    if test != true:
        raise AssertionError(f"{context} mismatch: test={test} true={true}")


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, complex, np.number)) and not isinstance(
        value, bool
    )


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


def _context_key(context: str, key: str) -> str:
    return f"{context}[{key!r}]"
