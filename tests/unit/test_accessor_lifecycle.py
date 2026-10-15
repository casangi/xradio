"""Memory-safety contract for XRADIO's registered xarray accessors.

Enforces the accessor rules in AGENT.md ("Writing xarray Accessors"): touching
a registered accessor must NOT create a reference cycle that keeps the
Dataset/DataTree alive past its last reference. The 2026-08 Frontera diagnosis
traced multi-GB-per-task memory pinning to exactly such cycles; these tests
fail if anyone reverts an accessor to the (cycle-forming) pattern from the
xarray documentation (a cached accessor holding its object strongly).

Two designs are in use:

* ``.xr_img`` and ``.xr_ms`` are NOT cached on the object: every access builds
  a new accessor that holds its object strongly, so chained calls on
  temporaries work and an accessor kept by the caller keeps its object alive.
* ``.xr_ps`` is cached by xarray (it caches its summary) and holds its tree
  WEAKLY, so an accessor kept beyond its tree raises ReferenceError.

Every accessor added to XRADIO must gain a case here.
"""

import gc
import weakref

import numpy as np
import pytest
import xarray as xr

import xradio.image  # noqa: F401  (registers .xr_img)
import xradio.measurement_set  # noqa: F401  (registers .xr_ps / .xr_ms)


def _image_dataset():
    ds = xr.Dataset(
        {"SKY": (("l", "m"), np.zeros((2, 2)))},
        coords={"l": [-1.0, 0.0], "m": [0.0, 1.0]},
    )
    ds.attrs["type"] = "image_dataset"
    ds.attrs["data_groups"] = {"base": {}}
    return ds


def _processing_set_tree():
    xdt = xr.DataTree(name="ps")
    xdt.attrs["type"] = "processing_set"
    return xdt


def _measurement_set_tree():
    xdt = xr.DataTree(name="ms")
    xdt.attrs["type"] = "visibility"
    return xdt


def _measurement_set_tree_with_child():
    """An MSv4-like tree whose root and child form a reference cycle, so a
    temporary of it dies only at the next garbage collection."""
    xdt = xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"VISIBILITY": ("time", np.zeros(3))}),
            "/antenna_xds": xr.Dataset({"ANTENNA_POSITION": ("antenna", np.zeros(2))}),
        },
        name="ms",
    )
    xdt.attrs["type"] = "visibility"
    xdt.attrs["data_groups"] = {"base": {"correlated_data": "VISIBILITY"}}
    return xdt


_ALL_ACCESSORS = [
    (_image_dataset, "xr_img"),
    (_processing_set_tree, "xr_ps"),
    (_measurement_set_tree, "xr_ms"),
]

_UNCACHED_ACCESSORS = [
    (_image_dataset, "xr_img", "_xds"),
    (_measurement_set_tree, "xr_ms", "_xdt"),
]


@pytest.mark.parametrize(("make", "accessor_name"), _ALL_ACCESSORS)
def test_accessor_leaves_no_cycle(make, accessor_name):
    """After touching the accessor, the object must die by REFCOUNT alone
    (no gc.collect()) when the last reference is dropped."""
    obj = make()
    getattr(obj, accessor_name)  # build (and, for a cached accessor, cache) it
    ref = weakref.ref(obj)
    del obj
    assert ref() is None, (
        f".{accessor_name} created a reference cycle: the object survived "
        "its last reference and now needs a garbage-collection pass to die. "
        "See AGENT.md 'Writing xarray Accessors'."
    )


@pytest.mark.parametrize(("make", "accessor_name", "attribute"), _UNCACHED_ACCESSORS)
def test_uncached_accessor_is_not_stored_on_the_object(make, accessor_name, attribute):
    obj = make()
    first = getattr(obj, accessor_name)
    second = getattr(obj, accessor_name)
    assert first is not second
    assert getattr(first, attribute) is obj
    assert accessor_name not in (getattr(obj, "_cache", None) or {})


@pytest.mark.parametrize(("make", "accessor_name", "attribute"), _UNCACHED_ACCESSORS)
def test_kept_uncached_accessor_keeps_its_object(make, accessor_name, attribute):
    """An accessor kept by the caller holds its object strongly; both die by
    reference counting once the accessor is released."""
    accessor = getattr(make(), accessor_name)
    gc.collect()
    obj = getattr(accessor, attribute)
    assert obj.attrs["type"] in ("image_dataset", "visibility")
    ref = weakref.ref(obj)
    del obj, accessor
    assert ref() is None


def test_chained_image_accessor_on_temporaries():
    """Chained calls on temporaries work (they raised ReferenceError when the
    accessor held its dataset weakly)."""
    assert _image_dataset().isel(l=[0, 1]).xr_img.test_func() == "Hallo"
    cell = _image_dataset().xr_img.sel(m=[0.0, 1.0]).xr_img.get_lm_cell_size()
    np.testing.assert_allclose(cell, [1.0, 1.0])


def test_chained_ms_accessor_on_a_cyclic_temporary():
    """A DataTree temporary is cyclic garbage as soon as it is created; a
    garbage collection between building the accessor and calling it must
    not break the call."""
    accessor = _measurement_set_tree_with_child().xr_ms
    gc.collect()
    xdt = accessor.delete_data_variables(["VISIBILITY"])
    assert "VISIBILITY" not in xdt.data_vars
    assert xdt.attrs["data_groups"] == {"base": {}}


def test_accessor_class_on_the_xarray_class():
    """Class attribute access gives the accessor class (documentation tools)."""
    from xradio.image.image_xds import ImageXds
    from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt

    assert xr.Dataset.xr_img is ImageXds
    assert xr.DataTree.xr_ms is MeasurementSetXdt


def test_cached_weak_accessor_outliving_object_raises():
    """``.xr_ps`` keeps the cached weak design: an accessor instance kept
    beyond its tree's life must raise ReferenceError (not silently resurrect
    or segfault)."""
    obj = _processing_set_tree()
    accessor = obj.xr_ps
    assert obj.xr_ps is accessor  # cached by xarray
    del obj
    with pytest.raises(ReferenceError):
        _ = accessor._xdt


def test_direct_construction_keeps_wrapper_semantics():
    """Standalone construction (no accessor protocol) must hold the object
    STRONGLY: ``ProcessingSetXdt(xr.DataTree())`` used as an inline wrapper
    keeps its tree alive."""
    from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt

    wrapper = ProcessingSetXdt(_processing_set_tree())
    assert wrapper._xdt is not None  # would be dead under a pure-weakref design
