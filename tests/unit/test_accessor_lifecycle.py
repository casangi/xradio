"""Memory-safety contract for XRADIO's registered xarray accessors.

Enforces the accessor rules in AGENT.md ("Writing xarray Accessors"): touching
a registered accessor (which xarray caches in ``obj._cache``) must NOT create
a reference cycle that keeps the Dataset/DataTree alive past its last
reference. The 2026-08 Frontera diagnosis traced multi-GB-per-task memory
pinning to exactly such cycles; these tests fail if anyone reverts an accessor
to the (cycle-forming) pattern from the xarray documentation.

Every accessor added to XRADIO must gain a case here.
"""

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


@pytest.mark.parametrize(
    ("make", "accessor_name"),
    [
        (_image_dataset, "xr_img"),
        (_processing_set_tree, "xr_ps"),
        (_measurement_set_tree, "xr_ms"),
    ],
)
def test_accessor_leaves_no_cycle(make, accessor_name):
    """After touching the accessor, the object must die by REFCOUNT alone
    (no gc.collect()) when the last reference is dropped."""
    obj = make()
    getattr(obj, accessor_name)  # populate xarray's accessor cache
    ref = weakref.ref(obj)
    del obj
    assert ref() is None, (
        f".{accessor_name} created a reference cycle: the object survived "
        "its last reference and now needs a garbage-collection pass to die. "
        "See AGENT.md 'Writing xarray Accessors' (weakref factory pattern)."
    )


@pytest.mark.parametrize(
    ("make", "accessor_name"),
    [
        (_image_dataset, "xr_img"),
        (_processing_set_tree, "xr_ps"),
        (_measurement_set_tree, "xr_ms"),
    ],
)
def test_accessor_outliving_object_raises(make, accessor_name):
    """An accessor instance kept beyond its object's life must raise
    ReferenceError (not silently resurrect or segfault)."""
    obj = make()
    accessor = getattr(obj, accessor_name)
    del obj
    with pytest.raises(ReferenceError):
        _ = accessor._xds if accessor_name == "xr_img" else accessor._xdt


def test_direct_construction_keeps_wrapper_semantics():
    """Standalone construction (no accessor protocol) must hold the object
    STRONGLY: ``ProcessingSetXdt(xr.DataTree())`` used as an inline wrapper
    keeps its tree alive."""
    from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt

    wrapper = ProcessingSetXdt(_processing_set_tree())
    assert wrapper._xdt is not None  # would be dead under a pure-weakref design
