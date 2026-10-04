import shutil

import pytest
import xarray as xr

mem_estimate_min_ms = 0.011131428182125092


@pytest.mark.parametrize(
    "input_name, partition_scheme, expected_estimation",
    [
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            [],
            (mem_estimate_min_ms, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID"],
            (mem_estimate_min_ms, 4, 1),
        ),
        (
            # "test_ms_minimal_required.ms",
            "ms_minimal_required",
            ["FIELD_ID", "SCAN_NUMBER"],
            (mem_estimate_min_ms, 4, 1),
        ),
    ],
)
def test_estimate_conversion_memory_and_cores_minimal(
    input_name, partition_scheme, expected_estimation, request
):
    from xradio.measurement_set import estimate_conversion_memory_and_cores

    fixture = request.getfixturevalue(input_name)
    input_path = fixture.fname if isinstance(fixture, tuple) else fixture

    res = estimate_conversion_memory_and_cores(
        input_path, partition_scheme=partition_scheme
    )
    assert res[0] == pytest.approx(expected_estimation[0], rel=1e-6)
    assert res[1:] == expected_estimation[1:]


@pytest.mark.parametrize(
    "input_path, expected_error",
    [
        (
            "inexistent_foo_path_.mms",
            pytest.raises(RuntimeError, match="does not exist"),
        ),
    ],
)
def test_estimate_conversion_memory_and_cores_with_errors(input_path, expected_error):
    from xradio.measurement_set import estimate_conversion_memory_and_cores

    # if not input_path:
    #    input_path = "ms_minimal_required.ms"
    with expected_error:
        res = estimate_conversion_memory_and_cores(input_path, [])
        assert res


def test_convert_msv2_to_processing_set_with_other_opts(ms_minimal_misbehaved):
    """Uses a few options that are not exercised in other tests"""
    from xradio.measurement_set import (
        convert_msv2_to_processing_set,
        open_processing_set,
    )
    from xradio.schema.check import check_datatree

    out_path = "test_convert_msv2_to_proc_set_without_ps_zarr_ending"
    out_path_with_ending = out_path + ".ps.zarr"
    try:
        convert_msv2_to_processing_set(
            ms_minimal_misbehaved.fname,
            out_file=out_path,
            partition_scheme=["FIELD_ID"],
            persistence_mode="w",
            parallel_mode="bogus_mode",
        )
        ps_xdt = xr.open_datatree(out_path_with_ending, engine="zarr")
        check_datatree(ps_xdt)

        # TODO: break this out to a proper test_open_processing_set:
        open_xdt = open_processing_set(out_path_with_ending, scan_intents="faulty")
        check_datatree(open_xdt)

    finally:
        shutil.rmtree(out_path_with_ending)


@pytest.mark.parametrize("parallel_mode", ["none", "partition"])
def test_convert_msv2_to_processing_set_subtable_cache_identical(
    ms_minimal_required, parallel_mode, tmp_path, monkeypatch
):
    """The processing set is the same with and without the sub-table cache (shared
    by the partitions, also by the dask threads of parallel_mode="partition")."""
    import importlib

    import dask
    import numpy as np

    from xradio.measurement_set import convert_msv2_to_processing_set
    from xradio.measurement_set._utils._msv2 import conversion
    from xradio.measurement_set._utils._msv2._tables import subtable_cache

    built = []
    cache_class = subtable_cache.SubtableCache

    def spy_cache(**kwargs):
        built.append(cache_class(**kwargs))
        return built[-1]

    # (the package re-exports the function under the module's name)
    converter_module = importlib.import_module(
        "xradio.measurement_set.convert_msv2_to_processing_set"
    )
    monkeypatch.setattr(converter_module, "SubtableCache", spy_cache)
    resolve = conversion.resolve_subtable_cache
    trees = {}
    for mode in ("0", "1"):
        # "0": no cache, every sub-table read per partition
        monkeypatch.setattr(
            conversion,
            "resolve_subtable_cache",
            resolve if mode == "1" else (lambda cache: None),
        )
        built.clear()
        out_file = str(tmp_path / f"cache{mode}.ps.zarr")
        with dask.config.set(num_workers=2):
            convert_msv2_to_processing_set(
                ms_minimal_required.fname,
                out_file=out_file,
                partition_scheme=["FIELD_ID"],
                # the other SPWs have no PHASE_CAL rows (conversion fails)
                partition_filter=lambda p: p["DATA_DESC_ID"][0] in (0, 1),
                parallel_mode=parallel_mode,
                pointing_interpolate=True,
                persistence_mode="w",
            )
        trees[mode] = xr.open_datatree(out_file, engine="zarr")
    assert len(built) == 1  # one cache for the conversion
    assert built[0].stats["pointing_cached"] == len(trees["1"].children) > 1
    paths = {node.path for node in trees["0"].subtree}
    assert paths == {node.path for node in trees["1"].subtree}
    for path in paths:
        ds_0 = trees["0"][path].to_dataset(inherit=False)
        ds_1 = trees["1"][path].to_dataset(inherit=False)
        assert list(ds_0.variables) == list(ds_1.variables), path
        for name, var in ds_0.variables.items():
            assert var.dims == ds_1[name].dims, (path, name)
            assert var.dtype == ds_1[name].dtype, (path, name)
            np.testing.assert_array_equal(var.values, ds_1[name].values)


def _assert_trees_identical(tree_a, tree_b):
    import numpy as np

    paths = {node.path for node in tree_a.subtree}
    assert paths == {node.path for node in tree_b.subtree}
    for path in paths:
        ds_a = tree_a[path].to_dataset(inherit=False)
        ds_b = tree_b[path].to_dataset(inherit=False)
        assert list(ds_a.variables) == list(ds_b.variables), path
        for name, var in ds_a.variables.items():
            assert var.dims == ds_b[name].dims, (path, name)
            assert var.dtype == ds_b[name].dtype, (path, name)
            np.testing.assert_array_equal(var.values, ds_b[name].values)


def test_convert_msv2_to_processing_set_partition_filter_dicts(
    ms_minimal_required, tmp_path
):
    """
    partition_filter gets the plain partition descriptions (lists, JSON
    serializable, as from create_partitions). A filter that changes a
    description in place gets the rows of the changed description (the MAIN
    row runs computed for the original description are not used): the same
    MSv4 as a conversion of the changed description.
    """
    import json
    import os

    from xradio.measurement_set import convert_msv2_to_processing_set
    from xradio.measurement_set._utils._msv2 import conversion

    changed = []

    def partition_filter(partition):
        json.dumps(partition)
        if partition["DATA_DESC_ID"] == [0]:
            partition["DATA_DESC_ID"] = [1]  # now selects the rows of DDI 1
            changed.append(dict(partition))
            return True
        return False

    out_file = str(tmp_path / "filtered.ps.zarr")
    convert_msv2_to_processing_set(
        ms_minimal_required.fname,
        out_file=out_file,
        partition_scheme=[],
        partition_filter=partition_filter,
        persistence_mode="w",
    )
    filtered = xr.open_datatree(out_file, engine="zarr")
    assert len(filtered.children) == 1 and len(changed) == 1

    direct_file = str(tmp_path / "direct.ps.zarr")
    msv4_name = list(filtered.children)[0]
    conversion.convert_and_write_partition(
        ms_minimal_required.fname,
        direct_file,
        msv4_name.rsplit("_", 1)[1],
        partition_info=changed[0],
        use_table_iter=False,
        persistence_mode="w",
    )
    direct = xr.open_datatree(os.path.join(direct_file, msv4_name), engine="zarr")
    _assert_trees_identical(
        xr.open_datatree(os.path.join(out_file, msv4_name), engine="zarr"), direct
    )


def test_convert_msv2_to_processing_set_subtable_cache_lifetime(
    ms_minimal_required, tmp_path, monkeypatch
):
    """
    The cache is sized by the selected partitions (one partition: no
    whole-table POINTING read for it) and is cleared even when the conversion
    fails (a kept traceback must not keep its data alive).
    """
    import importlib

    from xradio.measurement_set import convert_msv2_to_processing_set
    from xradio.measurement_set._utils._msv2._tables import subtable_cache

    built = []
    cache_class = subtable_cache.SubtableCache

    def spy_cache(**kwargs):
        built.append((kwargs, cache_class(**kwargs)))
        return built[-1][1]

    converter_module = importlib.import_module(
        "xradio.measurement_set.convert_msv2_to_processing_set"
    )
    monkeypatch.setattr(converter_module, "SubtableCache", spy_cache)

    convert_msv2_to_processing_set(
        ms_minimal_required.fname,
        out_file=str(tmp_path / "one.ps.zarr"),
        partition_scheme=[],
        partition_filter=lambda p: p["DATA_DESC_ID"] == [0],
        persistence_mode="w",
    )
    kwargs, cache = built[-1]
    assert kwargs == {"n_partitions": 1}
    assert cache.stats["pointing_uncached"] == 1
    assert cache.stats["value_builds"] == 0

    # DDIs 2 and 3 have no PHASE_CAL rows: their conversion raises
    with pytest.raises(AttributeError):
        convert_msv2_to_processing_set(
            ms_minimal_required.fname,
            out_file=str(tmp_path / "all.ps.zarr"),
            partition_scheme=[],
            persistence_mode="w",
        )
    kwargs, cache = built[-1]
    assert kwargs == {"n_partitions": 4}
    assert cache.stats["value_builds"] >= 1  # POINTING built for the partitions
    assert not cache._state.values and not cache._state.memo  # cleared


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
