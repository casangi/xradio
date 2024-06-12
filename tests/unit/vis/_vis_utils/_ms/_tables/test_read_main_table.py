import pytest

def test_rename_vars():
    from xradio.vis._vis_utils._ms._tables.read_main_table import rename_vars

    
def test_redim_id_data_vars():
    from xradio.vis._vis_utils._ms._tables.read_main_table import redim_id_data_vars


@pytest.mark.parametrize("where, expected_pids", [
    ("where DATA_DESC_ID = 0 AND SCAN_NUMBER = 1 AND STATE_ID = 1", {'array_id': [], 'observation_id': [], 'processor_id': []}),
    ("where DATA_DESC_ID = 0 AND SCAN_NUMBER = 1 AND STATE_ID = 0", {'array_id': [0], 'observation_id': [0], 'processor_id': [0]}),
])
def test_get_parittion_ids(ms_minimal_required, where, expected_pids):
    from xradio.vis._vis_utils._ms._tables.read_main_table import get_partition_ids
    from xradio.vis._vis_utils._ms._tables.table_query import open_table_ro

    with open_table_ro(ms_minimal_required.fname) as mtable:
        pids = get_partition_ids(mtable, where)
        assert pids == expected_pids


def test_read_expanded_main_table():
    from xradio.vis._vis_utils._ms._tables.read_main_table import read_expanded_main_table

    # tested one level up for now


def test_read_main_table_chunks():
    from xradio.vis._vis_utils._ms._tables.read_main_table import read_main_table_chunks

    # tested one level up for now


# TODO: missing:

# get_utimes_tol

# get_baselines

# read_all_cols_bvars

# concat_bvars_update_tvars

# concat_tvars_to_mvars


def test_read_flat_main_table_w_scan_subscan(ms_minimal_required):
    from xradio.vis._vis_utils._ms._tables.read_main_table import read_flat_main_table
    import xarray as  xr
    
    # with pytest.raises(IndexError, match=""):
    res = read_flat_main_table(ms_minimal_required.fname, 0, (1, 0))
    assert res
    assert isinstance(res, tuple)
    assert isinstance(res[0], xr.Dataset)
    assert isinstance(res[1], dict)
    assert isinstance(res[2], dict)


def test_read_flat_main_table_no_scan(ms_empty_required):
    from xradio.vis._vis_utils._ms._tables.read_main_table import read_flat_main_table
    import xarray as xr
    
    res = read_flat_main_table(ms_empty_required.fname, 0, None)
    assert isinstance(res, tuple)
    assert isinstance(res[0], xr.Dataset)
    assert isinstance(res[1], dict)
    assert isinstance(res[2], dict)


def test_read_flat_main_table_no_subscan(ms_empty_required):
    from xradio.vis._vis_utils._ms._tables.read_main_table import read_flat_main_table
    import xarray as xr
    
    res = read_flat_main_table(ms_empty_required.fname, 1, (1, None))
    assert isinstance(res, tuple)
    assert isinstance(res[0], xr.Dataset)
    assert isinstance(res[1], dict)
    assert isinstance(res[2], dict)
