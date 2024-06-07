import pytest

@pytest.mark.parametrize("cols, expected_output", [
    (['observation_id', 'vis', 'flag'],
     {"OBSERVATION_ID": "observation_id", "DATA": "vis", "FLAG": "flag"}),
    (["antenna1_id", "feed1_id"],
     {"ANTENNA1": "antenna1_id", "FEED1": "feed1_id"}),
])
def test_cols_from_xds_to_ms(cols, expected_output):
    from xradio.vis._vis_utils._ms._tables.write_exp_api import cols_from_xds_to_ms

    assert cols_from_xds_to_ms(cols) == expected_output


def test_write_ms_with_cds():
    from xradio.vis._vis_utils._ms._tables.write_exp_api import write_ms
    from xradio.vis._vis_utils._utils.cds import CASAVisSet

    with pytest.raises(IndexError, match="out of range"):
        write_ms(CASAVisSet({}, {}, "empty"), "test_out_vis.ms")


def test_write_ms_empty():
    from xradio.vis._vis_utils._ms._tables.write_exp_api import write_ms
    from xradio.vis._vis_utils._utils.xds_helper import vis_xds_packager_mxds

    mxds = vis_xds_packager_mxds({}, {}, add_global_coords=True)
    with pytest.raises(IndexError, match="out of range"):
        write_ms(mxds, outfile="test_out_write_ms_empty.ms")


def test_write_ms_serial_empty():
    from xradio.vis._vis_utils._ms._tables.write_exp_api import write_ms_serial
    from xradio.vis._vis_utils._utils.xds_helper import vis_xds_packager_mxds

    mxds = vis_xds_packager_mxds({}, {}, add_global_coords=True)
    with pytest.raises(IndexError, match="out of range"):
        write_ms_serial(mxds, outfile="test_out_write_ms_empty.ms")


def test_write_ms_cds_min(cds_minimal_required):
    from xradio.vis._vis_utils._ms._tables.write_exp_api import write_ms
    from xradio.vis._vis_utils._utils.xds_helper import vis_xds_packager_mxds

    mxds_min = vis_xds_packager_mxds(
        cds_minimal_required.partitions,
        cds_minimal_required.metainfo,
        add_global_coords=False,
    )
    write_ms(mxds_min, "test_write_cds_min_blah.ms", subtables=True, modcols={"FLAG": "flag"})


def test_write_ms_serial_cds_min(cds_minimal_required):
    from xradio.vis._vis_utils._ms._tables.write_exp_api import write_ms_serial
    from xradio.vis._vis_utils._utils.xds_helper import vis_xds_packager_mxds

    mxds_min = vis_xds_packager_mxds(
        cds_minimal_required.partitions,
        cds_minimal_required.metainfo,
        add_global_coords=False,
    )
    write_ms_serial(mxds_min, "test_write_cds_min_blah.ms", subtables=True, verbose=True)
