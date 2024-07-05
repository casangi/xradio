def test_cds():
    from xradio.vis._vis_utils._utils.xds_helper import CASAVisSet

    descr = "any"
    cds = CASAVisSet({}, {}, "any")
    assert cds["metainfo"] == {}
    assert cds["partitions"] == {}
    assert cds["descr"] == descr
    assert print(cds) == None
    expected_keywords = ["metainfo", "partitions", "descr"]
    assert all([key in cds.__repr__() for key in expected_keywords])
    assert all([key in cds._repr_html_() for key in expected_keywords])
