from typing import Union
from xradio.vis._vis_utils._utils.cds import CASAVisSet


def check_cds(
    vis: CASAVisSet,
    partition_scheme: str = "intent",
    descr: dict = None,
    subtables: Union[list, None] = None,
    chunks: bool = False,
):
    """
    Loose consistency check on a cngi-io/xradio cds visibilities dataset.
    This does not claim to be complete (at all) but should catch all obvious
    issue and most common issues. To be replaced/superseded by a proper
    MSv4 schema validator + additional checks.

    The check functions called from here do not return anything, they simply
    assert (which is much nicer if the asserts are rewritten using
    pytest.register_assert_rewrite).

    Parameters
    ----------
    vis : CASAVisSet :

    partition_scheme : str (Default value = "intent")
        the scheme used to load the MS. Relevant for metainfo and partition
        keys.
    descr : dict (Default value = None)
        description of the visibilities dataset (used in gen_test_ms to
        produce it or otherwise describing some of the dataset properties)
    subtables : Union[list, None] (Default value = None)
        subtables that have been loaded and are expected to pass checks.
        If left to its default (None) the check functions will check
        a baseline list of subtables (growing but not exhaustive)
    chunks : bool (Default value = False)
        whether data blocks/chunks have been loaded

    Returns
    -------

    """

    assert vis is not None
    check_vis_metainfo(vis, partition_scheme, descr, subtables)
    check_vis_partitions(vis, partition_scheme, descr, chunks)


def check_vis_metainfo(
    vis: CASAVisSet,
    partition_scheme: str,
    descr: str,
    subtables: Union[list, None] = None,
):
    """

    Parameters
    ----------
    vis : CASAVisSet

    partition_scheme : str

    descr : str

    subtables : Union[list, None] (Default value = None)

    Returns
    -------

    """
    if subtables == None:
        subtables = [
            "antenna",
            "spectral_window",
            "polarization",
            "field",
            "source",
            "state",
        ]

    assert not subtables or all(subt in vis.metainfo for subt in subtables)

    # TODO: dict subt_name => check functions
    if "antenna" in subtables:
        check_subxds_antenna(vis.metainfo["antenna"])
    if "spectral_window" in subtables:
        check_subxds_spectral_window(vis.metainfo["spectral_window"])
    # check_subxds_polarization(vis.metainfo["spectral_window"]
    if "field" in subtables:
        check_subxds_field(vis.metainfo["field"])
    # check_subxds_source(vis.metainfo["source"]
    # check_subxds_state(vis.metainfo["state"]


def check_subxds_antenna(xds):
    # TODO: use dicts from code itself
    assert "antenna_id" in xds.dims
    res = [var in xds.data_vars for var in ["position", "offset", "station"]]
    assert xds.attrs


def check_subxds_spectral_window(xds):
    # TODO: use dicts from code itself
    assert "spectral_window_id" in xds.dims
    res = [var in xds.data_vars for var in ["ref_frequency", "effective_bw"]]
    assert xds.attrs


def check_subxds_field(xds):
    # TODO: use dicts from code itself
    assert "field_id" in xds.dims
    res = [var in xds.data_vars for var in ["name", "code", "time", "source_id"]]
    assert xds.attrs


def check_vis_partitions(vis, partition_scheme, descr, chunks):
    expected_part_key = (0, 0, "scan_intent#subscan_intent")

    # assert len(vis.partitions) == 1
    # assert expected_part_key in vis.partitions

    # part = vis.partitions[expected_part_key]
    # check_partition_data(part)
    # check_partition_metainfo(part)

    # These two paths should be better merged...
    if not chunks:
        check_vis_partitions_read(vis, partition_scheme, descr, chunks)
    else:
        check_vis_partitions_load(vis, partition_scheme, descr, chunks)


def check_vis_partitions_read(vis, partition_scheme, descr, chunks):
    for key, part in vis.partitions.items():
        check_partition_data(part, descr)
        check_partition_metainfo(part, partition_scheme, descr, chunks)


def check_vis_partitions_load(vis, partition_scheme, descr, chunks):
    for key, val in vis.partitions.items():
        for ckey, part in val.items():
            check_partition_data(part, descr)
            check_partition_metainfo(part, partition_scheme, descr, chunks)


def check_partition_data(part, descr):
    expected_coords_with_dim = {"time", "baseline", "pol", "freq", "antenna_id"}
    check_partition_coords(part.coords, expected_coords_with_dim)

    expected_dims_wo_coords = {"uvw_coords"}

    # TODO: dimensions without coordiates: uvw_coords + (poly_id, ra/dec) if
    # in descr.
    check_partition_dims(part.sizes, expected_dims_wo_coords)

    # TODO: assert len(part.data_vars) >/== 16*
    check_partition_data_vars(part.data_vars)


def check_partition_coords(coords, expected_coords_with_dim):
    keys = coords.keys()
    assert keys == expected_coords_with_dim

    # assert all([coo in coords for coo in expected_coords])
    time = "time"
    assert all([att in coords[time].attrs for att in ["units", "measure"]])
    assert all(
        [matt in coords[time].attrs["measure"] for matt in ["type", "ref_frame"]]
    )
    assert coords[time].attrs["units"] == "s"
    assert coords[time].attrs["measure"]["type"] == "epoch"
    assert coords[time].attrs["measure"]["ref_frame"] == "UTC"

    freq = "freq"
    assert all([att in coords[freq].attrs for att in ["units", "measure"]])
    assert all(
        [matt in coords[freq].attrs["measure"] for matt in ["type", "ref_frame"]]
    )
    assert coords[freq].attrs["units"] == "Hz"
    assert coords[freq].attrs["measure"]["type"] == "frequency"
    # incomplete set
    assert coords[freq].attrs["measure"]["ref_frame"] in {"TOPO", "LSRK", "REST"}


def check_partition_dims(dims, expected_dims_wo_coords):
    assert all([dim in dims.keys() for dim in expected_dims_wo_coords])


def check_partition_data_vars(data_vars):
    exp_data_vars = {
        "uvw",
        "flag",
        "array_id",
        "exposure",
        "feed1_id",
        "feed2_id",
        "field_id",
        "interval",
        "observation_id",
        "processor_id",
        "scan_number",
        "state_id",
        "time_centroid",
        "vis",
        "baseline_ant1_id",
        "baseline_ant2_id",
    }  # vis_correcred
    assert exp_data_vars <= data_vars.keys()


def check_partition_metainfo(part, partition_scheme, descr, chunks):
    check_part_attrs(part.attrs, partition_scheme, chunks)
    exp_ids = {
        "array_id": [0],
        "observation_id": [0],
        "pol_setup_id": 0,
        "processor_id": [0],
        "spw_id": 0,
    }
    # For the values we'd need a properly populated 'descr'
    assert part.attrs["partition_ids"].keys() == exp_ids.keys()


def check_part_attrs(attrs, partition_scheme, chunks):
    exp_attrs = {"partition_ids", "vis_groups"}
    if not chunks:
        exp_attrs.add("other")

    # only present when intent is not considered for partitioning
    if not chunks and partition_scheme == "intent":
        exp_attrs.add("scan_subscan_intents")

    assert exp_attrs == attrs.keys()
