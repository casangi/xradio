from typing import Dict, TypedDict, Union

import xarray as xr


PartitionIds = TypedDict(
    "PartitionIds",
    {
        "array_id": int,
        "observation_id": int,
        "pol_setup_id": int,
        "processor_id": int,
        "spw_id": int,
    },
)

VisGroup = TypedDict(
    "VisGroup",
    {
        "seq_id": int,
        "vis": str,
        "flag": str,
        "weight": str,
        "uvw": str,
        "imaging_weight": Union[str, None],
        "descr": str,
    },
)


def make_vis_group_attr(xds: xr.Dataset) -> Dict:
    """
    Add an attribute with the initial data/vis groups that have been
    read from the MS (DATA / CORRECTED_DATA / MODEL_DATA)

    Parameters
    ----------
    xds : xr.Dataset
        dataset to make the vis_group depending on its data_vars

    Returns
    -------
    Dict
        vis_group derived form this dataset
    """
    msv2_extended_vis_vars = ["vis", "vis_corrected", "vis_model"]
    msv2_col_names = ["DATA", "CORRECTED_DATA", "MODEL_DATA"]
    # example test MS with imaging_weight col: vla/ic2233_1.ms
    imgw_name = "imaging_weight"
    img_weight = imgw_name if imgw_name in xds.data_vars else None
    vis_groups = {}
    seq_id = 1
    for var, msv2_name in zip(msv2_extended_vis_vars, msv2_col_names):
        if var in xds.data_vars:
            grp: VisGroup = {
                "seq_id": seq_id,
                "vis": var,
                "flag": "flag",
                "weight": "weight",
                "uvw": "uvw",
                "imaging_weight": img_weight,
                "descr": "xradio.vis.ms.read_ms from MSv2",
            }
            seq_id += 1
            vis_groups[f"MeasurementSet/{msv2_name}"] = grp

    return vis_groups


def init_partition_ids(
    xds: xr.Dataset,
    ddi: int,
    ddi_xds: xr.Dataset,
    part_ids: PartitionIds,
) -> PartitionIds:
    spw_id = ddi_xds.SPECTRAL_WINDOW_ID.values[ddi]
    pol_setup_id = ddi_xds.POLARIZATION_ID.values[ddi]
    ids: PartitionIds = {
        # The -1 are expected to be be updated from part_ids
        "array_id": -1,
        "observation_id": -1,
        "pol_setup_id": pol_setup_id,
        "processor_id": -1,
        "spw_id": spw_id,
    }
    ids.update(part_ids)

    return ids


def add_partition_attrs(
    xds: xr.Dataset,
    ddi: int,
    ddi_xds: xr.Dataset,
    part_ids: PartitionIds,
    other_attrs: Dict,
) -> xr.Dataset:
    """
    add attributes to the xr.Dataset:
    - sub-dict of partition-id related ones
    - sub-dict of data/vis groups
    - sub-dict of attributes coming from the lower level read
      functions (MSv2 stuff, etc.)

    Produces the partition IDs that can be retrieved from the DD subtable and also
    adds the ones passed in part_ids

    Parameters
    ----------
    xds : xr.Dataset
        dataset partition
    ddi : int
        DDI of this partition
    ddi_xds : xr.Dataset
        dataset for the DATA_DESCRIPTION subtable
    part_ids : PartitionIds
        partition id attrs
    other_attrs : Dict
        additional attributes produced by the read functions

    Returns
    -------
    xr.Dataset
        dataset with attributes added
    """

    xds = xds.assign_attrs(
        {"partition_ids": init_partition_ids(xds, ddi, ddi_xds, part_ids)}
    )
    xds = xds.assign_attrs({"vis_groups": make_vis_group_attr(xds)})
    xds = xds.assign_attrs(other_attrs)
    return xds
