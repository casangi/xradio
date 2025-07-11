from pathlib import Path

import xarray as xr

import toolviper.utils.logger as logger
import pyasdm

from xradio.measurement_set._utils._asdm.create_partitions import create_partitions
from xradio.measurement_set._utils._asdm.open_partition import open_partition


def open_asdm(asdm_path: str, partition_scheme: list = ["fieldId"]):
    """
    TODO

    Parameters
    ----------
    asdm_path:
        Input ASDM path
    partition_scheme:
        partition axes: ["execBlockId", "dataDescriptionId", "scanIntent"]
        + optional ones

    Returns
    -------
    xr.DataTree
        Datatree with processing set of MSv4s populated from the input ASDM
    """

    ps_xdt = xr.DataTree()
    ps_xdt.attrs["type"] = "processing_set"

    asdm = pyasdm.ASDM()
    asdm.setFromFile(asdm_path)

    partitions = create_partitions(asdm, partition_scheme)

    for msv4_idx, partition_descr in enumerate(partitions):
        logger.info(
            "execBlock "
            + str(partition_descr["execBlockId"])
            + ", dataDesciption: "
            + str(partition_descr["dataDescriptionId"])
            + ", scanIntent: "
            + str(partition_descr["scanIntent"])
            + ", field: "
            + str(partition_descr["fieldId"])
            + ", scan: "
            + str(partition_descr["scanNumber"])
            + ", subscan: "
            + str(partition_descr["subscanNumber"])
            + (
                ", antenna: " + str(partition_descr["antennaId"])
                if "antennaId" in partition_descr
                else ""
            )
        )

        msv4_xdt = open_partition(asdm, partition_descr)

        msv4_idx = f"{msv4_idx:0>{len(str(len(partitions) - 1))}}"
        msv4_name = f"{Path(asdm_path).name}_{msv4_idx}"
        ps_xdt[msv4_name] = msv4_xdt

    return ps_xdt
