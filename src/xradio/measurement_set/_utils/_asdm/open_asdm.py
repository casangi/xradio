import os
import traceback
from pathlib import Path

import pyasdm
import xarray as xr

from xradio._utils.logging import xradio_logger
from xradio.measurement_set._utils._asdm.create_partitions import create_partitions
from xradio.measurement_set._utils._asdm.open_partition import open_partition


def open_asdm(
    asdm_path: str | os.PathLike,
    partition_scheme: list[str] | None = None,
    include_processor_types: list[str] | None = None,
    include_spectral_resolution_types: list[str] | None = None,
    with_pointing: bool = False,
    pointing_for_only_spectral_resolution_types: list[str] | None = None,
) -> xr.DataTree:
    """
    Opens an ASDM (ALMA Science Data Model) dataset and presents it as an Xarray
    DataTree (processing set).
    The ASDM is partitioned according to the specified scheme and each
    partition is opened lazily as a separate MSv4 (Measurement Set version 4).

    This function is also available as ``xradio.measurement_set.open_asdm`` and
    is used by the ``xradio_asdm`` Xarray backend engine
    (``xr.open_datatree(asdm_path, engine="xradio_asdm", ...)``). It requires the
    optional dependency pyasdm (``pip install 'xradio[asdm]'``).

    Parameters
    ----------
    asdm_path : str | os.PathLike
        Input ASDM path (directory). It is made absolute (and "~" expanded), so
        that the lazily loaded data can be read later regardless of the current
        working directory.
    partition_scheme : list[str] | None, optional
        Optional axes to partition the data on, in addition to the mandatory
        axes, which are always used: ["execBlockId", "configDescriptionId",
        "dataDescriptionId", "scanIntent"].
        The optional axes are: "fieldId", "scanNumber", "subscanNumber".
        None (default) means ["fieldId"]. An empty list means no optional axes
        (partition only on the mandatory axes). Other names raise a ValueError.
    include_processor_types : list[str] | None, optional
        when opening the ASDM, produce MSv4s only for partitions with these processor types.
        Possible values are (from the ASDM ProcessorType enumeration):
        "CORRELATOR", "SPECTROMETER", "RADIOMETER".
        Default is (subject to change) ["CORRELATOR", "SPECTROMETER"], which implies
        radiometer data (WVR and the like) are excluded.
    include_spectral_resolution_types : list[str] | None, optional
        when opening the ASDM, produce MSv4s only for partitions with these spectral resolution
        types. Possible values are (from the ASDM SpectralResolutionType enumeration):
        "CHANNEL_AVERAGE", "BASEBAND_WIDE", "FULL_RESOLUTION".
        Default is (subject to change) ["FULL_RESOLUTION", "BASEBAND_WIDE"], which implies
        channel average data are excluded.
    with_pointing : bool, optional
        whether to read the Pointing table from the ASDM into pointing_xds sub-datasets included
        in the resulting DataTree. Default is False. A pointing_xds has the antennas of its MSv4
        (NaN for antennas without Pointing samples in its time range). When the Pointing table has
        no samples for an MSv4, or cannot be converted (inconsistent rows, or polynomial
        pointing, which is logged as an error), that MSv4 has no pointing_xds.
    pointing_for_only_spectral_resolution_types : list[str] | None, optional
        When with_pointing is enabled, this parameter can be used to give a list of the spectral
        resolution types for which the pointing dataset should be created. The MSv4s created
        for partitions with spectral resolution types not included in the list will not have
        a pointing dataset. When the list is not given or is empty, all spectral resolution
        types will have a pointing dataset.

    Returns
    -------
    xr.DataTree
        Datatree with processing set of MSv4s populated from the input ASDM.
        Each node of the tree represents a partition of the original ASDM data.
        The DataTree has a 'type' attribute set to 'processing_set'. Node names are
        formatted as '{asdm_name}_{index}', where index is the index of the
        partition, zero-padded to the number of digits of the largest partition
        index (for example '_00' to '_11' for 12 partitions). Partitions that
        cannot be opened are skipped (with an error logged), so the indices may
        have gaps.

    Raises
    ------
    RuntimeError
        If none of the partitions of the ASDM can be opened.
    """

    asdm_path = os.path.abspath(os.path.expanduser(os.fspath(asdm_path)))

    if not include_processor_types:
        include_processor_types = ["CORRELATOR", "SPECTROMETER"]

    if not include_spectral_resolution_types:
        include_spectral_resolution_types = ["FULL_RESOLUTION", "BASEBAND_WIDE"]

    asdm = pyasdm.ASDM()
    asdm.setFromFile(asdm_path)

    partitions = create_partitions(
        asdm,
        partition_scheme,
        include_processor_types,
        include_spectral_resolution_types,
    )

    if len(partitions) == 0:
        raise RuntimeError(f"No partitions to open in the ASDM {asdm_path}")

    ps_xdt = xr.DataTree()
    ps_xdt.attrs["type"] = "processing_set"

    asdm_name = Path(asdm_path).name
    idx_width = len(str(len(partitions) - 1))
    failed_partitions = []
    for msv4_idx, partition_descr in enumerate(partitions):
        xradio_logger().info(
            "Opening partition for: execBlock "
            + str(partition_descr["execBlockId"])
            + ", dataDescription: "
            + str(partition_descr["dataDescriptionId"])
            + ", scanIntent: "
            + str(partition_descr["scanIntent"])
            + ", field: "
            + str(partition_descr["fieldId"])
            + ", scan: "
            + str(partition_descr["scanNumber"])
            + ", subscan: "
            + str(partition_descr["subscanNumber"])
            + ", state: "
            + str(partition_descr["stateId"])
        )

        try:
            msv4_xdt = open_partition(
                asdm,
                partition_descr,
                with_pointing,
                pointing_for_only_spectral_resolution_types,
            )
        except Exception as exc:
            trace = traceback.format_exc()
            xradio_logger().error(
                f"Skipping partition {msv4_idx} of {len(partitions)}, which could not "
                f"be opened, with {partition_descr=}.\n"
                f"Error: {exc!r}\n"
                f"\nTraceback: {trace}\n"
            )
            failed_partitions.append(msv4_idx)
            continue

        msv4_name = f"{asdm_name}_{msv4_idx:0>{idx_width}}"
        ps_xdt[msv4_name] = msv4_xdt

    if failed_partitions:
        if len(failed_partitions) == len(partitions):
            raise RuntimeError(
                f"None of the {len(partitions)} partitions of the ASDM {asdm_path} "
                "could be opened (see the errors logged for every partition)."
            )
        xradio_logger().warning(
            f"{len(failed_partitions)} of {len(partitions)} partitions of the ASDM "
            f"{asdm_path} could not be opened and were skipped (partition indices: "
            f"{failed_partitions})."
        )

    return ps_xdt
