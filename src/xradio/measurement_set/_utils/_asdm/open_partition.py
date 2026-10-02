import copy
import datetime
import importlib.metadata
import math

import numpy as np
import pandas as pd
import pyasdm
import xarray as xr

from xradio._utils.dict_helpers import (
    make_quantity,
    make_quantity_attrs,
    make_spectral_coord_measure_attrs,
    make_spectral_coord_reference_dict,
    make_time_measure_attrs,
)
from xradio._utils.logging import xradio_logger
from xradio._utils.schema import casacore_to_msv4_measure_type
from xradio.measurement_set._utils._asdm import asdm_backend_arrays
from xradio.measurement_set._utils._asdm._utils._bdf.bdf_description_checks import (
    BDFOpenError,
    open_bdf,
)
from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
    load_times_from_partition_bdfs,
)
from xradio.measurement_set._utils._asdm._utils._bdf.shapes import select_apc_index
from xradio.measurement_set._utils._asdm._utils.field_source import make_field_name
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.spectral_window import (
    ensure_spw_name_conforms,
    get_chan_width,
    get_reference_frame,
    get_spw_frequency_centers,
    get_spw_name,
)
from xradio.measurement_set._utils._asdm._utils.time import (
    ASDM_TIME_FORMAT,
    ASDM_TIME_SCALE,
)
from xradio.measurement_set._utils._asdm.create_antenna_xds import create_antenna_xds
from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set._utils._asdm.create_info_dicts import create_info_dicts
from xradio.measurement_set._utils._asdm.create_pointing_xds import (
    POINTING_CONVERSION_ERRORS,
    create_pointing_xds,
)
from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION

#: Margin (seconds) added on both sides of the time range of a partition when
#: selecting the Pointing samples for its pointing_xds.
POINTING_TIME_RANGE_MARGIN = 1.0

#: Target size (bytes) of the preferred dask chunks of the correlated dataset
#: when dask's "array.chunk-size" setting is not available (it is the dask
#: default).
DEFAULT_PREFERRED_CHUNK_BYTES = 128 * 2**20

#: ASDM SpectralWindow.measFreqRef (FrequencyReferenceCode) -> MSv4 spectral
#: coordinate ``observer``. Same translation as the MSv2 converter (MEAS_FREQ_REF),
#: plus LABREST (laboratory rest frequency) as REST. GALACTO has no MSv4
#: equivalent.
ASDM_FREQUENCY_FRAME_TO_MSV4_OBSERVER = {
    **casacore_to_msv4_measure_type["frequency"]["Ref_map"],
    "LABREST": "REST",
}


def open_partition(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
    with_pointing: bool = False,
    pointing_for_only_spectral_resolution_types: list[str] | None = None,
) -> xr.DataTree:
    """
    Opens an ASDM partition as an MSv4 DataTree.

    The main (correlated data) dataset is a
    :py:class:`~xradio.measurement_set.schema.VisibilityXds` (type "visibility",
    or "radiometer" for RADIOMETER processors) for interferometric data, and a
    :py:class:`~xradio.measurement_set.schema.SpectrumXds` (type "spectrum") for
    AUTO_ONLY (single dish) data.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        Input ASDM object
    partition_descr : dict[str, np.ndarray]
        Description of the partition, as produced by
        :func:`~xradio.measurement_set._utils._asdm.create_partitions.create_partitions`:
        "ID ASDM key/attribute" -> 1-D arrays of IDs, plus "BDFPath" (one BDF per
        Main row, in time order), "scanIntent" ("INTENT#SUBINTENT" strings) and
        "per_bdf" (scanNumber, fieldId, etc. of every BDF, aligned with
        "BDFPath").
    with_pointing : bool, optional
        Whether to read the Pointing table from the ASDM into a pointing xds
        sub-dataset (restricted to the time range of the partition, with the
        antennas of the partition). By default False. When the Pointing table
        has no samples for the partition or cannot be converted, the partition
        is opened without pointing xds.
    pointing_for_only_spectral_resolution_types : list[str] | None, optional
        When with_pointing is enabled, this parameter can be used to give a list of
        the spectral resolution types for which the pointing dataset should be
        created. The MSv4 created for this partition will not have a pointing
        dataset if its spectral resolution type is not included in the list. When
        the list is not given or is empty, the pointing dataset is created
        regardless of the spectral resolution type.

    Returns
    -------
    xr.DataTree
        Datatree with MSv4 populated from the ASDM partition
    """

    # correlated_xds with already populated coordinates, data variables and info_dicts
    correlated_xds, num_antenna, spw_id, is_single_dish = create_correlated_xds(
        asdm, partition_descr
    )

    antenna_xds = _create_partition_antenna_xds(
        asdm, partition_descr, spw_id, correlated_xds.polarization
    )

    # TODO: gain_curve_xds, phase_calibration_xds, system_calibration_xds,
    # weather_xds, phased_array_xds

    field_and_source_xds = create_field_and_source_xds(
        asdm, partition_descr, spw_id, is_single_dish
    )

    if not is_single_dish:
        phase_center_direction = _select_phase_center_direction_by_time(
            field_and_source_xds, correlated_xds.coords["field_name"]
        )
        uvw_data_var = _create_uvw_data_var(
            correlated_xds.sizes,
            correlated_xds.coords["time"],
            correlated_xds.coords["baseline_antenna1_name"],
            correlated_xds.coords["baseline_antenna2_name"],
            antenna_xds.data_vars["ANTENNA_POSITION"],
            phase_center_direction,
        )
        # same preferred time chunks as the other variables (consistent dask
        # chunks when opened with chunks={})
        correlated_xds = _set_preferred_time_chunk(
            correlated_xds.assign(uvw_data_var),
            _get_preferred_time_chunk(correlated_xds),
        )
        data_groups = copy.deepcopy(correlated_xds.attrs["data_groups"])
        data_groups["base"]["uvw"] = "UVW"
        correlated_xds.attrs["data_groups"] = data_groups

    msv4_xdt = xr.DataTree(dataset=correlated_xds)
    msv4_xdt["/antenna_xds"] = antenna_xds
    msv4_xdt["/field_and_source_base_xds"] = field_and_source_xds

    if _is_pointing_requested(
        partition_descr, with_pointing, pointing_for_only_spectral_resolution_types
    ):
        pointing_xds = _create_partition_pointing_xds(
            asdm, correlated_xds, antenna_xds.coords["antenna_name"].values
        )
        if pointing_xds is not None:
            msv4_xdt["/pointing_xds"] = pointing_xds

    xradio_logger().debug(
        f"Opened partition, type={correlated_xds.attrs['type']}, {spw_id=}, "
        f"{num_antenna=}"
    )

    return msv4_xdt


def create_correlated_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
) -> tuple[xr.Dataset, int, int, bool]:
    """
    Create the correlated data xarray Dataset (main dataset of an MSv4) of an ASDM
    partition.

    The dataset has its coordinates, the lazily loaded data variables
    (VISIBILITY or SPECTRUM, FLAG, WEIGHT), the time variables
    (TIME_CENTROID, EFFECTIVE_INTEGRATION_TIME) and the metadata attributes
    (info dicts, data groups) of the MSv4 schema. UVW is added by
    :func:`open_partition`, as it needs the antenna and field datasets.

    All the variables with a time dimension (except the time index) have the
    same ``preferred_chunks`` along time in their encoding (see
    :func:`preferred_time_chunk`), which xarray uses when the ASDM is opened with
    ``chunks={}``.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the raw data.
    partition_descr : dict[str, np.ndarray]
        Dictionary containing the partition description (see
        :func:`open_partition`).

    Returns
    -------
    tuple[xr.Dataset, int, int, bool]
        A tuple containing:

        - xds : xr.Dataset
            The xarray Dataset with the correlated data, with appropriate
            coordinates, variables, and metadata.
        - num_antenna : int
            The number of antennas of the partition (ConfigDescription.numAntenna).
        - spw_id : int
            The spectral window ID.
        - is_single_dish : bool
            Whether the dataset is single dish (a SpectrumXds, type "spectrum") or
            interferometric/radiometer (a VisibilityXds, type "visibility" or
            "radiometer").

    Notes
    -----
    The dataset type follows the ASDM processor type and correlation mode of
    the partition (see :func:`find_xds_type`).
    """

    datetime_now = datetime.datetime.now(datetime.UTC).isoformat()
    xds = xr.Dataset(
        attrs={
            "schema_version": MSV4_SCHEMA_VERSION,
            "creator": {
                "software_name": "xradio",
                "version": importlib.metadata.version("xradio"),
            },
            "creation_date": datetime_now,
            "type": "visibility",
        }
    )

    info_dicts = create_info_dicts(asdm, xds, partition_descr)
    xds.attrs.update(info_dicts)

    config = _get_partition_config(asdm, partition_descr)
    xds_type = find_xds_type(
        config["correlationMode"], info_dicts["processor_info"]["type"]
    )
    xds.attrs["type"] = xds_type
    is_single_dish = xds_type == "spectrum"

    (
        coords,
        coord_attrs,
        num_antenna,
        spw_id,
        bdf_spw_id,
        time_vars,
        time_indices_by_bdf,
    ) = create_coordinates(asdm, partition_descr, is_single_dish)
    xds = xds.assign_coords(coords)
    for coord_name in coords:
        if coord_name in coord_attrs:
            xds.coords[coord_name].attrs = coord_attrs[coord_name]

    xds = xds.assign(time_vars)
    xds = xds.assign(
        create_data_vars(
            xds, partition_descr["BDFPath"], bdf_spw_id, time_indices_by_bdf
        )
    )
    xds = _set_preferred_time_chunk(xds, preferred_time_chunk(xds, time_indices_by_bdf))

    description = "Base data group derived from data in ASDM BDFs"
    # the APC applies to the crossData only (not to AUTO_ONLY data, e.g. WVR)
    if not is_single_dish and _enum_name(config["correlationMode"]) != "AUTO_ONLY":
        loaded_apc = _get_loaded_atm_phase_correction(config)
        if loaded_apc:
            description += (
                f" (cross-correlations with atmospheric phase correction {loaded_apc})"
            )
    data_group_base = {
        "correlated_data": "SPECTRUM" if is_single_dish else "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "field_and_source": "field_and_source_base_xds",
        "description": description,
        "date": datetime_now,
    }
    xds.attrs.update({"data_groups": {"base": data_group_base}})

    return xds, num_antenna, spw_id, is_single_dish


def find_xds_type(correlation_mode, processor_type: str) -> str:
    """
    Find the MSv4 dataset type of a partition.

    Parameters
    ----------
    correlation_mode : pyasdm.enumerations.CorrelationMode or str
        Correlation mode of the ConfigDescription of the partition
        ("CROSS_AND_AUTO", "AUTO_ONLY", ...).
    processor_type : str
        Processor type of the partition ("CORRELATOR", "SPECTROMETER",
        "RADIOMETER"), as in ``processor_info["type"]``.

    Returns
    -------
    str
        "radiometer" for RADIOMETER processors (interferometric layout, as the
        MSv2 converter does), "spectrum" for (other) AUTO_ONLY data (single dish),
        "visibility" otherwise.
    """
    if _enum_name(processor_type) == "RADIOMETER":
        return "radiometer"
    if _enum_name(correlation_mode) == "AUTO_ONLY":
        return "spectrum"
    return "visibility"


def create_data_vars(
    xds: xr.Dataset, bdf_paths: list[str], bdf_spw_id: int, time_indices_by_bdf: dict
) -> dict[str, tuple]:
    """
    Create the lazily loaded data variables of the correlated dataset.

    The layout follows the coordinates of ``xds``: interferometric
    (``baseline_id`` dimension, VISIBILITY) or single dish (``antenna_name``
    dimension, SPECTRUM).

    Parameters
    ----------
    xds : xr.Dataset
        Input xarray Dataset with the dimensions 'time', 'baseline_id' (or
        'antenna_name' for single dish), 'frequency' and 'polarization'.
    bdf_paths : list[str]
        Paths to BDFs with data/flags for the partition, in time order
    bdf_spw_id : int
        Index of the SPW to load in the BDFs (position in the
        ConfigDescription/BDF list of SPWs)
    time_indices_by_bdf : dict
        Dictionary that gives the list of BDFs ("bdf_names") and the index of
        their first integration along the time axis ("bdf_start", with one more
        entry for the end of the last BDF)

    Returns
    -------
    dict[str, tuple]
        A dictionary with the (dims, array, attrs) tuples that define the data
        variables:

        - VISIBILITY (complex) or SPECTRUM (real): correlated data, dims (time,
          baseline_id | antenna_name, frequency, polarization)
        - WEIGHT : weights with the same dims
        - FLAG : boolean flags with the same dims

        Their preferred dask chunks are set by :func:`create_correlated_xds`,
        together with the other time-dependent variables (see
        :func:`preferred_time_chunk`).
    """

    data_vars = {}

    if "baseline_id" not in xds.sizes and "antenna_name" in xds.sizes:
        second_dim = "antenna_name"
        correlated_data_name = "SPECTRUM"
        correlated_data_class = asdm_backend_arrays.SpectrumArray
    else:
        second_dim = "baseline_id"
        correlated_data_name = "VISIBILITY"
        correlated_data_class = asdm_backend_arrays.VisibilityArray

    dims = ["time", second_dim, "frequency", "polarization"]
    shape = tuple(xds.sizes[dim] for dim in dims)

    data_vars[correlated_data_name] = (
        dims,
        xr.core.indexing.LazilyIndexedArray(
            correlated_data_class(shape, bdf_paths, bdf_spw_id, time_indices_by_bdf)
        ),
        # The BDFs give uncalibrated correlator values
        {"type": "quantity", "units": ""},
    )

    data_vars["WEIGHT"] = (
        dims,
        xr.core.indexing.LazilyIndexedArray(asdm_backend_arrays.WeightArray(shape)),
        {},
    )

    data_vars["FLAG"] = (
        dims,
        xr.core.indexing.LazilyIndexedArray(
            asdm_backend_arrays.FlagArray(
                shape, bdf_paths, bdf_spw_id, time_indices_by_bdf
            )
        ),
        {},
    )

    return data_vars


def _preferred_chunk_target_bytes() -> int:
    """
    Target size (bytes) of the preferred dask chunks of the correlated dataset:
    the dask "array.chunk-size" setting (128 MiB by default), or
    DEFAULT_PREFERRED_CHUNK_BYTES when dask or the setting is not available.
    """
    try:
        import dask
        from dask.utils import parse_bytes

        target = int(parse_bytes(dask.config.get("array.chunk-size")))
    except (ImportError, KeyError, TypeError, ValueError):
        return DEFAULT_PREFERRED_CHUNK_BYTES
    return target if target > 0 else DEFAULT_PREFERRED_CHUNK_BYTES


def preferred_time_chunk(
    xds: xr.Dataset, time_indices_by_bdf: dict, target_bytes: int | None = None
) -> int:
    """
    Preferred dask chunk size along time (number of integrations per chunk) of
    the variables of a correlated dataset.

    xarray uses it when the ASDM is opened with ``chunks={}`` (and as a hint
    with ``chunks="auto"``, which can merge several preferred chunks). The BDF
    loaders reach the integrations of a BDF one after the other, so every chunk
    opens its BDFs again and skips the integrations of the BDF before its start:
    with one integration per chunk, reading a BDF of N integrations costs N opens
    and O(N^2) subset header parses. The preferred chunk is therefore the BDF
    (the storage unit of the ASDM data), split when needed to fit in a memory
    target:

    - the number of integrations of the largest time-dependent data variable that
      fit in ``target_bytes`` (at least 1),
    - at most the largest number of integrations of a BDF of the partition (the
      length of the time axis when the time indices of the BDFs are not
      available),
    - aligned with the BDFs when they have the same number of integrations (see
      :func:`_align_time_chunk_with_bdfs`). Otherwise some chunks span two BDFs.

    It is one size for all the chunks (dask chunks (k, k, ..., rest)), as Zarr
    needs uniform chunks: a dataset opened with ``chunks={}`` can be written with
    ``to_zarr``. Note that xarray warns when explicit ``chunks`` split the
    preferred chunks (for example a BDF in several chunks).

    Parameters
    ----------
    xds : xr.Dataset
        Correlated dataset with its time-dependent data variables (VISIBILITY or
        SPECTRUM, WEIGHT, FLAG, ...).
    time_indices_by_bdf : dict
        Time indices of the BDFs, with "bdf_start": index of the first integration
        of every BDF, plus the end of the last BDF.
    target_bytes : int | None, optional
        Target size of the chunks of the largest variable, by default the dask
        "array.chunk-size" setting (128 MiB).

    Returns
    -------
    int
        Number of integrations per chunk (the last chunk can be smaller).
    """
    if target_bytes is None:
        target_bytes = _preferred_chunk_target_bytes()
    num_time = int(xds.sizes.get("time", 0))

    bytes_per_integration = max(
        (
            var.dtype.itemsize
            * math.prod(size for dim, size in var.sizes.items() if dim != "time")
            for var in xds.data_vars.values()
            if "time" in var.dims
        ),
        default=1,
    )
    time_chunk = max(1, int(target_bytes) // max(1, bytes_per_integration))

    num_integrations = _bdf_num_integrations(time_indices_by_bdf, num_time)
    if num_integrations is None:
        return min(time_chunk, max(1, num_time))
    time_chunk = min(time_chunk, max(1, int(num_integrations.max())))
    return _align_time_chunk_with_bdfs(time_chunk, num_integrations)


def _bdf_num_integrations(
    time_indices_by_bdf: dict | None, num_time: int
) -> np.ndarray | None:
    """
    Number of integrations of every BDF of a partition, from the time indices of
    the BDFs ("bdf_start"). None when they are not available or do not match
    the time axis (first index 0, last index num_time, non-decreasing).
    """
    if not time_indices_by_bdf or time_indices_by_bdf.get("bdf_start") is None:
        return None
    bdf_start = np.asarray(time_indices_by_bdf["bdf_start"], dtype=np.int64)
    if bdf_start.ndim != 1 or len(bdf_start) < 2:
        return None
    num_integrations = np.diff(bdf_start)
    if bdf_start[0] != 0 or bdf_start[-1] != num_time or np.any(num_integrations < 0):
        return None
    return num_integrations


def _align_time_chunk_with_bdfs(time_chunk: int, num_integrations: np.ndarray) -> int:
    """
    Adjust a time chunk size, smaller than the BDFs, so that the chunk
    boundaries fall on BDF boundaries: no chunk then needs integrations from two
    BDFs.

    This is done when the BDFs have the same number ``n`` of integrations, except
    the last one which can have fewer: the chunk becomes the largest divisor of
    ``n`` not larger than the chunk, if it is at least half of it.

    Parameters
    ----------
    time_chunk : int
        Number of integrations per chunk (>= 1).
    num_integrations : np.ndarray
        Number of integrations of every BDF, in time order.

    Returns
    -------
    int
        The aligned chunk size, never larger than time_chunk. time_chunk itself
        when it is not smaller than the BDFs, when there is only one BDF, when the
        BDFs have different numbers of integrations or when no divisor fits.
    """
    if len(num_integrations) < 2:
        return time_chunk
    n = int(num_integrations[0])
    if (
        time_chunk >= n
        or np.any(num_integrations[:-1] != n)
        or not 0 < num_integrations[-1] <= n
    ):
        return time_chunk

    for divisor in range(time_chunk, (time_chunk + 1) // 2 - 1, -1):
        if n % divisor == 0:
            return divisor
    return time_chunk


def _set_preferred_time_chunk(xds: xr.Dataset, time_chunk: int) -> xr.Dataset:
    """
    A (shallow) copy of a dataset where every variable with a time dimension,
    except the time index, has ``preferred_chunks`` {"time": time_chunk} in its
    encoding, so that opening with ``chunks={}`` gives the same dask chunks along
    time for all the variables (Dataset.chunks, map_blocks, to_zarr need it).
    """
    xds = xds.copy(deep=False)
    for name, var in xds.variables.items():
        if "time" not in var.dims or name in xds.xindexes:
            continue
        preferred_chunks = dict(var.encoding.get("preferred_chunks", {}))
        preferred_chunks["time"] = int(time_chunk)
        var.encoding = {**var.encoding, "preferred_chunks": preferred_chunks}
    return xds


def _get_preferred_time_chunk(xds: xr.Dataset) -> int:
    """
    Preferred dask chunk size along time of a correlated dataset (set by
    :func:`create_correlated_xds`, the same for all its time-dependent
    variables), from its FLAG variable.
    """
    return int(xds["FLAG"].encoding["preferred_chunks"]["time"])


def create_coordinates(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
    is_single_dish: bool = False,
) -> tuple[dict, dict, int, int, int, dict, dict]:
    """
    Create coordinate systems and associated metadata from ASDM data.

    This function extracts and processes coordinate information from an ALMA
    Science Data Model (ASDM) dataset, including time, frequency, polarization,
    baseline (or antenna), scan and field coordinates. It handles both
    interferometric and single-dish observations.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        Input ASDM object containing the observation data
    partition_descr : dict[str, np.ndarray]
        Partition description (see :func:`open_partition`). Expected keys include
        'BDFPath', 'configDescriptionId', 'dataDescriptionId', 'scanNumber',
        'fieldId', 'scanIntent' and 'per_bdf'.
    is_single_dish : bool, optional
        If True, the coordinates of a SpectrumXds (``antenna_name`` dimension) are
        created. Otherwise those of a VisibilityXds (``baseline_id`` dimension,
        baseline antenna names and ``uvw_label``). Default is False.

    Returns
    -------
    coords : dict
        Dictionary of coordinate arrays and their dimensions, ready for xarray
        dataset creation.
    attrs : dict
        Dictionary of coordinate attributes (time and frequency measures, scan
        intents).
    num_antenna : int
        Number of antennas of the partition.
    spw_id : int
        Spectral window ID.
    bdf_spw_id : int
        Index of the spectral window in the BDFs.
    time_vars : dict
        Dictionary containing the time-related variables EFFECTIVE_INTEGRATION_TIME
        and TIME_CENTROID.
    time_indices_by_bdf : dict
        Dictionary that gives the list of BDFs and the index of their first
        integration along the time axis

    Raises
    ------
    RuntimeError
        If the metadata of the partition is inconsistent (missing table rows, BDF
        not matching the SpectralWindow, per-BDF metadata not matching the
        integrations, etc.).
    """

    # Metadata checks first (cheap), before reading the times from all the BDFs
    config = _get_partition_config(asdm, partition_descr)
    antenna_names = _get_config_antenna_names(asdm, config)
    polarization, spw_id, data_description_id = _create_polarizations_coord(
        asdm, partition_descr
    )
    frequency, frequency_attrs, num_chan = _create_frequency_coord_attrs(asdm, spw_id)
    bdf_spw_id = _find_bdf_spw_id(config["dataDescriptionId"], data_description_id)
    _check_bdf_spectral_window(
        np.atleast_1d(partition_descr["BDFPath"])[0], bdf_spw_id, spw_id, num_chan
    )

    scans_metadata_df = _get_scans_metadata(asdm, partition_descr)
    time_centers, durations, actual_times, actual_durations, time_indices_by_bdf = (
        load_times_from_partition_bdfs(partition_descr["BDFPath"], scans_metadata_df)
    )
    time_centers = np.asarray(time_centers, dtype=np.float64)
    num_time = len(time_centers)
    if num_time == 0:
        raise RuntimeError(
            f"No integrations found in the BDFs of the partition: "
            f"{partition_descr['BDFPath']}"
        )

    coords = {}
    attrs = {}

    # Absolute times are seconds since the Unix epoch (load_times_from_partition_bdfs)
    coords["time"] = (["time"], time_centers)
    attrs["time"] = make_time_measure_attrs(
        "s", ASDM_TIME_SCALE, time_format=ASDM_TIME_FORMAT
    )
    # durations is not always unique in ASDM partitions (short last integrations,
    # etc.), so check_if_consistent would fail
    attrs["time"]["integration_time"] = make_quantity(float(durations[0]), "s")

    coords["scan_name"], attrs["scan_name"] = _create_scan_name_coord_attrs(
        partition_descr, time_indices_by_bdf, num_time
    )
    coords["field_name"] = _create_field_name_coord(
        asdm, partition_descr, time_indices_by_bdf, num_time
    )

    num_antenna = len(antenna_names)
    if is_single_dish:
        coords["antenna_name"] = (["antenna_name"], np.array(antenna_names, dtype=str))
        second_dim = "antenna_name"
        num_rows = num_antenna
    else:
        baseline_coords = _create_baseline_coords(
            antenna_names, config["correlationMode"]
        )
        coords.update(baseline_coords)
        second_dim = "baseline_id"
        num_rows = len(baseline_coords["baseline_id"])
        coords["uvw_label"] = np.array(["u", "v", "w"])

    coords["polarization"] = polarization
    coords["frequency"] = frequency
    attrs["frequency"] = frequency_attrs

    time_vars = _create_time_vars(actual_durations, actual_times, num_rows, second_dim)

    return (
        coords,
        attrs,
        num_antenna,
        spw_id,
        bdf_spw_id,
        time_vars,
        time_indices_by_bdf,
    )


def _enum_name(value) -> str:
    """Name of a pyasdm enumeration value (or the str of a plain value)."""
    if hasattr(value, "getName"):
        return str(value.getName())
    return str(value)


def _get_single_value(partition_descr: dict, key: str):
    """The single value of an entry of the partition description."""
    values = pd.unique(np.atleast_1d(np.asarray(partition_descr[key])).ravel())
    if len(values) != 1:
        raise RuntimeError(
            f"Expected exactly one {key} in the partition description, got "
            f"{list(values)}"
        )
    return values[0]


def _get_partition_config(asdm: pyasdm.ASDM, partition_descr: dict) -> pd.Series:
    """
    The ConfigDescription row of a partition (configDescriptionId, numAntenna,
    correlationMode, dataDescriptionId, antennaId and atmPhaseCorrection lists).
    """
    config_description_id = int(
        _get_single_value(partition_descr, "configDescriptionId")
    )
    sdm_config_description_attrs = [
        "configDescriptionId",
        "numAntenna",
        "correlationMode",
        "dataDescriptionId",
        "antennaId",
        "atmPhaseCorrection",
    ]
    config_description_df = exp_asdm_table_to_df(
        asdm, "ConfigDescription", sdm_config_description_attrs
    )
    config = config_description_df.loc[
        config_description_df["configDescriptionId"] == config_description_id
    ]
    if config.empty:
        raise RuntimeError(
            f"No row with configDescriptionId={config_description_id} in the "
            "ConfigDescription table"
        )
    return config.iloc[0]


def _get_loaded_atm_phase_correction(config: pd.Series) -> str | None:
    """
    Atmospheric phase correction of the cross-correlation data loaded from the
    BDFs (AP_UNCORRECTED when there are several, see
    :func:`~xradio.measurement_set._utils._asdm._utils._bdf.shapes.select_apc_index`),
    from ConfigDescription.atmPhaseCorrection. None if it cannot be determined.

    Called once per partition when it is opened, so this is where the warning
    about loading an APC other than AP_UNCORRECTED (when it is the only one) is
    logged. The BDF loaders, which run for every BDF and chunk, do not warn.
    """
    apc_names = [
        _enum_name(apc) for apc in np.atleast_1d(config.get("atmPhaseCorrection", []))
    ]
    if not apc_names:
        return None
    try:
        return apc_names[select_apc_index(apc_names, warn=True)]
    except NotImplementedError:
        return None


def _get_scans_metadata(asdm: pyasdm.ASDM, partition_descr: dict) -> pd.DataFrame:
    """The Scan table rows of the scans (of the execution block) of a partition."""
    sdm_scan_attrs = ["execBlockId", "scanNumber", "startTime", "endTime", "numSubscan"]
    scan_df = exp_asdm_table_to_df(asdm, "Scan", sdm_scan_attrs)
    selected = scan_df["scanNumber"].isin(np.atleast_1d(partition_descr["scanNumber"]))
    if "execBlockId" in partition_descr:
        selected &= scan_df["execBlockId"].isin(
            np.atleast_1d(partition_descr["execBlockId"])
        )
    return scan_df.loc[selected]


def _get_config_antenna_ids(config: pd.Series) -> list[int]:
    """
    Antenna ids of a ConfigDescription, in the order of the BDF antenna slots
    (the first numAntenna entries of ConfigDescription.antennaId).

    Parameters
    ----------
    config : pd.Series
        ConfigDescription row (see :func:`_get_partition_config`)

    Returns
    -------
    list[int]
        Antenna ids, one per BDF antenna slot (numAntenna).

    Raises
    ------
    RuntimeError
        If the ConfigDescription has fewer antennaId entries than numAntenna.
    """
    num_antenna = int(config["numAntenna"])
    antenna_ids = [int(antenna_id) for antenna_id in np.atleast_1d(config["antennaId"])]
    if len(antenna_ids) < num_antenna:
        raise RuntimeError(
            f"ConfigDescription {config['configDescriptionId']} has numAntenna="
            f"{num_antenna} but only {len(antenna_ids)} antennaId entries: "
            f"{antenna_ids}"
        )
    if len(antenna_ids) > num_antenna:
        xradio_logger().warning(
            f"ConfigDescription {config['configDescriptionId']} has numAntenna="
            f"{num_antenna} but {len(antenna_ids)} antennaId entries. Using the first "
            f"{num_antenna} antennaId entries: {antenna_ids[:num_antenna]}"
        )
        antenna_ids = antenna_ids[:num_antenna]

    return antenna_ids


def _get_config_antenna_names(asdm: pyasdm.ASDM, config: pd.Series) -> list[str]:
    """
    Names of the antennas of a ConfigDescription, in the order of the BDF
    antenna slots (ConfigDescription.antennaId).

    Parameters
    ----------
    asdm : pyasdm.ASDM
        Input ASDM object
    config : pd.Series
        ConfigDescription row (see :func:`_get_partition_config`)

    Returns
    -------
    list[str]
        Antenna names, one per BDF antenna slot (numAntenna).

    Raises
    ------
    RuntimeError
        If the ConfigDescription has fewer antennaId entries than numAntenna, or
        refers to antennas not present in the Antenna table.
    """
    antenna_ids = _get_config_antenna_ids(config)

    antenna_df = exp_asdm_table_to_df(asdm, "Antenna", ["antennaId", "name"])
    name_by_id = {
        int(antenna_id): str(name)
        for antenna_id, name in zip(
            antenna_df["antennaId"], antenna_df["name"], strict=True
        )
    }
    missing = [antenna_id for antenna_id in antenna_ids if antenna_id not in name_by_id]
    if missing:
        raise RuntimeError(
            f"ConfigDescription {config['configDescriptionId']} refers to antennaId "
            f"{missing}, not found in the Antenna table (antennaId "
            f"{sorted(name_by_id)})"
        )

    return [name_by_id[antenna_id] for antenna_id in antenna_ids]


def _per_bdf_values(partition_descr: dict, key: str) -> np.ndarray:
    """
    Values of a Main table column (scanNumber, fieldId, ...) for every BDF of
    the partition, aligned with partition_descr["BDFPath"].

    They come from partition_descr["per_bdf"]. For partition descriptions without
    per-BDF information, a single value of partition_descr[key] can be used for
    all the BDFs.

    Raises
    ------
    RuntimeError
        If the per-BDF values are not aligned with the BDFs, or are not
        available and partition_descr[key] has more than one value.
    """
    bdf_paths = np.atleast_1d(partition_descr["BDFPath"])
    per_bdf = partition_descr.get("per_bdf")
    if per_bdf is not None and key in per_bdf:
        values = np.atleast_1d(np.asarray(per_bdf[key]))
        if len(values) != len(bdf_paths):
            raise RuntimeError(
                f"partition_descr['per_bdf']['{key}'] has {len(values)} values for "
                f"{len(bdf_paths)} BDFs"
            )
        if "BDFPath" in per_bdf and not np.array_equal(
            np.asarray(per_bdf["BDFPath"], dtype=str), bdf_paths.astype(str)
        ):
            raise RuntimeError(
                "partition_descr['per_bdf'] is not aligned with "
                "partition_descr['BDFPath']"
            )
        return values

    unique_values = pd.unique(np.atleast_1d(np.asarray(partition_descr[key])).ravel())
    if len(unique_values) != 1:
        raise RuntimeError(
            f"The partition has several {key} values ({list(unique_values)}) but no "
            f"per-BDF {key} information (partition_descr['per_bdf']) to assign them "
            "to its integrations"
        )
    return np.full(len(bdf_paths), unique_values[0])


def _expand_per_bdf_to_time(
    values_per_bdf: np.ndarray, time_indices_by_bdf: dict, num_time: int, what: str
) -> np.ndarray:
    """
    Expand values given per BDF to the time axis (one value per integration).

    Parameters
    ----------
    values_per_bdf : np.ndarray
        One value per BDF of the partition (in the order of the BDFs).
    time_indices_by_bdf : dict
        Time indices of the BDFs, with "bdf_start": index of the first integration
        of every BDF, plus the end of the last BDF.
    num_time : int
        Length of the time axis.
    what : str
        Name of the values, for error messages.

    Returns
    -------
    np.ndarray
        One value per integration (time).

    Raises
    ------
    RuntimeError
        If the time indices of the BDFs do not match the values or the time axis,
        or are not available and the values are not all the same.
    """
    values_per_bdf = np.asarray(values_per_bdf)
    bdf_start = None
    if time_indices_by_bdf:
        bdf_start = time_indices_by_bdf.get("bdf_start")

    if bdf_start is None or len(bdf_start) == 0:
        unique_values = pd.unique(values_per_bdf.ravel())
        if len(unique_values) == 1:
            return np.full(num_time, values_per_bdf.ravel()[0])
        raise RuntimeError(
            f"Cannot assign the {what} values {list(unique_values)} of the BDFs to the "
            "integrations without the time indices of the BDFs"
        )

    bdf_start = np.asarray(bdf_start, dtype=np.int64)
    num_integrations = np.diff(bdf_start)
    if (
        len(num_integrations) != len(values_per_bdf)
        or bdf_start[0] != 0
        or bdf_start[-1] != num_time
        or np.any(num_integrations < 0)
    ):
        raise RuntimeError(
            f"The time indices of the BDFs ({bdf_start.tolist()}) do not match the "
            f"{len(values_per_bdf)} {what} values and the {num_time} integrations of "
            "the partition"
        )
    return np.repeat(values_per_bdf, num_integrations)


def _create_scan_name_coord_attrs(
    partition_descr: dict, time_indices_by_bdf: dict, num_time: int
) -> tuple[tuple, dict]:
    """
    scan_name coordinate (scan number of every integration, as str) and its
    scan_intents attribute (one "INTENT#SUBINTENT" string per intent of the
    partition).
    """
    scan_numbers = _per_bdf_values(partition_descr, "scanNumber")
    scan_numbers_by_time = _expand_per_bdf_to_time(
        scan_numbers.astype(np.int64), time_indices_by_bdf, num_time, "scanNumber"
    )
    coord_scan_name = (["time"], scan_numbers_by_time.astype(str))
    attrs_scan_name = {
        "scan_intents": [
            str(intent) for intent in np.atleast_1d(partition_descr["scanIntent"])
        ]
    }

    return coord_scan_name, attrs_scan_name


def _create_field_name_coord(
    asdm: pyasdm.ASDM, partition_descr: dict, time_indices_by_bdf: dict, num_time: int
) -> tuple:
    """
    field_name coordinate: field of every integration, named
    "<Field.fieldName>_<fieldId>" as in the field_and_source dataset.
    """
    field_ids = _per_bdf_values(partition_descr, "fieldId").astype(np.int64)

    sdm_field_attrs = ["fieldId", "fieldName"]
    field_df = exp_asdm_table_to_df(asdm, "Field", sdm_field_attrs)
    name_by_id = {
        int(field_id): name
        for field_id, name in zip(
            field_df["fieldId"], field_df["fieldName"], strict=True
        )
    }
    missing = sorted({int(fid) for fid in field_ids if int(fid) not in name_by_id})
    if missing:
        raise RuntimeError(f"fieldId {missing} not found in the Field table")

    field_names = np.array(
        [make_field_name(name_by_id[int(fid)], int(fid)) for fid in field_ids],
        dtype=str,
    )
    field_names_by_time = _expand_per_bdf_to_time(
        field_names, time_indices_by_bdf, num_time, "fieldId"
    )
    return (["time"], field_names_by_time)


def _create_baseline_coords(antenna_names: list[str], correlation_mode) -> dict:
    """
    baseline_id, baseline_antenna1_name and baseline_antenna2_name coordinates,
    in the order of the BDF baselines.

    Parameters
    ----------
    antenna_names : list[str]
        Antenna names in the order of the BDF antenna slots.
    correlation_mode : pyasdm.enumerations.CorrelationMode or str
        AUTO_ONLY (one auto-correlation per antenna) or CROSS_AND_AUTO (cross
        baselines in BDF order followed by the auto-correlations).

    Returns
    -------
    dict
        The baseline coordinates.
    """
    num_antenna = len(antenna_names)
    mode = _enum_name(correlation_mode)
    if mode == "AUTO_ONLY":
        baseline_antenna1_id = baseline_antenna2_id = np.arange(num_antenna)
    elif mode == "CROSS_AND_AUTO":
        baseline_antenna1_id, baseline_antenna2_id = (
            _generate_baseline_antennax_id_as_in_bdf(num_antenna)
        )
    else:
        raise RuntimeError(
            "Only AUTO_ONLY and CROSS_AND_AUTO correlation modes supported in ALMA. "
            f"Found {mode=}"
        )

    names = np.array(antenna_names, dtype=str)
    coords_baselines = {
        "baseline_antenna1_name": (["baseline_id"], names[baseline_antenna1_id]),
        "baseline_antenna2_name": (["baseline_id"], names[baseline_antenna2_id]),
        "baseline_id": np.arange(len(baseline_antenna1_id)),
    }

    return coords_baselines


def _create_polarizations_coord(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> tuple[np.ndarray, int, int]:
    """
    polarization coordinate, SPW id and DataDescription id of the partition.
    """
    # From dataDescriptionId get SPW and polarization IDs
    dd_id = int(_get_single_value(partition_descr, "dataDescriptionId"))
    sdm_dd_attrs = ["dataDescriptionId", "spectralWindowId", "polOrHoloId"]
    data_description_df = exp_asdm_table_to_df(asdm, "DataDescription", sdm_dd_attrs)
    data_description = data_description_df.loc[
        data_description_df["dataDescriptionId"] == dd_id
    ]
    if data_description.empty:
        raise RuntimeError(
            f"No row with dataDescriptionId={dd_id} in the DataDescription table"
        )
    spw_id = int(data_description["spectralWindowId"].values[0])
    pol_setup_id = int(data_description["polOrHoloId"].values[0])

    # polarization coord
    sdm_polarization_attrs = ["polarizationId", "numCorr", "corrType"]
    polarization_df = exp_asdm_table_to_df(asdm, "Polarization", sdm_polarization_attrs)
    polarization_metadata = polarization_df.loc[
        polarization_df["polarizationId"] == pol_setup_id
    ]
    if polarization_metadata.empty:
        raise RuntimeError(
            f"No row with polarizationId={pol_setup_id} in the Polarization table"
        )
    num_corr = int(polarization_metadata["numCorr"].values[0])
    polarization_setup = np.array(
        polarization_metadata["corrType"].values[0][:num_corr], dtype=str
    )

    return polarization_setup, spw_id, dd_id


def _per_time_data(
    values: np.ndarray, num_rows: int, what: str
) -> xr.core.indexing.LazilyIndexedArray:
    """
    Per-integration values as the lazily indexed (time, num_rows) data of a
    variable (see asdm_backend_arrays.PerTimeArray): no memory is allocated
    per row, and the accessed values are writeable arrays.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"Cannot use {what} values with shape {values.shape} for {num_rows} "
            "baselines/antennas: one value per integration (1-D) is expected"
        )
    return xr.core.indexing.LazilyIndexedArray(
        asdm_backend_arrays.PerTimeArray(values, num_rows)
    )


def _create_time_vars(
    actual_durations: np.ndarray,
    actual_times: np.ndarray,
    num_rows: int,
    second_dim: str = "baseline_id",
) -> dict:
    """
    EFFECTIVE_INTEGRATION_TIME and TIME_CENTROID data variables.

    Parameters
    ----------
    actual_durations : np.ndarray
        Actual duration (s) of every integration.
    actual_times : np.ndarray
        Actual time (centroid) of every integration, seconds since the Unix
        epoch.
    num_rows : int
        Number of baselines (or antennas, single dish).
    second_dim : str, optional
        Name of the second dimension, "baseline_id" (default) or "antenna_name".

    Returns
    -------
    dict
        The (dims, data, attrs) of the variables, dims (time, second_dim). The
        times and durations from the BDFs do not depend on the baseline, so the
        same value is used for all the baselines of an integration. The data
        are lazily indexed, like the other data variables (see
        :func:`_per_time_data`).
    """
    time_vars = {
        "EFFECTIVE_INTEGRATION_TIME": (
            ["time", second_dim],
            _per_time_data(actual_durations, num_rows, "EFFECTIVE_INTEGRATION_TIME"),
            make_quantity_attrs("s"),
        ),
        "TIME_CENTROID": (
            ["time", second_dim],
            _per_time_data(actual_times, num_rows, "TIME_CENTROID"),
            make_time_measure_attrs("s", ASDM_TIME_SCALE, time_format=ASDM_TIME_FORMAT),
        ),
    }

    return time_vars


def _get_frequency_observer(asdm: pyasdm.ASDM, spw_id: int) -> str:
    """
    MSv4 spectral coordinate observer of an SPW, from SpectralWindow.measFreqRef
    (TOPO when absent).
    """
    frame = get_reference_frame(asdm, spw_id)
    observer = ASDM_FREQUENCY_FRAME_TO_MSV4_OBSERVER.get(frame)
    if observer is None:
        xradio_logger().warning(
            f"The frequency reference frame (measFreqRef) {frame} of spectral window "
            f"{spw_id} has no MSv4 equivalent. It is used as is for the frequency "
            "observer."
        )
        observer = frame
    return observer


def _create_frequency_coord_attrs(
    asdm: pyasdm.ASDM, spw_id: int
) -> tuple[tuple, dict, int]:
    """frequency coordinate and its attributes, and the number of channels."""
    sdm_spw_attrs = [
        "spectralWindowId",
        "numChan",
        "refFreq",
    ]
    # These are optional attrs of the ASDM table, better dealt with via util
    # functions that check for their presence and alternatives: "chanFreqStart",
    # "chanFreqStep", "chanFreqArray", "chanWidthArray", "effectiveBwArray",
    # "measFreqRef".
    spw_df = exp_asdm_table_to_df(asdm, "SpectralWindow", sdm_spw_attrs)
    spectral_window = spw_df.loc[spw_df["spectralWindowId"] == spw_id]
    if spectral_window.empty:
        raise RuntimeError(
            f"No row with spectralWindowId={spw_id} in the SpectralWindow table"
        )
    spw_name = get_spw_name(asdm, spw_id)
    num_chan = int(spectral_window["numChan"].values[0])
    frequency_centers = np.asarray(
        get_spw_frequency_centers(asdm, spw_id, num_chan), dtype=np.float64
    )

    observer = _get_frequency_observer(asdm, spw_id)
    frequency_coord = (["frequency"], frequency_centers)
    frequency_attrs = make_spectral_coord_measure_attrs("Hz", observer=observer)
    frequency_attrs.update(
        {
            "spectral_window_name": ensure_spw_name_conforms(spw_name, spw_id),
            "spectral_window_intents": ["UNSPECIFIED"],
            "reference_frequency": make_spectral_coord_reference_dict(
                float(spectral_window["refFreq"].values[0]), "Hz", observer
            ),
            # nominal channel bandwidth, positive (as in the MSv2 converter)
            "channel_width": make_quantity(abs(get_chan_width(asdm, spw_id)), "Hz"),
        }
    )

    return frequency_coord, frequency_attrs, num_chan


def _find_bdf_spw_id(
    config_data_description_ids: np.ndarray, data_description_id: int
) -> int:
    """
    Index of the SPW of a partition in its BDFs.

    The BDF spectral windows (basebands, and SPWs within basebands) are in the
    order of the ConfigDescription.dataDescriptionId list. The BDFs have no SPW
    or DataDescription ids, only positions.

    Parameters
    ----------
    config_data_description_ids : np.ndarray
        ConfigDescription.dataDescriptionId of the partition, in config order.
    data_description_id : int
        DataDescription id of the partition.

    Returns
    -------
    int
        Position of the data description in the config (and BDF) list.

    Raises
    ------
    RuntimeError
        If the data description is not in the ConfigDescription list.
    """
    config_dd_ids = [int(dd_id) for dd_id in np.atleast_1d(config_data_description_ids)]
    if int(data_description_id) not in config_dd_ids:
        raise RuntimeError(
            f"dataDescriptionId {data_description_id} is not in the "
            f"ConfigDescription.dataDescriptionId list {config_dd_ids}"
        )
    return config_dd_ids.index(int(data_description_id))


def _check_bdf_spectral_window(
    bdf_path: str, bdf_spw_id: int, spw_id: int, num_chan: int
) -> None:
    """
    Check that the SPW at position bdf_spw_id of a BDF has the number of
    channels of the SpectralWindow (numSpectralPoint == numChan).

    The check is skipped (with a debug message) if the BDF header cannot be
    read.

    Raises
    ------
    RuntimeError
        If the BDF has no SPW at that position, or a different number of channels.
    """
    try:
        with open_bdf(bdf_path) as bdf_reader:
            basebands = bdf_reader.getHeader().getBasebandsList()
    except BDFOpenError as exc:
        xradio_logger().debug(
            f"Could not read the header of BDF {bdf_path} to check its spectral "
            f"windows: {exc}"
        )
        return

    bdf_spws = [spw for baseband in basebands for spw in baseband["spectralWindows"]]
    if bdf_spw_id >= len(bdf_spws):
        raise RuntimeError(
            f"Spectral window {spw_id} should be at position {bdf_spw_id} in the BDF "
            f"{bdf_path}, which has only {len(bdf_spws)} spectral windows"
        )
    num_spectral_point = int(bdf_spws[bdf_spw_id]["numSpectralPoint"])
    if num_spectral_point != num_chan:
        raise RuntimeError(
            f"Spectral window {spw_id} has {num_chan} channels but the spectral window "
            f"at position {bdf_spw_id} in the BDF {bdf_path} has {num_spectral_point} "
            "(numSpectralPoint)"
        )


def _create_partition_antenna_xds(
    asdm: pyasdm.ASDM,
    partition_descr: dict,
    spw_id: int,
    polarization: xr.DataArray,
) -> xr.Dataset:
    """
    antenna_xds with the antennas of the partition, in the order of the BDF
    antenna slots (ConfigDescription.antennaId).
    """
    antenna_ids = _get_config_antenna_ids(_get_partition_config(asdm, partition_descr))
    return create_antenna_xds(
        asdm, len(antenna_ids), spw_id, polarization, antenna_id=antenna_ids
    )


def _select_phase_center_direction_by_time(
    field_and_source_xds: xr.Dataset, field_name: xr.DataArray
) -> xr.DataArray:
    """
    Phase center direction of every integration (from the field of every
    integration), dims (time, sky_dir_label), with the attributes (frame,
    units) of FIELD_PHASE_CENTER_DIRECTION.

    Raises
    ------
    RuntimeError
        If a field of the partition is missing from the field_and_source dataset.
    """
    phase_center_direction = field_and_source_xds["FIELD_PHASE_CENTER_DIRECTION"]
    names = np.asarray(field_name.values, dtype=str)
    available = set(map(str, phase_center_direction.coords["field_name"].values))
    missing = sorted(set(names) - available)
    if missing:
        raise RuntimeError(
            f"The fields {missing} of the partition are not in the field_and_source "
            f"dataset (field_name {sorted(available)})"
        )
    by_time = phase_center_direction.sel(
        field_name=xr.DataArray(names, dims="time")
    ).transpose("time", "sky_dir_label")
    # drop the field_name (and source_name, etc.) values along time
    by_time = by_time.reset_coords(drop=True).assign_coords(
        time=field_name.coords["time"].variable
    )
    by_time.attrs = dict(phase_center_direction.attrs)
    return by_time


def _create_uvw_data_var(
    correlated_xds_sizes: dict,
    time: xr.DataArray,
    baseline_antenna1_name: xr.DataArray,
    baseline_antenna2_name: xr.DataArray,
    antenna_position: xr.DataArray,
    field_phase_center_direction: xr.DataArray,
) -> dict:
    """
    UVW data variable (lazily computed).

    Parameters
    ----------
    correlated_xds_sizes : dict
        Sizes of the dimensions time, baseline_id and uvw_label.
    time : xr.DataArray
        time coordinate (with its format/scale measure attributes).
    baseline_antenna1_name : xr.DataArray
        Antenna name of the 1st antenna of every baseline.
    baseline_antenna2_name : xr.DataArray
        Antenna name of the 2nd antenna of every baseline.
    antenna_position : xr.DataArray
        ANTENNA_POSITION of the antenna_xds.
    field_phase_center_direction : xr.DataArray
        Phase center direction, dims (time, sky_dir_label) (one direction per
        time) or (field_name, sky_dir_label) with one field.

    Returns
    -------
    dict
        {"UVW": (dims, array, attrs)}
    """
    dims_uvw = ["time", "baseline_id", "uvw_label"]
    shape_uvw = (
        correlated_xds_sizes["time"],
        correlated_xds_sizes["baseline_id"],
        correlated_xds_sizes["uvw_label"],
    )
    uvw = (
        dims_uvw,
        xr.core.indexing.LazilyIndexedArray(
            asdm_backend_arrays.UVWArray(
                shape_uvw,
                time,
                baseline_antenna1_name,
                baseline_antenna2_name,
                antenna_position,
                field_phase_center_direction,
            )
        ),
        {"type": "uvw", "frame": "icrs", "units": "m"},
    )

    return {"UVW": uvw}


def _is_pointing_requested(
    partition_descr: dict,
    with_pointing: bool,
    pointing_for_only_spectral_resolution_types: list[str] | None,
) -> bool:
    """Whether the partition should get a pointing_xds."""
    if not with_pointing:
        return False
    if not pointing_for_only_spectral_resolution_types:
        return True
    spectral_types = np.atleast_1d(partition_descr.get("spectralType", []))
    return any(
        str(spectral_type) in pointing_for_only_spectral_resolution_types
        for spectral_type in spectral_types
    )


def _create_partition_pointing_xds(
    asdm: pyasdm.ASDM, correlated_xds: xr.Dataset, antenna_names
) -> xr.Dataset | None:
    """
    pointing_xds of a partition: the Pointing samples of its antennas within its
    time range (with a margin).

    Its antenna_name coordinate has the antennas of the partition, in the
    partition order (the same as in antenna_xds and, for single dish, the
    antenna_name index of the correlated dataset, which the pointing_xds node
    must align with in the DataTree). Antennas without samples in the time
    range have NaN values.

    Returns None (the partition is then opened without pointing_xds) when there
    are no such samples, or when the Pointing table cannot be converted
    (POINTING_CONVERSION_ERRORS: inconsistent rows, or polynomial pointing,
    which create_pointing_xds logs as an error once per ASDM), without failing
    the whole partition.
    """
    antenna_names = [str(name) for name in antenna_names]
    time_range = _pointing_time_range(correlated_xds)
    try:
        pointing_xds = create_pointing_xds(
            asdm, time_range=time_range, antenna_names=antenna_names
        )
    except POINTING_CONVERSION_ERRORS as exc:
        xradio_logger().info(
            "The partition will not have a pointing_xds, as the ASDM Pointing table "
            f"cannot be converted. {type(exc).__name__}: {exc}"
        )
        return None

    if pointing_xds is None:
        xradio_logger().info(
            f"No pointing samples found in the time range {time_range} of the "
            "partition, it will not have a pointing_xds"
        )
        return None
    return pointing_xds.reindex(antenna_name=antenna_names)


def _pointing_time_range(correlated_xds: xr.Dataset) -> tuple[float, float]:
    """
    Time range (seconds since the Unix epoch) covering the integrations of a
    partition, with a margin, to select its pointing samples.
    """
    time = correlated_xds.coords["time"]
    half_interval = 0.5 * float(time.attrs["integration_time"]["data"])
    durations = correlated_xds.data_vars.get("EFFECTIVE_INTEGRATION_TIME")
    if durations is not None and durations.size > 0:
        # the same for all the baselines/antennas of an integration (see
        # _create_time_vars): only the first one is read
        durations = durations.isel({dim: 0 for dim in durations.dims if dim != "time"})
        half_interval = max(half_interval, 0.5 * float(durations.max()))
    margin = half_interval + POINTING_TIME_RANGE_MARGIN
    return float(time.min()) - margin, float(time.max()) + margin


def _generate_baseline_antennax_id_as_in_bdf(
    num_antenna,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate antenna IDs for baselines following ASDM BDF ordering.

    This function generates pairs of antenna IDs that match the baseline ordering used in
    ALMA Science Data Model (ASDM) Binary Data Format (BDF) files. It creates pairs for all
    possible baseline combinations, including autocorrelations.

    The baseline ordering follows a lower triangular matrix pattern in row-major order,
    which effectively emulates the upper triangular matrix in column-major order used in BDFs.
    Autocorrelation pairs (antenna paired with itself) are appended at the end.

    Parameters
    ----------
    num_antenna : int
        Number of antennas in the array

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two 1D arrays containing the antenna IDs for each baseline:
        - First array contains the first antenna IDs of each baseline
        - Second array contains the second antenna IDs of each baseline
        The length of each array is num_baselines = (num_antenna * (num_antenna - 1))/2 + num_antenna,
        where the last term accounts for autocorrelations

    Notes
    -----
    The baseline pairs (antenna1, antenna2) are ordered as follows:
    1. Cross-correlations: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), ...
    2. Autocorrelations: (0,0), (1,1), (2,2), ...
    """

    antenna_ids = np.arange(num_antenna)

    antenna1_id_in_baselines, antenna2_id_in_baselines = np.meshgrid(
        antenna_ids, antenna_ids
    )

    # Trying lower matrix + row-major indexing to emulate the BDF upper matrix + column-major
    upper_matrix_indices = np.tril_indices(num_antenna, k=-1)
    antenna2_id_in_baselines = antenna2_id_in_baselines[upper_matrix_indices]
    antenna1_id_in_baselines = antenna1_id_in_baselines[upper_matrix_indices]

    auto_corr_indices = np.arange(num_antenna)
    baseline_antenna1_id = np.append(antenna1_id_in_baselines, auto_corr_indices)
    baseline_antenna2_id = np.append(antenna2_id_in_baselines, auto_corr_indices)

    return baseline_antenna1_id, baseline_antenna2_id
