import datetime
import importlib
import itertools

import dask
import numpy as np
import pandas as pd
import xarray as xr

import pyasdm

from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION
from xradio._utils.list_and_array import check_if_consistent
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
from xradio.measurement_set._utils._asdm._utils._bdf.load_time import (
    get_times_from_bdfs,
)
from xradio.measurement_set._utils._asdm._utils._bdf.robust_load_data_flags import (
    load_visibilities_from_partition_bdfs,
    load_flags_from_partition_bdfs,
)
from xradio.measurement_set._utils._asdm.create_antenna_xds import create_antenna_xds
from xradio.measurement_set._utils._asdm.create_field_and_source_xds import (
    create_field_and_source_xds,
)
from xradio.measurement_set._utils._asdm.create_info_dicts import create_info_dicts
from xradio._utils.dict_helpers import (
    make_quantity,
    make_quantity_attrs,
    make_spectral_coord_measure_attrs,
    make_spectral_coord_reference_dict,
    make_time_measure_attrs,
)


def open_partition(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
) -> xr.DataTree:
    """
    TODO: opens a partition as an MSv4 DataTre

    Parameters
    ----------
    asdm:
        Input ASDM object
    partition_descr:
        description of partition IDs in a dictionary of "ID ASDM key/attribute" -> numeric IDs

    Returns
    -------
    xr.DataTree
        Datatree with MSv4 populated from the ASDM partition
    """
    msv4_xdt = xr.DataTree()

    correlated_xds, num_antenna, spw_id = create_correlated_xds(asdm, partition_descr)
    msv4_xdt.ds = correlated_xds

    msv4_xdt["/antenna_xds"] = create_antenna_xds(
        asdm, num_antenna, spw_id, correlated_xds.polarization
    )

    # TODO:

    # gain_curve_xds

    # phase_calibration_xds

    # system_calibration_xds

    # weather_xds

    # pointing_xds

    # phased_array_xds

    # field_and_source_xds
    is_single_dish = False
    msv4_xdt["/field_and_source_base_xds"] = create_field_and_source_xds(
        asdm, partition_descr, spw_id, is_single_dish
    )

    # info_dicts

    return msv4_xdt


def create_correlated_xds(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> tuple[xr.DataTree, int]:
    """Create a correlated data xarray Dataset from ASDM data.
    This function creates an xarray Dataset containing correlated visibility data
    from an ASDM (ALMA Science Data Model) object. It sets up the necessary coordinates,
    data variables, and metadata attributes according to the MSv4 schema.
    Parameters
    ----------
    asdm : pyasdm.ASDM
        The ASDM object containing the raw data.
    partition_descr : dict[str, np.ndarray]
        Dictionary containing partition descriptions for the ASDM data.
    Returns
    -------
    tuple[xr.DataTree, int]
        A tuple containing:
        - xds : xr.Dataset
            The xarray Dataset containing the correlated visibility data with
            appropriate coordinates, variables, and metadata.
        - num_antenna : int
            The number of antennas in the dataset.
        - spw_id : int
            The spectral window ID.
    Notes
    -----
    The created Dataset follows the MSv4 schema and includes:
        - Basic metadata (schema version, creator info, creation date)
        - Coordinate systems
        - Time variables
        - Data variables for visibility data
        - Data group definitions
        - Additional metadata from the ASDM
    The function sets up a non-single-dish observation structure.
    """

    datetime_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
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

    is_single_dish = False
    coords, coord_attrs, num_antenna, spw_id, bdf_spw_id, time_vars = (
        create_coordinates(asdm, partition_descr, is_single_dish)
    )
    xds = xds.assign_coords(coords)
    for coord_name in coords:
        if coord_name in coord_attrs:
            xds.coords[coord_name].attrs = coord_attrs[coord_name]

    xds = xds.assign(time_vars)
    xds = xds.assign(create_data_vars(xds, partition_descr["BDFPath"], bdf_spw_id))

    data_group_base = {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "field_and_source": "field_and_source_base_xds",
        "description": "Base data group derived from data in ASDM BDFs",
        "date": datetime_now,
    }
    xds.attrs.update({"data_groups": {"base": data_group_base}})

    return xds, num_antenna, spw_id


def create_data_vars(
    xds: xr.Dataset, bdf_paths: list[str], bdf_spw_id: int
) -> dict[str, tuple]:
    """
    Create a dictionary of data variables for a radio astronomy dataset.
    This function initializes the fundamental data structures needed for radio interferometry
    data, including visibilities, weights, flags, and UVW coordinates.
    Parameters
    ----------
    xds : xr.Dataset
        Input xarray Dataset containing the dimension sizes for 'time', 'baseline_id',
        'frequency', 'polarization', and 'uvw_label'.
    bdf_paths : list[str]
        Paths to BDFs with data/flags for the partition
    bdf_spw_id : int
        Index of the SPW to load (index in the BDF, derived from the metadata SPW index)

    Returns
    -------
    dict[str, tuple]
        A dictionary containing the following data variables:
        - VISIBILITY : Complex visibility data with shape (time, baseline_id, frequency, polarization)
        - WEIGHT : Visibility weights with shape (time, baseline_id, frequency, polarization)
        - FLAG : Boolean flags with shape (time, baseline_id, frequency, polarization)
        - UVW : UVW coordinates with shape (time, baseline_id, uvw_label)

    Notes
    -----
    All arrays are initialized with ones. The actual data should be filled in later.
    VISIBILITY includes metadata for units and field/source information.
    UVW includes metadata specifying the coordinate frame (ICRS) and units (meters).
    """

    data_vars = {}

    dims_vis_weight_flag = ["time", "baseline_id", "frequency", "polarization"]
    shape_vis_weight_flag = (
        xds.sizes["time"],
        xds.sizes["baseline_id"],
        xds.sizes["frequency"],
        xds.sizes["polarization"],
    )
    data_vars["VISIBILITY"] = (
        dims_vis_weight_flag,
        dask.array.from_delayed(
            dask.delayed(load_visibilities_from_partition_bdfs)(
                bdf_paths, bdf_spw_id, {}
            ),
            shape=shape_vis_weight_flag,
            dtype="complex128",
        ),
        {
            "type": "quantity",
            "units": "",  # Do the ASDM/BDFs give anything?
            "field_and_source_xds": None,
        },
    )

    data_vars["WEIGHT"] = (
        dims_vis_weight_flag,
        dask.array.from_delayed(
            dask.delayed(produce_weight_data_var)(xds),
            shape=shape_vis_weight_flag,
            dtype="float64",
        ),
    )

    data_vars["FLAG"] = (
        dims_vis_weight_flag,
        dask.array.from_delayed(
            dask.delayed(load_flags_from_partition_bdfs)(bdf_paths, bdf_spw_id, {}),
            shape=shape_vis_weight_flag,
            dtype="bool",
        ),
    )

    dims_uvw = ["time", "baseline_id", "uvw_label"]
    shape_uvw = (
        xds.sizes["time"],
        xds.sizes["baseline_id"],
        xds.sizes["uvw_label"],
    )
    data_vars["UVW"] = (
        dims_uvw,
        dask.array.from_delayed(
            dask.delayed(produce_uvw_data_var)(xds), shape=shape_uvw, dtype="float64"
        ),
        {"type": "uvw", "frame": "icrs", "units": "m"},
    )

    return data_vars


def create_coordinates(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
    is_single_dish: bool = False,
) -> tuple[dict, dict, int, int, dict]:
    """
    Create coordinate systems and associated metadata from ASDM data.

    This function extracts and processes coordinate information from an ALMA Science Data Model
    (ASDM) dataset, including time, frequency, polarization, baseline, and field coordinates.
    It handles both interferometric and single-dish observations.

    Parameters
    ----------
    asdm : pyasdm.ASDM
        Input ASDM object containing the observation data
    partition_descr : dict[str, np.ndarray]
        Dictionary mapping ASDM keys/attributes to their corresponding numeric IDs.
        Expected keys include 'scanNumber', 'BDFPath', 'configDescriptionId',
        'fieldId', and 'dataDescriptionId'
    is_single_dish : bool, optional
        Flag indicating if the data is from single-dish observations. If False,
        UVW coordinates will be included. Default is False.

    Returns
    -------
    coords : dict
        Dictionary of coordinate arrays and their dimensions, ready for xarray
        dataset creation. Includes coordinates for time, scan, baseline,
        polarization, frequency, and field name.
    attrs : dict
        Dictionary of coordinate attributes including units, reference frames,
        and other metadata.
    num_antenna : int
        Number of antennas in the observation.
    spw_id : int
        Spectral window ID.
    time_vars : dict
        Dictionary containing time-related variables including effective
        integration time and time centroid information.

    Notes
    -----
    The function processes various ASDM tables including Scan, Main, DataDescription,
    Polarization, SpectralWindow, and Field to create a complete coordinate system
    suitable for radio astronomy data analysis.
    """

    # Closest to time is the Scan/startTime,endTime,etc. but time values will probably will be
    # read from the subscans BDFs?
    # This is for now very incomplete. Subscan/startTime,numIntegrations,etc.
    sdm_scan_attrs = ["execBlockId", "scanNumber", "startTime", "endTime", "numSubscan"]
    scan_df = exp_asdm_table_to_df(asdm, "Scan", sdm_scan_attrs)
    scans_metadata = scan_df.loc[
        scan_df["scanNumber"].isin(partition_descr["scanNumber"])
    ]

    time_centers, durations, actual_times, actual_durations = get_times_from_bdfs(
        partition_descr["BDFPath"], scans_metadata
    )
    integration_time = check_if_consistent(np.array(durations), "durations")
    coords = {}
    attrs = {}
    coords["time"] = (["time"], time_centers)
    attrs["time"] = make_time_measure_attrs("s", "tai", time_format="unix")
    attrs["time"].update(
        {"dims": ["time"], "integration_time": make_quantity(integration_time, "s")}
    )
    scan_numbers = np.array(scans_metadata["scanNumber"]).astype(str)
    # TODO: proper mapping begin/end scans, subscans -> BDFs
    if len(scan_numbers) != len(time_centers):
        scan_numbers = np.resize(scan_numbers, len(time_centers))
    coords["scan_name"] = (["time"], scan_numbers)
    attrs["scan_name"] = {"scan_intents": partition_descr["scanIntent"]}

    # baselines...
    sdm_main_attrs = ["time", "configDescriptionId", "fieldId", "numAntenna"]
    main_df = exp_asdm_table_to_df(asdm, "Main", sdm_main_attrs)
    configurations = main_df.loc[
        main_df["configDescriptionId"].isin(partition_descr["configDescriptionId"])
        & main_df["fieldId"].isin(partition_descr["fieldId"])
    ]
    num_antenna = configurations["numAntenna"].max()

    baseline_antenna1_id, baseline_antenna2_id = (
        generate_baseline_antennax_id_as_in_bdf(num_antenna)
    )

    # TODO: get proper names
    coords["baseline_antenna1_name"] = (
        ["baseline_id"],
        list(["antenna_" + str(idx) for idx in baseline_antenna1_id]),
    )
    coords["baseline_antenna2_name"] = (
        ["baseline_id"],
        list(["antenna_" + str(idx) for idx in baseline_antenna2_id]),
    )
    coords["baseline_id"] = np.arange(len(baseline_antenna1_id))

    # From dataDescriptionId get SPW and polarization IDs
    dd_id = partition_descr["dataDescriptionId"][0]
    sdm_dd_attrs = ["dataDescriptionId", "spectralWindowId", "polOrHoloId"]
    data_description_df = exp_asdm_table_to_df(asdm, "DataDescription", sdm_dd_attrs)
    data_description = data_description_df.loc[
        data_description_df["dataDescriptionId"] == dd_id
    ]
    spw_id = data_description["spectralWindowId"].values[0]
    pol_setup_id = data_description["polOrHoloId"].values[0]

    # polarization coord
    sdm_polarization_attrs = ["polarizationId", "numCorr", "corrType"]
    polarization_df = exp_asdm_table_to_df(asdm, "Polarization", sdm_polarization_attrs)
    polarization_metadata = polarization_df.loc[
        polarization_df["polarizationId"] == pol_setup_id
    ]
    num_corr = polarization_metadata["numCorr"].values[0]
    polarization_setup = polarization_metadata["corrType"].values[0][:num_corr]
    coords["polarization"] = polarization_setup

    # frequency coord
    sdm_spw_attrs = [
        "spectralWindowId",
        "numChan",
        "refFreq",
    ]
    # These are optional attrs of the ASDM table, better dealt with via util functions that
    # check for their presence and alternatives: "chanFreqStart", "chanFreqStep", "chanFreqArray",
    # "chanWidthArray", "effectiveBwArray", "measFreqRef".
    spw_df = exp_asdm_table_to_df(asdm, "SpectralWindow", sdm_spw_attrs)
    spectral_window = spw_df.loc[spw_df["spectralWindowId"] == spw_id]
    spw_name = get_spw_name(asdm, spw_id)
    num_chan = spectral_window["numChan"].values[0]
    frequency_centers = get_spw_frequency_centers(asdm, spw_id, num_chan)

    coords["frequency"] = (["frequency"], [freq for freq in frequency_centers])
    attrs["frequency"] = make_spectral_coord_measure_attrs("Hz", observer="TOPO")
    # Other keys of the frequency coord
    frequency_other_attrs = {
        "frame": get_reference_frame(asdm, spw_id),
        "spectral_window_name": ensure_spw_name_conforms(spw_name, spw_id),
        "spectral_window_intents": ["UNSPECIFIED"],
        "reference_frequency": make_spectral_coord_reference_dict(
            spectral_window["refFreq"].values[0], "Hz", "TOPO"
        ),
        "channel_width": make_quantity(get_chan_width(asdm, spw_id), "Hz"),
    }
    attrs["frequency"].update(frequency_other_attrs)

    # field_name will be created from field_and_source_xds?
    sdm_field_attrs = ["fieldId", "fieldName"]
    field_df = exp_asdm_table_to_df(asdm, "Field", sdm_field_attrs)
    fields = field_df.loc[field_df["fieldId"].isin(partition_descr["fieldId"])][
        "fieldName"
    ].values.astype(str)
    coords["field_name"] = (["time"], np.resize(fields, len(time_centers)))

    # TODO This redim should be done inside ._bdf/load_time
    # We need (time, baseline_id) dims but times and durations form ASDM/BDFs are independent of baseline
    redim_actual_durations = np.resize(
        actual_durations, (len(actual_durations), len(baseline_antenna1_id))
    )
    redim_actual_times = np.resize(
        actual_times, (len(actual_times), len(baseline_antenna1_id))
    )
    time_vars = {
        "EFFECTIVE_INTEGRATION_TIME": (
            ["time", "baseline_id"],
            redim_actual_durations,
            make_quantity_attrs("s"),
        ),
        "TIME_CENTROID": (
            ["time", "baseline_id"],
            redim_actual_times,
            make_time_measure_attrs("s", "tai", time_format="unix"),
        ),
    }

    if not is_single_dish:
        coords["uvw_label"] = np.array(["u", "v", "w"])

    sdm_config_description_attrs = [
        "configDescriptionId",
        "dataDescriptionId",
    ]
    config_description_df = exp_asdm_table_to_df(
        asdm, "ConfigDescription", sdm_config_description_attrs
    )
    bdf_spw_id = translate_asdm_tables_spw_id_to_bdf_spw_id(
        spw_id,
        data_description_df,
        spw_df,
        partition_descr["configDescriptionId"],
        config_description_df,
    )

    # TODO: this needs clean-up!
    return coords, attrs, num_antenna, spw_id, bdf_spw_id, time_vars


def produce_uvw_data_var(xds: xr.Dataset) -> xr.DataArray:
    # TODO: best guess is to try to reproduce sdm tool behavior?
    return np.ones(
        (
            xds.sizes["time"],
            xds.sizes["baseline_id"],
            xds.sizes["uvw_label"],
        ),
        dtype="float64",
    )


def produce_weight_data_var(xds: xr.Dataset) -> xr.DataArray:
    return np.ones(
        (
            xds.sizes["time"],
            xds.sizes["baseline_id"],
            xds.sizes["frequency"],
            xds.sizes["polarization"],
        ),
        dtype="float64",
    )


def translate_asdm_tables_spw_id_to_bdf_spw_id(
    spw_id: int,
    data_description_df: pd.DataFrame,
    spw_df: pd.DataFrame,
    # asdm: pyasdm.ASDM,
    config_description_id: int,
    config_description_df: pd.DataFrame,
) -> int:

    this_config_df = config_description_df.loc[
        config_description_df["configDescriptionId"].isin(config_description_id)
    ]
    data_description_ids = this_config_df["dataDescriptionId"].values[0]

    this_data_description_df = data_description_df.loc[
        data_description_df["dataDescriptionId"].isin(data_description_ids)
    ]
    spw_ids_in_data_descriptions = this_data_description_df["spectralWindowId"].values

    bdf_spw_id = int(np.argmax(spw_ids_in_data_descriptions == spw_id))
    return bdf_spw_id


def generate_baseline_antennax_id_as_in_bdf(
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
    The baseline pairs are ordered as follows:
    1. Cross-correlations in lower triangular order: (1,0), (2,0), (2,1), (3,0), ...
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


def generate_baseline_antennax_id_as_in_msv2(
    num_antenna,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate antenna ID pairs for all baselines in a measurement set.

    This function creates pairs of antenna IDs that form baselines in a radio
    interferometer array, including both auto-correlations and cross-correlations,
    following the MS v2 convention.

    Parameters
    ----------
    num_antenna : int
        Total number of antennas in the array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - baseline_antenna1_id: Array of first antenna IDs for each baseline
        - baseline_antenna2_id: Array of second antenna IDs for each baseline
        Both arrays have the same length and correspond to baseline pairs.

    Notes
    -----
    The function uses combinations_with_replacement to generate all possible antenna
    pairs, including auto-correlations (when both antennas are the same).
    The order follows the MS v2 convention for baseline organization.

    Examples
    --------
    >>> generate_baseline_antennax_id_as_in_msv2(3)
    (array([0, 0, 0, 1, 1, 2]), array([0, 1, 2, 1, 2, 2]))
    """

    # num_baselines = num_antenna * (num_antenna - 1) / 2
    # This might turn out too simplistic. We'll have to check how the baselines (auto-corrs and
    # cross-corrs) are read from the BDFs, and other factors.
    baseline_antenna1_id, baseline_antenna2_id = zip(
        *itertools.combinations_with_replacement(np.arange(num_antenna), 2)
    )

    return baseline_antenna1_id, baseline_antenna2_id
