import datetime
import importlib
import itertools

import numpy as np
import xarray as xr

import pyasdm

from xradio.measurement_set.schema import MSV4_SCHEMA_VERSION
from xradio.measurement_set._utils._asdm._utils.metadata_tables import (
    exp_asdm_table_to_df,
)
from xradio.measurement_set._utils._asdm._utils.spectral_window import (
    ensure_spw_name_conforms,
    get_chan_width,
    get_reference_frame,
    get_spw_frequency_centers,
)
from xradio._utils.dict_helpers import make_spectral_coord_reference_dict, make_quantity


def open_partition(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
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

    msv4_xdt.ds = open_correlated_xds(asdm, partition_descr)

    msv4_xdt["/antenna_xds"] = xr.Dataset(attrs={"type": "antenna"})

    # TODO:

    # gain_curve_xds

    # phase_calibration_xds

    # system_calibration_xds

    # weather_xds

    # pointing_xds

    # phased_array_xds

    # field_and_source_xds
    msv4_xdt["/field_and_source_base_xds"] = xr.Dataset(
        attrs={"type": "field_and_source"}
    )

    # info_dicts

    return msv4_xdt


def open_correlated_xds(
    asdm: pyasdm.ASDM, partition_descr: dict[str, np.ndarray]
) -> xr.DataTree:
    xds = xr.Dataset(
        attrs={
            "schema_version": MSV4_SCHEMA_VERSION,
            "creator": {
                "software_name": "xradio",
                "version": importlib.metadata.version("xradio"),
            },
            "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "visibility",
        }
    )

    coords, attrs = create_coordinates(asdm, partition_descr)
    xds = xds.assign_coords(coords)
    for coord_name in coords:
        if coord_name in attrs:
            xds.coords[coord_name].attrs = attrs[coord_name]

    return xds


def create_coordinates(
    asdm: pyasdm.ASDM,
    partition_descr: dict[str, np.ndarray],
    is_single_dish: bool = True,
) -> dict[str, np.ndarray]:
    """
    TODO

    Parameters
    ----------
    asdm:
        Input ASDM object
    partition_descr:
        description of partition IDs in a dictionary of "ID ASDM key/attribute" -> numeric IDs

    Returns
    -------
    dict
        coordinates dict ready to be added with 'assign_coords'
    """
    coords = {}
    attrs = {}

    # Closest to time is the Scan/startTime,endTime,etc. but time values will probably will be
    # read from the subscans BDFs?
    # This is for now very incomplete. Subscan/startTime,numIntegrations,etc.
    sdm_scan_attrs = ["execBlockId", "scanNumber", "startTime", "endTime", "numSubscan"]
    scan_df = exp_asdm_table_to_df(asdm, "Scan", sdm_scan_attrs)
    scans_metadata = scan_df.loc[
        scan_df["scanNumber"].isin(partition_descr["scanNumber"])
    ]
    time_centers = scans_metadata["startTime"].values
    coords["time"] = (["time"], time_centers)
    attrs["time"] = make_quantity(time_centers, "s", ["time"])
    scan_numbers = scans_metadata["scanNumber"]
    coords["scan_name"] = (["time"], scan_numbers)

    # baselines...
    sdm_main_attrs = ["time", "configDescriptionId", "fieldId", "numAntenna"]
    main_df = exp_asdm_table_to_df(asdm, "Main", sdm_main_attrs)
    configurations = main_df.loc[
        main_df["configDescriptionId"].isin(partition_descr["configDescriptionId"])
        & main_df["fieldId"].isin(partition_descr["fieldId"])
    ]
    num_antenna = configurations["numAntenna"].max()
    num_baselines = num_antenna * (num_antenna - 1) / 2
    # This might turn out too simplistic. We'll have to check how the baselines (auto-corrs and
    # cross-corrs) are read from the BDFs, and other factors.
    baseline_antenna1_id, baseline_antenna2_id = zip(
        *itertools.combinations_with_replacement(np.arange(num_antenna), 2)
    )
    coords["baseline_antenna1_id"] = (["baseline_id"], list(baseline_antenna1_id))
    coords["baseline_antenna2_id"] = (["baseline_id"], list(baseline_antenna2_id))
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
        "name",
        "numChan",
        "refFreq",
    ]
    # These are optional attrs of the ASDM table, better dealt with via util functions that
    # check for their presence and alternatives: "chanFreqStart", "chanFreqStep", "chanFreqArray",
    # "chanWidthArray", "effectiveBwArray", "measFreqRef".
    spw_df = exp_asdm_table_to_df(asdm, "SpectralWindow", sdm_spw_attrs)
    spectral_window = spw_df.loc[spw_df["spectralWindowId"] == spw_id]
    spw_name = spectral_window["name"].values
    num_chan = spectral_window["numChan"].values[0]
    frequency_centers = get_spw_frequency_centers(asdm, spw_id, num_chan)

    coords["frequency"] = frequency_centers
    attrs["frequency"] = make_quantity(frequency_centers, "Hz", ["frequency"])
    # Other keys of the frequency coord
    frequency_other_attrs = {
        "frame": get_reference_frame(asdm, spw_id),
        "spectral_window_name": ensure_spw_name_conforms(spw_name, spw_id),
        "reference_frequency": spectral_window["refFreq"].values[0],
        "channel_width": get_chan_width(asdm, spw_id),
    }
    attrs["frequency"].update(frequency_other_attrs)

    # field_name will be created from field_and_source_xds?
    sdm_field_attrs = ["fieldId", "fieldName"]
    field_df = exp_asdm_table_to_df(asdm, "Field", sdm_field_attrs)
    fields = field_df.loc[field_df["fieldId"].isin(partition_descr["fieldId"])][
        "fieldName"
    ].values
    coords["field_name"] = (["time"], np.repeat(fields, len(time_centers)))

    if not is_single_dish:
        coords["uvw_label"] = np.array(["u", "v", "w"])

    return coords, attrs
