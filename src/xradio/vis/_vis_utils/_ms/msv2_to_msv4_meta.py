import graphviper.utils.logger as logger

col_to_data_variable_names = {
    "FLOAT_DATA": "SPECTRUM",
    "DATA": "VISIBILITY",
    "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    "MODEL_DATA": "VISIBILITY_MODEL",
    "WEIGHT_SPECTRUM": "WEIGHT",
    "WEIGHT": "WEIGHT",
    "FLAG": "FLAG",
    "UVW": "UVW",
    "TIME_CENTROID": "TIME_CENTROID",
    "EXPOSURE": "EFFECTIVE_INTEGRATION_TIME",
}
col_dims = {
    "DATA": ("time", "baseline_id", "frequency", "polarization"),
    "CORRECTED_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "MODEL_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "frequency", "polarization"),
    "FLAG": ("time", "baseline_id", "frequency", "polarization"),
    "UVW": ("time", "baseline_id", "uvw_label"),
    "TIME_CENTROID": ("time", "baseline_id"),
    "EXPOSURE": ("time", "baseline_id"),
    "FLOAT_DATA": ("time", "baseline_id", "frequency", "polarization"),
}
col_to_coord_names = {
    "TIME": "time",
    "ANTENNA1": "baseline_ant1_id",
    "ANTENNA2": "baseline_ant2_id",
}

# Map casacore measures to astropy
casacore_to_msv4_measure_type = {
    "quanta": {
        "type": "quantity",
    },
    "direction": {"type": "sky_coord", "Ref": "frame", "Ref_map": {"J2000": "fk5"}},
    "epoch": {"type": "time", "Ref": "scale", "Ref_map": {"UTC": "utc"}},
    "frequency": {
        "type": "spectral_coord",
        "Ref": "frame",
        "Ref_map": {
            "REST": "REST",
            "LSRK": "LSRK",
            "LSRD": "LSRD",
            "BARY": "BARY",
            "GEO": "GEO",
            "TOPO": "TOPO",
            "GALACTO": "GALACTO",
            "LGROUP": "LGROUP",
            "CMB": "CMB",
            "Undefined": "Undefined",
        },
    },
    "position": {
        "type": "earth_location",
        "Ref": "ellipsoid",
        "Ref_map": {"ITRF": "GRS80"},
    },
    "uvw": {"type": "uvw", "Ref": "frame", "Ref_map": {"ITRF": "GRS80"}},
    "radialvelocity": {"type": "quantity"},
}

casa_frequency_frames = [
    "REST",
    "LSRK",
    "LSRD",
    "BARY",
    "GEO",
    "TOPO",
    "GALACTO",
    "LGROUP",
    "CMB",
    "Undefined",
]

casa_frequency_frames_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 64]


def column_description_casacore_to_msv4_measure(
    casacore_column_description, ref_code=None, time_format="UNIX"
):
    import numpy as np

    msv4_measure = {}
    if "MEASINFO" in casacore_column_description["keywords"]:
        measinfo = casacore_column_description["keywords"]["MEASINFO"]

        # Get conversion information
        msv4_measure_conversion = casacore_to_msv4_measure_type[measinfo["type"]]

        # Convert type, copy unit
        msv4_measure["type"] = msv4_measure_conversion["type"]
        msv4_measure["units"] = list(
            casacore_column_description["keywords"]["QuantumUnits"]
        )

        # Reference frame to convert?
        if "Ref" in msv4_measure_conversion:
            # Find reference frame
            if "TabRefCodes" in measinfo:
                ref_index = np.where(measinfo["TabRefCodes"] == ref_code)[0][0]
                casa_ref = measinfo["TabRefTypes"][ref_index]
            elif "Ref" in measinfo:
                casa_ref = measinfo["Ref"]
            elif measinfo["type"] == "frequency":
                # Some MSv2 don't have the "TabRefCodes".
                ref_index = np.where(casa_frequency_frames_codes == ref_code)[0][0]
                casa_ref = casa_frequency_frames[ref_index]
            else:
                logger.debug(
                    f"Could not determine {measinfo['type']} measure "
                    "reference frame!"
                )

            # Convert into MSv4 representation of reference frame, warn if unknown
            if casa_ref in msv4_measure_conversion.get("Ref_map", {}):
                casa_ref = msv4_measure_conversion["Ref_map"][casa_ref]
            else:
                logger.debug(
                    f"Unknown reference frame for {measinfo['type']} "
                    f"measure, using verbatim: {casa_ref}"
                )

            msv4_measure[msv4_measure_conversion["Ref"]] = casa_ref

        if msv4_measure["type"] == "time":
            msv4_measure["format"] = time_format
    elif "QuantumUnits" in casacore_column_description["keywords"]:
        msv4_measure = {
            "type": "quantity",
            "units": list(casacore_column_description["keywords"]["QuantumUnits"]),
        }

    return msv4_measure


def create_attribute_metadata(col, main_column_descriptions):
    attrs_metadata = column_description_casacore_to_msv4_measure(
        main_column_descriptions[col]
    )
    if col in ["DATA", "CORRECTED_DATA", "WEIGHT"]:
        if not attrs_metadata:
            attrs_metadata["type"] = "quanta"
            attrs_metadata["units"] = ["unkown"]

    return attrs_metadata
