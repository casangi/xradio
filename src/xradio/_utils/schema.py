import toolviper.utils.logger as logger
import xarray as xr


def convert_generic_xds_to_xradio_schema(
    generic_xds: xr.Dataset,
    msv4_xds: xr.Dataset,
    to_new_data_variables: dict[str, list],
    to_new_coords: dict[str, list],
) -> xr.Dataset:
    """Converts a generic xarray Dataset to the xradio schema.

    This function takes a generic xarray Dataset and converts it to an xradio schema
    represented by the msv4_xds Dataset. It performs the conversion based on the provided
    mappings in the to_new_data_variables and to_new_coords dictionaries.

    Parameters
    ----------
    generic_xds : xr.Dataset
        The generic xarray Dataset to be converted.
    msv4_xds : xr.Dataset
        The xradio schema represented by the msv4_xds Dataset.
    to_new_data_variables : dict
        A dictionary mapping the data variables/coordinates in the generic_xds Dataset to the new data variables
        in the msv4_xds Dataset. The keys are the old data variables/coordinates and the values are a list of the new name and a list of the new dimension names.
    to_new_coords : dict
        A dictionary mapping  data variables/coordinates in the generic_xds Dataset to the new coordinates
        in the msv4_xds Dataset. The keys are the old data variables/coordinates and the values are a list of the new name and a list of the new dimension names.

    Returns
    -------
    xr.Dataset
        The converted xradio schema represented by the msv4_xds Dataset.

    Notes
    -----
    Example to_new_data_variables:
    to_new_data_variables = {
        "POSITION": ["ANTENNA_POSITION",["name", "cartesian_pos_label"]],
        "OFFSET": ["ANTENNA_FEED_OFFSET",["name", "cartesian_pos_label"]],
        "DISH_DIAMETER": ["ANTENNA_DISH_DIAMETER",["name"]],
    }

    Example to_new_coords:
    to_new_coords = {
        "NAME": ["name",["name"]],
        "STATION": ["station",["name"]],
        "MOUNT": ["mount",["name"]],
        "PHASED_ARRAY_ID": ["phased_array_id",["name"]],
        "antenna_id": ["antenna_id",["name"]],
    }
    """

    column_description = generic_xds.attrs["other"]["msv2"]["ctds_attrs"][
        "column_descriptions"
    ]
    coords = {}
    coord_attrs = {}

    name_keys = list(generic_xds.data_vars.keys()) + list(generic_xds.coords.keys())

    for key in name_keys:

        if key in column_description:
            msv4_measure = column_description_casacore_to_msv4_measure(
                column_description[key]
            )
        else:
            msv4_measure = None

        if key in to_new_data_variables:
            new_dv = to_new_data_variables[key]
            msv4_xds[new_dv[0]] = xr.DataArray(generic_xds[key].data, dims=new_dv[1])

            if msv4_measure:
                msv4_xds[new_dv[0]].attrs.update(msv4_measure)

        if key in to_new_coords:
            new_coord = to_new_coords[key]
            coords[new_coord[0]] = (
                new_coord[1],
                generic_xds[key].data,
            )

            if msv4_measure:
                coord_attrs[new_coord[0]] = msv4_measure

    msv4_xds = msv4_xds.assign_coords(coords)
    for coord, coord_attrs in coord_attrs.items():
        msv4_xds.coords[coord].attrs.update(coord_attrs)

    return msv4_xds


def column_description_casacore_to_msv4_measure(
    casacore_column_description, ref_code=None, time_format="unix"
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

        # Beware: casa_ref won't be found in cases such as the custom
        # 'NRAO_GBT_USER/NRAO_GBT_USER_DIR_REF' in POINTING
        casa_ref = None
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
                    f"No translation defined for casacore reference frame ({measinfo['type']} "
                    f"measure), using verbatim: {casa_ref}"
                )
                if measinfo["type"] == "direction" and casa_ref == "B1950_VLA":
                    raise RuntimeError(
                        f"The direction reference frame {casa_ref} is not supported. "
                        "Please use CASA to convert the frame to supported frames like J2000 or ICRS."
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


# Map casacore measures to astropy
casacore_to_msv4_measure_type = {
    "quanta": {
        "type": "quantity",
    },
    "direction": {
        "type": "sky_coord",
        "Ref": "frame",
        "Ref_map": {
            "AZELGEO": "altaz",
            "ICRS": "icrs",
            "J2000": "fk5",
        },
    },
    "epoch": {
        "type": "time",
        "Ref": "scale",
        "Ref_map": {
            "UTC": "utc",
            "TAI": "tai",
        },
    },
    "frequency": {
        "type": "spectral_coord",
        "Ref": "observer",
        "Ref_map": {
            "REST": "REST",
            "LSRK": "lsrk",
            "LSRD": "lsrd",
            "BARY": "BARY",
            "GEO": "GEO",
            "TOPO": "TOPO",
        },  # The frames/observer we are not sure if/how to translate to astropy are uppercase
    },
    "position": {
        "type": "location",
        "Ref": "frame",
        "Ref_map": {"ITRF": "GRS80"},
    },
    "uvw": {
        "type": "uvw",
        "Ref": "frame",
        "Ref_map": {"J2000": "fk5", "APP": "APP"},
    },
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
