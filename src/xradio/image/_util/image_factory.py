from astropy.wcs import WCS
from collections import OrderedDict
import numpy as np
import xarray as xr
from typing import List, Union
from .common import _c, _compute_world_sph_dims, _l_m_attr_notes
import toolviper.utils.logger as logger
from xradio._utils.coord_math import _deg_to_rad
from xradio._utils.dict_helpers import (
    make_direction_location_dict,
    make_spectral_coord_reference_dict,
    make_quantity,
    make_skycoord_dict,
    make_time_coord_attrs,
)


def _input_checks(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
) -> None:
    """
    Validate input parameters for image creation functions.

    Parameters
    ----------
    phase_center : list or np.ndarray
        Image phase center coordinates. Must have exactly 2 elements.
    image_size : list or np.ndarray
        Number of pixels along each axis. Must have exactly 2 elements.
    cell_size : list or np.ndarray
        Size of pixels along each axis. Must have exactly 2 elements.

    Raises
    ------
    ValueError
        If any parameter does not have exactly 2 elements.
    """
    if len(image_size) != 2:
        raise ValueError("image_size must have exactly two elements")
    if len(phase_center) != 2:
        raise ValueError("phase_center must have exactly two elements")
    if len(cell_size) != 2:
        raise ValueError("cell_size must have exactly two elements")


def _make_coords(
    frequency_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
) -> dict:
    if not isinstance(frequency_coords, list) and not isinstance(
        frequency_coords, np.ndarray
    ):
        frequency_coords = [frequency_coords]
    frequency_coords = np.array(frequency_coords, dtype=np.float64)
    restfreq = frequency_coords[len(frequency_coords) // 2]
    vel_coords = (1 - frequency_coords / restfreq) * _c.to("m/s").value
    if not isinstance(time_coords, list) and not isinstance(time_coords, np.ndarray):
        time_coords = [time_coords]
    time_coords = np.array(time_coords, dtype=np.float64)
    return dict(
        chan=frequency_coords, vel=vel_coords, time=time_coords, restfreq=restfreq
    )


def _add_common_attrs(
    xds: xr.Dataset,
    restfreq: float,
    spectral_reference: str,
    direction_reference: str,
    phase_center: Union[List[float], np.ndarray],
    cell_size: Union[List[float], np.ndarray],
    projection: str,
) -> xr.Dataset:
    xds.time.attrs = make_time_coord_attrs(units="d", scale="utc", time_format="mjd")
    freq_vals = np.array(xds.frequency)
    xds.frequency.attrs = {
        "observer": spectral_reference.lower(),
        "reference_frequency": make_spectral_coord_reference_dict(
            value=freq_vals[len(freq_vals) // 2].item(),
            units="Hz",
            observer=spectral_reference.lower(),
        ),
        "rest_frequencies": make_quantity(restfreq, "Hz"),
        "rest_frequency": make_quantity(restfreq, "Hz"),
        "type": "spectral_coord",
        "units": "Hz",
        "wave_units": "mm",
    }
    xds.velocity.attrs = {"doppler_type": "radio", "type": "doppler", "units": "m/s"}
    reference = make_skycoord_dict(
        data=phase_center, units="rad", frame=direction_reference
    )
    reference["attrs"].update({"equinox": "j2000.0"})
    xds.attrs = {
        "data_groups": {"base": {}},
        "coordinate_system_info": {
            "reference_direction": reference,
            "native_pole_direction": make_direction_location_dict(
                [np.pi, 0.0], "rad", "native_projection"
            ),
            "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "projection": projection,
            "projection_parameters": [0.0, 0.0],
        },
        "type": "image",
    }
    return xds


def _make_common_coords(
    pol_coords: Union[list, np.ndarray],
    frequency_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
) -> dict:
    some_coords = _make_coords(frequency_coords, time_coords)
    return {
        "coords": {
            "time": some_coords["time"],
            "frequency": some_coords["chan"],
            "velocity": ("frequency", some_coords["vel"]),
            "polarization": pol_coords,
        },
        "restfreq": some_coords["restfreq"],
    }


def _make_lm_values(
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
) -> dict:
    # l follows RA as far as increasing/decreasing, see AIPS Meme 27, change in alpha
    # definition three lines below Figure 2 and the first of the pair of equations 10.
    l = [
        (i - image_size[0] // 2) * (-1) * abs(cell_size[0])
        for i in range(image_size[0])
    ]
    m = [(i - image_size[1] // 2) * abs(cell_size[1]) for i in range(image_size[1])]
    return {"l": l, "m": m}


def _make_sky_coords(
    projection: str,
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
    phase_center: Union[list, np.ndarray],
) -> dict:
    long, lat = _compute_world_sph_dims(
        projection=projection,
        shape=image_size,
        ctype=["RA", "Dec"],
        crpix=[image_size[0] // 2, image_size[1] // 2],
        crval=phase_center,
        cdelt=[-abs(cell_size[0]), abs(cell_size[1])],
        cunit=["rad", "rad"],
    )["value"]
    return {"right_ascension": (("l", "m"), long), "declination": (("l", "m"), lat)}


def _add_lm_coord_attrs(xds: xr.Dataset) -> xr.Dataset:
    attr_note = _l_m_attr_notes()
    xds.l.attrs = {
        "note": attr_note["l"],
    }
    xds.m.attrs = {
        "note": attr_note["m"],
    }


def _make_empty_sky_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    cell_size: Union[list, np.ndarray],
    frequency_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    _input_checks(phase_center, image_size, cell_size)
    cc = _make_common_coords(pol_coords, frequency_coords, time_coords)
    coords = cc["coords"]
    lm_values = _make_lm_values(image_size, cell_size)
    coords.update(lm_values)
    if do_sky_coords:
        coords.update(_make_sky_coords(projection, image_size, cell_size, phase_center))
    xds = xr.Dataset(coords=coords)
    xds = _move_beam_param_dim_coord(xds)
    _add_lm_coord_attrs(xds)
    _add_common_attrs(
        xds,
        cc["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        cell_size,
        projection,
    )
    return xds


def _make_uv_coords(
    xds: xr.Dataset,
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
) -> dict:
    uv_values = _make_uv_values(image_size, sky_image_cell_size)
    xds = xds.assign_coords(uv_values)
    attr = make_quantity(0.0, "lambda")
    xds.u.attrs = attr.copy()
    xds.v.attrs = attr.copy()
    return xds


def _make_uv_values(
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
) -> dict:
    im_size_wave = 1 / np.array(sky_image_cell_size)
    uv_cell_size = im_size_wave / np.array(image_size)
    u_vals = [
        (i - image_size[0] // 2) * abs(uv_cell_size[0]) for i in range(image_size[0])
    ]
    v_vals = [
        (i - image_size[1] // 2) * abs(uv_cell_size[1]) for i in range(image_size[1])
    ]
    return {"u": u_vals, "v": v_vals}


def _make_empty_aperture_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
    frequency_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
) -> xr.Dataset:
    _input_checks(phase_center, image_size, sky_image_cell_size)
    cc = _make_common_coords(pol_coords, frequency_coords, time_coords)
    coords = cc["coords"]
    xds = xr.Dataset(coords=coords)
    xds = _make_uv_coords(xds, image_size, sky_image_cell_size)
    _add_common_attrs(
        xds,
        cc["restfreq"],
        spectral_reference,
        direction_reference,
        phase_center,
        sky_image_cell_size,
        projection,
    )
    xds = _move_beam_param_dim_coord(xds)
    return xds


def _move_beam_param_dim_coord(xds: xr.Dataset) -> xr.Dataset:
    """
    Add beam_params_label coordinate to an xarray Dataset.

    Parameters
    ----------
    xds : xr.Dataset
        Input Dataset to which beam parameters will be added.

    Returns
    -------
    xr.Dataset
        Dataset with beam_params_label coordinate containing ['major', 'minor', 'pa'].
    """
    return xds.assign_coords(
        beam_params_label=("beam_params_label", ["major", "minor", "pa"])
    )


def _make_empty_lmuv_image(
    phase_center: Union[list, np.ndarray],
    image_size: Union[list, np.ndarray],
    sky_image_cell_size: Union[list, np.ndarray],
    frequency_coords: Union[list, np.ndarray],
    pol_coords: Union[list, np.ndarray],
    time_coords: Union[list, np.ndarray],
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    xds = _make_empty_sky_image(
        phase_center,
        image_size,
        sky_image_cell_size,
        frequency_coords,
        pol_coords,
        time_coords,
        direction_reference,
        projection,
        spectral_reference,
        do_sky_coords,
    )
    xds = _make_uv_coords(xds, image_size, sky_image_cell_size)
    xds = _move_beam_param_dim_coord(xds)
    return xds


def detect_store_type(store):
    """
    Detect the storage format type of an image store.

    Parameters
    ----------
    store : str or dict
        Path to the image store or a dictionary representation.

    Returns
    -------
    str
        The detected store type: 'fits', 'casa', or 'zarr'.

    Raises
    ------
    ValueError
        If the directory structure is unknown or the path does not exist.
    """
    import os

    if isinstance(store, str):
        if os.path.isfile(store):
            store_type = "fits"
        elif os.path.isdir(store):
            if "table.info" in os.listdir(store):
                store_type = "casa"
            elif ".zattrs" in os.listdir(store):
                store_type = "zarr"
            else:
                logger.error("Unknown directory structure.")
                raise ValueError("Unknown directory structure." + str(store))
        else:
            logger.error("Path does not exist.")
            raise ValueError("Path does not exist." + str(store))
    else:
        store_type = "zarr"

    return store_type


def detect_image_type(store):
    """
    Detect the image type from the store name or path.

    Infers the image type based on common naming patterns in the store path.

    Parameters
    ----------
    store : str or other
        Path to the image store. If not a string, returns 'ALL'.

    Returns
    -------
    str
        The detected image type. Possible values include:
        - 'SKY': Sky image (contains 'image' or 'im')
        - 'POINT_SPREAD_FUNCTION': PSF image (contains 'psf')
        - 'MODEL': Model image (contains 'model')
        - 'RESIDUAL': Residual image (contains 'residual')
        - 'MASK': Mask image (contains 'mask')
        - 'PRIMARY_BEAM': Primary beam image (contains 'pb')
        - 'APERTURE': Aperture image (contains 'aperture')
        - 'VISIBILITY': Visibility image (contains 'visibility')
        - 'VISIBILITY_NORMALIZATION': Visibility normalization (contains 'sumwt')
        - 'UNKNOWN': Could not detect type from name
        - 'ALL': Non-string store type
    """
    import os

    if isinstance(store, str):
        if "image" in store.lower():
            image_type = "SKY"
        elif "im" in store.lower():
            image_type = "SKY"
        elif "psf" in store.lower():
            image_type = "POINT_SPREAD_FUNCTION"
        elif "model" in store.lower():
            image_type = "MODEL"
        elif "residual" in store.lower():
            image_type = "RESIDUAL"
        elif "mask" in store.lower():
            image_type = "MASK"
        elif "pb" in store.lower():
            image_type = "PRIMARY_BEAM"
        elif "aperture" in store.lower():
            image_type = "APERTURE"
        elif "visibility" in store.lower():
            image_type = "VISIBILITY"
        elif "sumwt" in store.lower():
            image_type = "VISIBILITY_NORMALIZATION"
        else:
            image_type = "UNKNOWN"
    else:
        image_type = "ALL"

    return image_type


def create_store_dict(store_to_label):
    """
    Create a standardized dictionary mapping image types to their store information.

    Converts various input formats (string, list, or dict) into a consistent
    dictionary format where keys are image types and values contain store metadata.

    Parameters
    ----------
    store_to_label : str, list, or dict
        Input store specification. Can be:
        - str: Single store path
        - list: List of store paths (image types will be auto-detected)
        - dict: Mapping of image types to store paths

    Returns
    -------
    dict
        Dictionary with image types as keys. Each value is a dict with:
        - 'store_type': str, the format ('casa', 'fits', or 'zarr')
        - 'store': str, the path to the store

    Raises
    ------
    ValueError
        If image type cannot be detected or duplicate image types are found.
    """
    store_list = None
    if isinstance(store_to_label, str):
        store_list = [store_to_label]  # So can iterate over it.
    elif isinstance(store_to_label, list):
        store_list = store_to_label

    if (store_list is not None) and isinstance(store_list, list):
        store_dict_to_label = {i: v for i, v in enumerate(store_list)}
    else:
        store_dict_to_label = store_to_label

    store_dict = {}

    for image_type, store in store_dict_to_label.items():

        if isinstance(image_type, int):
            image_type = detect_image_type(store)

        image_type = image_type.upper()

        store_type = detect_store_type(store)

        if image_type == "UNKNOWN":
            logger.error(f"Could not detect image type for store {store}. ")
            example = "store={'sky': 'path/to/image.fits'}"
            raise ValueError(
                f"Could not detect image type for store {store}. Please label the store with the image type explicitly. For example: {example}"
            )

        if image_type in store_dict:
            logger.error(f"Duplicate image type {image_type} detected in store list.")
            raise ValueError(
                f"Duplicate image type {image_type} detected in store list. Please ensure each image type is unique. The store dict"
                + str(store_dict)
            )

        if store_type == "zarr":
            image_type = "ALL"  # Zarr can have multiple data variables.

        store_dict[image_type] = {"store_type": store_type, "store": store}

    return store_dict


def create_image_xds_from_store(
    store: Union[list, dict, str],
    access_store_casa: callable,
    casa_kwargs: dict,
    access_store_fits: callable,
    fits_kwargs: dict,
    access_store_zarr: callable,
    zarr_kwargs: dict,
) -> xr.Dataset:
    """
    Create an xarray Dataset from one or more image stores.

    This function reads image data from CASA, FITS, or zarr format stores and
    combines them into a single xarray Dataset with appropriate metadata and
    data variables.

    Parameters
    ----------
    store : str, list, or dict
        Image store specification:
        - str: Single store path
        - list: List of store paths
        - dict: Mapping of image types to store paths
    access_store_casa : callable
        Function to read CASA format images. Should accept a store path and
        keyword arguments, returning an xr.Dataset.
    casa_kwargs : dict
        Keyword arguments to pass to access_store_casa.
    access_store_fits : callable or None
        Function to read FITS format images. Should accept a store path and
        keyword arguments, returning an xr.Dataset. Can be None if FITS support
        is not needed.
    fits_kwargs : dict
        Keyword arguments to pass to access_store_fits.
    access_store_zarr : callable
        Function to read zarr format images. Should accept a store path and
        keyword arguments, returning an xr.Dataset.
    zarr_kwargs : dict
        Keyword arguments to pass to access_store_zarr.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the image data and metadata. The Dataset
        includes:
        - Data variables for each image type (e.g., 'SKY', 'MODEL', 'RESIDUAL')
        - Coordinates shared across all images
        - Attributes including 'type' and 'data_groups'

    Raises
    ------
    ValueError
        If zarr store with multiple data variables is combined with other stores.
    RuntimeError
        If FITS format is requested but access_store_fits is None, or if an
        unrecognized image format is encountered.

    Notes
    -----
    - Zarr stores can contain multiple data variables and will be returned as-is.
    - For other formats, data from multiple stores is combined into one Dataset.
    - BEAM_FIT_PARAMS from SKY images take precedence over POINT_SPREAD_FUNCTION.
    - Masks are renamed to MASK_<IMAGE_TYPE> for internal masks.
    """
    store_dict = create_store_dict(store)

    if "ALL" in store_dict and len(store_dict) > 1:
        logger.error(
            "When using a zarr store with multiple data variables, no other stores can be specified."
        )
        raise ValueError(
            "When using a zarr store with multiple data variables, no other stores can be specified."
        )

    if "ALL" in store_dict:
        img_xds = access_store_zarr(store[0], **zarr_kwargs)
        return img_xds

    img_xds = xr.Dataset()
    data_group = {}
    data_group_name = "base"

    generic_image_counter = 0
    for image_type, store_description in store_dict.items():

        store_type = store_description["store_type"]
        store = store_description["store"]

        fits_kwargs["image_type"] = image_type
        casa_kwargs["image_type"] = image_type

        if store_type == "casa":
            xds = access_store_casa(store, **casa_kwargs)
        elif store_type == "fits":
            if access_store_fits is None:
                logger.error("FITS not currently supported.")
                raise RuntimeError("FITS not currently supported.")
            xds = access_store_fits(store, **fits_kwargs)
        else:
            logger.error(
                f"Unrecognized image format for path {store}. Supported types are CASA, FITS, and zarr.\n"
            )
            raise RuntimeError(
                f"Unrecognized image format for path {store}. Supported types are CASA, FITS, and zarr.\n"
            )

        img_xds.attrs = img_xds.attrs | xds.attrs
        # print("image type:", image_type)
        img_xds[image_type] = xds[image_type]
        img_xds[image_type].attrs["type"] = image_type.lower()

        # SKY get precedence over POINT_SPREAD_FUNCTION for BEAM_FIT_PARAMS
        if "SKY" in store_dict.keys():
            if image_type == "SKY":
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]
                data_group["beam_fit_params"] = "BEAM_FIT_PARAMS"
                img_xds["BEAM_FIT_PARAMS"].attrs["type"] = "beam_fit_params"
        elif "POINT_SPREAD_FUNCTION" in store_dict.keys():
            if image_type == "POINT_SPREAD_FUNCTION":
                img_xds["BEAM_FIT_PARAMS"] = xds["BEAM_FIT_PARAMS"]
                data_group["beam_fit_params"] = "BEAM_FIT_PARAMS"
                img_xds["BEAM_FIT_PARAMS"].attrs["type"] = "beam_fit_params"

        if image_type == "MASK":
            img_xds["MASK"] = xds["MASK"]
            img_xds["MASK"].attrs["type"] = "mask"
            data_group["mask"] = "MASK"
        else:
            if "MASK_0" in xds:
                img_xds["MASK_" + image_type] = xds["MASK_0"]
                data_group["mask_" + image_type.lower()] = "MASK_" + image_type
                img_xds["MASK_" + image_type].attrs["type"] = "mask"

            if "MASK" in xds:
                img_xds["MASK_" + image_type] = xds["MASK"]
                data_group["mask_" + image_type.lower()] = "MASK_" + image_type
                img_xds["MASK_" + image_type].attrs["type"] = "mask"

        data_group[image_type.lower()] = image_type

    img_xds.attrs["type"] = "image"
    img_xds.attrs["data_groups"] = {data_group_name: data_group}
    return img_xds
