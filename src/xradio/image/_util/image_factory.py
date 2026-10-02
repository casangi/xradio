import datetime
import os

import numpy as np
import xarray as xr

from xradio._utils.dict_helpers import (
    make_direction_location_dict,
    make_quantity,
    make_skycoord_dict,
    make_spectral_coord_reference_dict,
    make_time_measure_attrs,
)
from xradio._utils.logging import xradio_logger
from xradio.image._util.common import (
    _c,
    _compute_world_sph_dims,
    _hz_per_unit,
    _l_m_attr_notes,
)
from xradio.image._util.conventions import (
    SINGLE_CHANNEL_WIDTH_FALLBACK_HZ,
    canonical_polarization_order,
    normalize_spectral_frame,
    normalize_sub_type,
    spectral_frame_to_observer,
    time_values_to_astropy,
)

# Equinox of the reference direction, by direction frame (frames that are
# absent have no equinox)
_DEFAULT_EQUINOX = {"fk5": "j2000.0", "fk4": "b1950.0"}


def _is_galactic_frame(direction_reference: str) -> bool:
    """
    Check whether a direction reference frame is Galactic.

    Parameters
    ----------
    direction_reference : str
        Direction frame name.

    Returns
    -------
    bool
        True when the frame is Galactic, otherwise False.
    """
    return direction_reference.lower() == "galactic"


def _spherical_ctype_for_frame(direction_reference: str) -> list[str]:
    """
    Get WCS CTYPE axis tokens for a direction reference frame.

    Parameters
    ----------
    direction_reference : str
        Direction frame name.

    Returns
    -------
    list[str]
        Two-element CTYPE list for longitude/latitude axes.
    """
    if _is_galactic_frame(direction_reference):
        return ["GLON", "GLAT"]
    return ["RA", "Dec"]


def _spherical_coord_names_for_frame(direction_reference: str) -> tuple[str, str]:
    """
    Get coordinate variable names for spherical sky coordinates.

    Parameters
    ----------
    direction_reference : str
        Direction frame name.

    Returns
    -------
    tuple[str, str]
        Coordinate names for longitude and latitude.
    """
    if _is_galactic_frame(direction_reference):
        return ("galactic_longitude", "galactic_latitude")
    return ("right_ascension", "declination")


def _input_checks(
    phase_center: list | np.ndarray,
    image_size: list | np.ndarray,
    cell_size: list | np.ndarray,
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

    Returns
    -------
    None
        This function validates inputs and raises on invalid shapes.
    """
    if len(image_size) != 2:
        raise ValueError("image_size must have exactly two elements")
    if len(phase_center) != 2:
        raise ValueError("phase_center must have exactly two elements")
    if len(cell_size) != 2:
        raise ValueError("cell_size must have exactly two elements")


def _time_coords_to_mjd(time_coords) -> np.ndarray:
    """
    Convert the time coordinate values given to an image factory to MJD days.

    Parameters
    ----------
    time_coords : float, sequence, np.ndarray, str, np.datetime64, datetime or astropy.time.Time
        Numbers are MJD (UTC) days and are kept as they are. ``datetime64``
        values, ISO time strings (for example ``"2020-01-01T12:00:00"``),
        ``datetime.datetime`` objects and astropy ``Time`` objects are
        converted to UTC MJD days with astropy (strings, ``datetime`` and
        ``datetime64`` values are taken as UTC).

    Returns
    -------
    np.ndarray
        One dimensional float64 array of MJD (UTC) days.

    Raises
    ------
    TypeError
        If the values are neither numbers nor times, for example
        ``timedelta64`` durations, booleans or complex numbers.
    ValueError
        If strings cannot be parsed as times.
    """
    from astropy.time import Time

    if isinstance(time_coords, Time):
        return np.atleast_1d(np.asarray(time_coords.utc.mjd, dtype=np.float64))
    values = np.atleast_1d(np.asarray(time_coords))
    kind = values.dtype.kind
    if kind in "iuf":
        return values.astype(np.float64)
    if kind in "US":
        # numbers given as strings are MJD days, as before
        try:
            return values.astype(np.float64)
        except ValueError:
            pass
    if kind == "M":
        return np.atleast_1d(Time(values, format="datetime64", scale="utc").mjd)
    if kind in "USO" and all(
        isinstance(value, str | datetime.datetime | Time) for value in values
    ):
        try:
            times = Time(values.tolist())
        except ValueError as exc:
            raise ValueError(
                f"time_coords {time_coords!r} cannot be parsed as times: {exc}"
            ) from exc
        return np.atleast_1d(np.asarray(times.utc.mjd, dtype=np.float64))
    raise TypeError(
        "time_coords must be MJD (UTC) days, datetime64 values, ISO time "
        "strings, datetime objects or astropy Time objects; got values of "
        f"type {values.dtype}"
    )


def _make_coords(
    frequency_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
) -> dict:
    """
    Build common time/frequency/velocity coordinate arrays.

    Parameters
    ----------
    frequency_coords : list or np.ndarray
        Frequency coordinate values in Hz.
    time_coords : list or np.ndarray
        Time coordinate values in MJD days, or times that astropy converts to
        MJD (see :func:`_time_coords_to_mjd`).

    Returns
    -------
    dict
        Dictionary containing normalized coordinate arrays and a rest frequency.
    """
    frequency_coords = np.atleast_1d(np.asarray(frequency_coords, dtype=np.float64))
    restfreq = frequency_coords[len(frequency_coords) // 2]
    # _c is in m/s already. _c.to("m/s") would parse "m/s" into a new
    # CompositeUnit on every call, which astropy leaves in a reference cycle
    # (its _decomposed_cache is the unit itself): cyclic garbage per call.
    vel_coords = (1 - frequency_coords / restfreq) * _c.value
    time_coords = _time_coords_to_mjd(time_coords)
    return dict(
        chan=frequency_coords, vel=vel_coords, time=time_coords, restfreq=restfreq
    )


#: Spectral reference frames of astropy without a casacore equivalent (see
#: :data:`xradio.schema.measures.AllowedSpectralCoordFrames`). The factories
#: accept them, as before, but the CASA and FITS writers cannot write images
#: in these frames.
_ASTROPY_ONLY_SPECTRAL_FRAMES = ("icrs", "hcrs", "lsr")


def _spectral_frame_and_observer(spectral_reference: str) -> tuple[str, str]:
    """
    Translate the ``spectral_reference`` of the image factories.

    Parameters
    ----------
    spectral_reference : str
        A casacore frame (``"LSRK"``), a FITS ``SPECSYS`` value
        (``"BARYCENT"``) or a ``reference_frequency`` observer (``"lsrk"``,
        ``"gcrs"``), in any case.

    Returns
    -------
    tuple[str, str]
        The casacore frame (the frequency coordinate's ``frame`` attribute)
        and the matching observer (of ``reference_frequency``). The astropy
        frames without a casacore equivalent (``"icrs"``, ``"hcrs"``,
        ``"lsr"``) are kept as they are, in lower case, as both: such images
        cannot be written to CASA or FITS.

    Raises
    ------
    ValueError
        If the frame is neither a casacore nor an astropy spectral frame.
    """
    try:
        frame = normalize_spectral_frame(spectral_reference)
    except ValueError as exc:
        name = str(spectral_reference).strip().lower()
        if name in _ASTROPY_ONLY_SPECTRAL_FRAMES:
            return name, name
        raise ValueError(
            f"Unsupported spectral_reference: {exc} (or the astropy frames "
            f"{', '.join(_ASTROPY_ONLY_SPECTRAL_FRAMES)}, which cannot be "
            "written to CASA or FITS)"
        ) from None
    return frame, spectral_frame_to_observer(frame)


def _add_common_attrs(
    xds: xr.Dataset,
    restfreq: float,
    spectral_reference: str,
    direction_reference: str,
    phase_center: list[float] | np.ndarray,
    cell_size: list[float] | np.ndarray,
    projection: str,
) -> xr.Dataset:
    """
    Attach common image-level coordinate attributes and metadata.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to enrich with metadata.
    restfreq : float
        Rest frequency in Hz.
    spectral_reference : str
        Spectral frame identifier: a casacore frame, a FITS ``SPECSYS`` value
        or a ``reference_frequency`` observer (see
        :func:`_spectral_frame_and_observer`).
    direction_reference : str
        Direction frame identifier.
    phase_center : list[float] or np.ndarray
        Two-element phase center in radians.
    cell_size : list[float] or np.ndarray
        Pixel cell size in radians.
    projection : str
        Projection code.

    Returns
    -------
    xr.Dataset
        Input dataset with updated coordinate attrs and dataset attrs.

    Raises
    ------
    ValueError
        If ``spectral_reference`` is not a spectral reference frame.
    """
    frame, observer = _spectral_frame_and_observer(spectral_reference)
    xds.time.attrs = make_time_measure_attrs(units="d", scale="utc", time_format="mjd")
    freq_vals = np.array(xds.frequency)
    xds.frequency.attrs = {
        "frame": frame,
        "observer": observer,
        "reference_frequency": make_spectral_coord_reference_dict(
            value=freq_vals[len(freq_vals) // 2].item(),
            units="Hz",
            observer=observer,
        ),
        "rest_frequencies": make_quantity(restfreq, "Hz"),
        "rest_frequency": make_quantity(restfreq, "Hz"),
        "type": "spectral_coord",
        "units": "Hz",
        "wave_units": "mm",
    }
    xds.velocity.attrs = {"doppler_type": "radio", "type": "doppler", "units": "m/s"}
    reference = make_skycoord_dict(
        data=np.asarray(phase_center, dtype=np.float64),
        units="rad",
        frame=direction_reference,
    )
    # FK5 and FK4 coordinates need an equinox (the CASA reader gives the same
    # values for casacore's J2000 and B1950); ICRS and galactic ones have none
    equinox = _DEFAULT_EQUINOX.get(direction_reference.lower())
    if equinox is not None:
        reference["attrs"]["equinox"] = equinox
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
        "type": "image_dataset",
    }
    return xds


def _check_polarization_order(pol_coords) -> None:
    """
    Check that polarization labels are in canonical order.

    Image datasets keep the polarization axis in the casacore ``Stokes``
    order (``I, Q, U, V``; ``RR, RL, LR, LL``; ``XX, XY, YX, YY``), so that
    the correlations of a pair of feeds map onto 2x2 Jones matrices, and the
    readers return that order. The factories do not reorder the labels they
    are given, as data filled in later by position would then be mislabelled.

    Parameters
    ----------
    pol_coords : sequence of str
        Polarization labels.

    Raises
    ------
    ValueError
        If the labels are not in canonical order.
    """
    labels = [str(label) for label in np.atleast_1d(np.asarray(pol_coords))]
    order = canonical_polarization_order(labels)
    if order != list(range(len(labels))):
        raise ValueError(
            f"pol_coords {labels} are not in the canonical (casacore Stokes, "
            "Jones matrix) order of image datasets: pass them as "
            f"{[labels[i] for i in order]} (and order the image planes the same "
            "way)"
        )


def _make_common_coords(
    pol_coords: list | np.ndarray,
    frequency_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
) -> dict:
    """
    Build shared non-direction coordinates used by image constructors.

    Parameters
    ----------
    pol_coords : list or np.ndarray
        Polarization labels, in canonical order (see
        :func:`_check_polarization_order`).
    frequency_coords : list or np.ndarray
        Frequency coordinate values in Hz.
    time_coords : list or np.ndarray
        Time coordinate values in MJD days.

    Returns
    -------
    dict
        Dictionary with assembled coordinate mapping and rest frequency.

    Raises
    ------
    ValueError
        If the polarization labels are not in canonical order.
    """
    _check_polarization_order(pol_coords)
    some_coords = _make_coords(frequency_coords, time_coords)
    return {
        "coords": {
            "time": some_coords["time"],
            "frequency": some_coords["chan"],
            "velocity": ("frequency", some_coords["vel"]),
            # a tuple would be taken as (dims, data) by xarray
            "polarization": list(pol_coords)
            if isinstance(pol_coords, tuple)
            else pol_coords,
        },
        "restfreq": some_coords["restfreq"],
    }


def _make_lm_values(
    image_size: list | np.ndarray,
    cell_size: list | np.ndarray,
) -> dict:
    """
    Build linear ``l`` and ``m`` coordinate arrays from image geometry.

    Parameters
    ----------
    image_size : list or np.ndarray
        Two-element image size in pixels.
    cell_size : list or np.ndarray
        Two-element cell size in radians.

    Returns
    -------
    dict
        Dictionary containing ``l`` and ``m`` coordinate arrays.
    """
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
    image_size: list | np.ndarray,
    cell_size: list | np.ndarray,
    phase_center: list | np.ndarray,
    direction_reference: str,
) -> dict:
    """
    Compute spherical sky-coordinate grids for the requested direction frame.

    Parameters
    ----------
    projection : str
        Spherical projection code.
    image_size : list or np.ndarray
        Two-element image size in pixels.
    cell_size : list or np.ndarray
        Two-element cell size in radians.
    phase_center : list or np.ndarray
        Two-element phase center in radians.
    direction_reference : str
        Direction frame identifier.

    Returns
    -------
    dict
        Mapping from spherical coordinate names to ``(dims, values)`` tuples.
    """
    long, lat = _compute_world_sph_dims(
        projection=projection,
        shape=image_size,
        ctype=_spherical_ctype_for_frame(direction_reference),
        crpix=[image_size[0] // 2, image_size[1] // 2],
        crval=phase_center,
        cdelt=[-abs(cell_size[0]), abs(cell_size[1])],
        cunit=["rad", "rad"],
    )["value"]
    lon_name, lat_name = _spherical_coord_names_for_frame(direction_reference)
    return {lon_name: (("l", "m"), long), lat_name: (("l", "m"), lat)}


def _add_lm_coord_attrs(xds: xr.Dataset) -> None:
    """
    Attach explanatory notes to ``l`` and ``m`` coordinates. The input Dataset is modified in-place.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset containing ``l`` and ``m`` coordinates.

    Returns
    -------
    None
    """
    attr_note = _l_m_attr_notes()
    xds.l.attrs = {
        "note": attr_note["l"],
    }
    xds.m.attrs = {
        "note": attr_note["m"],
    }


def _make_empty_sky_image(
    phase_center: list | np.ndarray,
    image_size: list | np.ndarray,
    cell_size: list | np.ndarray,
    frequency_coords: list | np.ndarray,
    pol_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    """
    Create an empty sky image dataset containing only coordinates.

    Parameters
    ----------
    phase_center : list or np.ndarray
        Two-element phase center in radians.
    image_size : list or np.ndarray
        Two-element image size in pixels.
    cell_size : list or np.ndarray
        Two-element cell size in radians.
    frequency_coords : list or np.ndarray
        Frequency coordinates in Hz.
    pol_coords : list or np.ndarray
        Polarization labels.
    time_coords : list or np.ndarray
        Time coordinates in MJD days.
    direction_reference : str
        Direction frame identifier.
    projection : str
        Projection code.
    spectral_reference : str
        Spectral frame identifier.
    do_sky_coords : bool
        Whether to add spherical sky-coordinate grids.

    Returns
    -------
    xr.Dataset
        Empty image dataset with coordinates and metadata.
    """
    _input_checks(phase_center, image_size, cell_size)
    # fail on an unsupported frame before computing any coordinates
    _spectral_frame_and_observer(spectral_reference)
    phase_center = np.asarray(phase_center, dtype=np.float64)
    cc = _make_common_coords(pol_coords, frequency_coords, time_coords)
    coords = cc["coords"]
    lm_values = _make_lm_values(image_size, cell_size)
    coords.update(lm_values)
    if do_sky_coords:
        coords.update(
            _make_sky_coords(
                projection, image_size, cell_size, phase_center, direction_reference
            )
        )
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
    image_size: list | np.ndarray,
    sky_image_cell_size: list | np.ndarray,
) -> xr.Dataset:
    """
    Attach ``u`` and ``v`` coordinates to a dataset.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to augment with uv coordinates.
    image_size : list or np.ndarray
        Two-element image size in pixels.
    sky_image_cell_size : list or np.ndarray
        Two-element sky image cell size in radians.

    Returns
    -------
    xr.Dataset
        Dataset with ``u`` and ``v`` coordinates and attrs. As written by the
        CASA image reader, the attrs hold the units (``"lambda"``), the value
        at the reference pixel (``crval``, 0) and the increment (``cdelt``).
    """
    uv_values = _make_uv_values(image_size, sky_image_cell_size)
    xds = xds.assign_coords(uv_values)
    uv_cell_size = _uv_cell_size(image_size, sky_image_cell_size)
    for i, name in enumerate(("u", "v")):
        xds[name].attrs = {
            "units": "lambda",
            "crval": 0.0,
            "cdelt": float(uv_cell_size[i]),
            "type": "quantity",
        }
    return xds


def _uv_cell_size(
    image_size: list | np.ndarray,
    sky_image_cell_size: list | np.ndarray,
) -> np.ndarray:
    """
    Compute the ``u`` and ``v`` increments of an aperture image.

    Parameters
    ----------
    image_size : list or np.ndarray
        Two-element image size in pixels.
    sky_image_cell_size : list or np.ndarray
        Two-element sky image cell size in radians.

    Returns
    -------
    np.ndarray
        Two-element array of ``u`` and ``v`` increments, in wavelengths.
    """
    im_size_wave = 1 / np.array(sky_image_cell_size)
    return im_size_wave / np.array(image_size)


def _make_uv_values(
    image_size: list | np.ndarray,
    sky_image_cell_size: list | np.ndarray,
) -> dict:
    """
    Compute linear ``u`` and ``v`` coordinate arrays.

    Parameters
    ----------
    image_size : list or np.ndarray
        Two-element image size in pixels.
    sky_image_cell_size : list or np.ndarray
        Two-element sky image cell size in radians.

    Returns
    -------
    dict
        Dictionary containing ``u`` and ``v`` coordinate arrays.
    """
    uv_cell_size = _uv_cell_size(image_size, sky_image_cell_size)
    u_vals = [(i - image_size[0] // 2) * uv_cell_size[0] for i in range(image_size[0])]
    v_vals = [(i - image_size[1] // 2) * uv_cell_size[1] for i in range(image_size[1])]
    return {"u": u_vals, "v": v_vals}


def _make_empty_aperture_image(
    phase_center: list | np.ndarray,
    image_size: list | np.ndarray,
    sky_image_cell_size: list | np.ndarray,
    frequency_coords: list | np.ndarray,
    pol_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
    direction_reference: str,
    projection: str,
    spectral_reference: str,
) -> xr.Dataset:
    """
    Create an empty aperture image dataset containing only coordinates.

    Parameters
    ----------
    phase_center : list or np.ndarray
        Two-element phase center in radians.
    image_size : list or np.ndarray
        Two-element image size in pixels.
    sky_image_cell_size : list or np.ndarray
        Two-element sky image cell size in radians.
    frequency_coords : list or np.ndarray
        Frequency coordinates in Hz.
    pol_coords : list or np.ndarray
        Polarization labels.
    time_coords : list or np.ndarray
        Time coordinates in MJD days.
    direction_reference : str
        Direction frame identifier.
    projection : str
        Projection code.
    spectral_reference : str
        Spectral frame identifier.

    Returns
    -------
    xr.Dataset
        Empty aperture-image dataset with coordinates and metadata.
    """
    _input_checks(phase_center, image_size, sky_image_cell_size)
    # fail on an unsupported frame before computing any coordinates
    _spectral_frame_and_observer(spectral_reference)
    phase_center = np.asarray(phase_center, dtype=np.float64)
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
    phase_center: list | np.ndarray,
    image_size: list | np.ndarray,
    sky_image_cell_size: list | np.ndarray,
    frequency_coords: list | np.ndarray,
    pol_coords: list | np.ndarray,
    time_coords: list | np.ndarray,
    direction_reference: str,
    projection: str,
    spectral_reference: str,
    do_sky_coords: bool,
) -> xr.Dataset:
    """
    Create an empty image dataset with both lm and uv coordinates.

    Parameters
    ----------
    phase_center : list or np.ndarray
        Two-element phase center in radians.
    image_size : list or np.ndarray
        Two-element image size in pixels.
    sky_image_cell_size : list or np.ndarray
        Two-element sky image cell size in radians.
    frequency_coords : list or np.ndarray
        Frequency coordinates in Hz.
    pol_coords : list or np.ndarray
        Polarization labels.
    time_coords : list or np.ndarray
        Time coordinates in MJD days.
    direction_reference : str
        Direction frame identifier.
    projection : str
        Projection code.
    spectral_reference : str
        Spectral frame identifier.
    do_sky_coords : bool
        Whether to include spherical sky-coordinate grids.

    Returns
    -------
    xr.Dataset
        Empty image dataset with lm and uv coordinate systems.
    """
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
            elif ".zattrs" in os.listdir(store) or "zarr.json" in os.listdir(store):
                store_type = "zarr"
            else:
                xradio_logger().error("Unknown directory structure.")
                raise ValueError("Unknown directory structure." + str(store))
        else:
            xradio_logger().error("Path does not exist.")
            raise ValueError(
                f"Path does not exist. The current path: {os.getcwd()} The "
                f"given store {store}"
            )
    else:
        store_type = "zarr"

    return store_type


#: Image type of a store named after an image role: the last dot separated
#: token of its name (after removing a ".fits" extension), when it is one of
#: these (tclean product names, for example "target.psf" or
#: "target.residual.fits", and the data group roles the CASA and FITS writers
#: end output names with, for example "out.base.point_spread_function.fits").
#: "SKY" stands for a sky or an aperture image, told apart by the coordinates
#: of CASA images.
_IMAGE_TYPE_OF_NAME_TOKEN = {
    "image": "SKY",
    "im": "SKY",
    "sky": "SKY",
    "fits": "SKY",
    "aperture": "APERTURE",
    "psf": "POINT_SPREAD_FUNCTION",
    "point_spread_function": "POINT_SPREAD_FUNCTION",
    "pb": "PRIMARY_BEAM",
    "primary_beam": "PRIMARY_BEAM",
    "residual": "SKY_RESIDUAL",
    "model": "SKY_MODEL",
    "dirty": "SKY_DIRTY",
    "mask": "MASK",
    "sumwt": "VISIBILITY_NORMALIZATION",
    "visibility_normalization": "VISIBILITY_NORMALIZATION",
    "visibility": "VISIBILITY",
    "uv_sampling": "UV_SAMPLING",
    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
    "aperture_normalization": "APERTURE_NORMALIZATION",
    # zarr stores hold whole image datasets
    "zarr": "ALL",
}

#: Last name tokens that name no particular image: file name extensions of
#: sky images and the role "sky". Stores whose name ends in one of them, or in
#: a token that names no role, are classified by
#: :data:`_IMAGE_TYPE_OF_NAME_SUBSTRING`.
_GENERIC_NAME_TOKENS = frozenset({"image", "im", "sky", "fits"})

#: Substrings of a store name (the whole file name, in lower case) and the
#: image types they give, in order of precedence: the classification of
#: xradio 1.2.4 and earlier, used when the last token of the name names no
#: particular image, so that names such as "target_psf.im", "ngc1234_pb" or
#: "cube_residual" keep their image type. Any name containing "fits", "image"
#: or "sky" is a sky image, as before. Two rules of 1.2.4 are not kept: "uv"
#: (aperture images are told apart by the coordinates of CASA images instead)
#: and "im" inside a word (only the token "im" counts, see
#: :func:`_image_type_from_name_substrings`, so that for example
#: "simulation_mask" is a mask); the role names of the image schema were added.
_IMAGE_TYPE_OF_NAME_SUBSTRING = (
    ("fits", "SKY"),
    ("image", "SKY"),
    ("sky", "SKY"),
    ("point_spread_function", "POINT_SPREAD_FUNCTION"),
    ("psf", "POINT_SPREAD_FUNCTION"),
    ("model", "SKY_MODEL"),
    ("residual", "SKY_RESIDUAL"),
    ("dirty", "SKY_DIRTY"),
    ("primary_beam", "PRIMARY_BEAM"),
    ("pb", "PRIMARY_BEAM"),
    ("aperture_normalization", "APERTURE_NORMALIZATION"),
    ("aperture", "APERTURE"),
    ("visibility_normalization", "VISIBILITY_NORMALIZATION"),
    ("visibility", "VISIBILITY"),
    ("sumwt", "VISIBILITY_NORMALIZATION"),
    ("uv_sampling_normalization", "UV_SAMPLING_NORMALIZATION"),
    ("uv_sampling", "UV_SAMPLING"),
    ("zarr", "ALL"),
)

#: Image types given as store dict keys that name a role by another name:
#: tclean product names, and the former image types of the deconvolution
#: products, which are versions of the sky image, each in its own data group.
_IMAGE_TYPE_ALIASES = {
    "IMAGE": "SKY",
    "PSF": "POINT_SPREAD_FUNCTION",
    "PB": "PRIMARY_BEAM",
    "SUMWT": "VISIBILITY_NORMALIZATION",
    "MODEL": "SKY_MODEL",
    "RESIDUAL": "SKY_RESIDUAL",
    "DIRTY": "SKY_DIRTY",
    "MASK_DECONVOLVE": "MASK",
}


def _casa_image_plane(store: str) -> str | None:
    """
    Tell whether a CASA image is a sky or an aperture image.

    Parameters
    ----------
    store : str
        Path to an image store.

    Returns
    -------
    str or None
        ``"SKY"`` for a CASA image with a direction coordinate,
        ``"APERTURE"`` for one with a linear ``UU``/``VV`` coordinate, and
        ``None`` when ``store`` is not a CASA image or its coordinates cannot
        be read (for example without python-casacore and casatools; the image
        reader then reports the problem).
    """
    if not os.path.isfile(os.path.join(store, "table.info")):
        return None
    try:
        from xradio._utils._casacore.tables import open_table_ro

        with open_table_ro(store) as table:
            if "coords" not in table.keywordnames():
                return None
            coords = table.getkeyword("coords")
    except Exception:
        # detection only: the image reader reports unreadable images
        return None
    if any(name.startswith("direction") for name in coords):
        return "SKY"
    for name, coord in coords.items():
        if name.startswith("linear") and isinstance(coord, dict):
            axes = {str(axis).upper() for axis in coord.get("axes", [])}
            if {"UU", "VV"} <= axes:
                return "APERTURE"
    return None


def _image_type_from_name_substrings(name: str) -> str | None:
    """
    Classify a store name by the substrings it contains.

    Parameters
    ----------
    name : str
        File name of the store, in lower case.

    Returns
    -------
    str or None
        The image type of the first substring of
        :data:`_IMAGE_TYPE_OF_NAME_SUBSTRING` in ``name``; else ``"SKY"`` for
        a name with the token ``im`` (``"my_mask.im"``), ``"MASK"`` for a name
        containing ``mask``, and None when nothing matches.
    """
    for substring, image_type in _IMAGE_TYPE_OF_NAME_SUBSTRING:
        if substring in name:
            return image_type
    if "im" in name.split("."):
        return "SKY"
    if "mask" in name:
        return "MASK"
    return None


def detect_image_type(store):
    """
    Detect the image type of a store from its name and, for CASA images, its
    coordinates.

    The type is that of the last dot separated token of the store name
    (after removing a ``.fits`` extension) when that token names an image
    role, for example ``target.psf``, ``target.residual.fits`` or
    ``out.base.point_spread_function.fits`` (see
    :data:`_IMAGE_TYPE_OF_NAME_TOKEN`). Names that end in a token that names
    no particular image (``image``, ``im``, ``sky``) or no role at all are
    classified by the role names they contain, as xradio 1.2.4 and earlier
    did (see :data:`_IMAGE_TYPE_OF_NAME_SUBSTRING`): ``target_psf.im`` and
    ``ngc1234_pb`` are a point spread function and a primary beam, and any
    name containing ``fits``, ``image`` or ``sky`` (``my_psf.image``,
    ``target_pb.fits``) is a sky image. Whether a CASA image classified as a
    sky or an aperture image, or with no role in its name, holds a sky or an
    aperture image is decided by its coordinate system (direction or linear
    ``UU``/``VV`` axes); other images without a role in their name are sky
    images, with a warning. Pass a dict ``{role: path}`` to
    :func:`xradio.image.open_image` (or ``image_type`` to the xarray engines)
    to choose the role.

    Parameters
    ----------
    store : str or other
        Path to the image store. If not a string, returns 'ALL'.

    Returns
    -------
    str
        The detected image type, for example:
        - 'SKY': Sky image (``image``, ``im``, ``sky``, or no role in the name)
        - 'APERTURE': Aperture image (a CASA image with UU/VV axes)
        - 'POINT_SPREAD_FUNCTION': PSF image (``psf``, ``point_spread_function``)
        - 'PRIMARY_BEAM': Primary beam image (``pb``, ``primary_beam``)
        - 'SKY_MODEL', 'SKY_RESIDUAL', 'SKY_DIRTY': Model, residual and dirty
          sky images (``model``, ``residual``, ``dirty``)
        - 'MASK': Deconvolution mask (``mask``)
        - 'VISIBILITY_NORMALIZATION': Sum of weights (``sumwt``)
        - 'ALL': zarr stores, which hold whole image datasets, and non-string
          stores
    """
    if not isinstance(store, str):
        return "ALL"
    # Classify by the file name only: matching against the full path would
    # pick up tokens from directory names.
    full_name = os.path.basename(os.path.normpath(store)).lower()
    name = full_name
    if name.endswith(".fits"):
        name = name[: -len(".fits")]
    last_token = name.split(".")[-1]
    if last_token in _IMAGE_TYPE_OF_NAME_TOKEN and (
        last_token not in _GENERIC_NAME_TOKENS
    ):
        image_type = _IMAGE_TYPE_OF_NAME_TOKEN[last_token]
    else:
        image_type = _image_type_from_name_substrings(full_name)
    if image_type == "ALL" or (
        os.path.isdir(store)
        and (
            os.path.isfile(os.path.join(store, ".zattrs"))
            or os.path.isfile(os.path.join(store, "zarr.json"))
        )
    ):
        return "ALL"
    if image_type in (None, "SKY", "APERTURE"):
        plane = _casa_image_plane(store)
        if image_type is None and plane is None:
            xradio_logger().warning(
                f"The name of {store} names no image role: it is opened as a sky "
                "image (SKY). To open it as another image, pass a dict such as "
                "{'point_spread_function': path} to open_image, or image_type to "
                "the xarray engines."
            )
        image_type = plane or image_type or "SKY"
    return image_type


def _normalize_image_type(image_type: str) -> str:
    """
    Return the image type (the data variable name) for a store dict key.

    Parameters
    ----------
    image_type : str
        Store dict key, for example ``"sky"``, ``"psf"`` or ``"residual"``.

    Returns
    -------
    str
        Upper case image type, with the aliases in :data:`_IMAGE_TYPE_ALIASES`
        replaced, for example ``"POINT_SPREAD_FUNCTION"`` or
        ``"SKY_RESIDUAL"``.
    """
    image_type = image_type.upper()
    return _IMAGE_TYPE_ALIASES.get(image_type, image_type)


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
        - list: List of store paths (image types are detected by
          :func:`detect_image_type`)
        - dict: Mapping of image types to store paths (the image types are
          case insensitive; ``psf``, ``pb``, ``sumwt``, ``image``, ``model``,
          ``residual``, ``dirty`` and ``mask_deconvolve`` are taken as
          ``point_spread_function``, ``primary_beam``,
          ``visibility_normalization``, ``sky``, ``sky_model``,
          ``sky_residual``, ``sky_dirty`` and ``mask``)

    Returns
    -------
    store_dict : dict
        Dictionary with image types as keys. Each value is a dict with:
        - 'store_type': str, the format ('casa', 'fits', or 'zarr')
        - 'store': str, the path to the store
    data_groups : dict
        The data groups: one per sky image (``base`` for ``SKY``, ``model``
        for ``SKY_MODEL``, ...) or aperture image (``base``), or a single
        ``base`` group if there is neither. The other images are added to
        every group by :func:`create_image_xds_from_store`.

    Raises
    ------
    ValueError
        If duplicate image types are found.
    """
    store_list = None
    if isinstance(store_to_label, os.PathLike):
        store_to_label = os.fspath(store_to_label)
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
        if isinstance(store, os.PathLike):
            store = os.fspath(store)
        if isinstance(store, str):
            # as write_image does
            store = os.path.expanduser(store)
        if isinstance(image_type, int):
            image_type = detect_image_type(store)

        image_type = _normalize_image_type(image_type)

        store_type = detect_store_type(store)

        if image_type in store_dict:
            xradio_logger().error(
                f"Duplicate image type {image_type} detected in store list."
            )
            example = "store={'sky': 'a.image', 'sky_residual': 'b.image'}"
            raise ValueError(
                f"Duplicate image type {image_type} detected in store list. Please "
                "ensure each image type is unique, labelling the stores with their "
                f"image types if needed, for example {example}. The store dict"
                + str(store_dict)
            )

        if store_type == "zarr":
            image_type = "ALL"  # Zarr can have multiple data variables.

        store_dict[image_type] = {"store_type": store_type, "store": store}

    data_groups = {}

    for image_type in store_dict.keys():
        if "sky" in image_type.lower():
            if "sky" == image_type.lower():
                data_groups["base"] = {"sky": image_type}
            else:
                data_group_name = image_type.lower().replace("sky_", "")
                data_groups[data_group_name] = {"sky": image_type}
        if "aperture" == image_type.lower():
            data_groups["base"] = {"aperture": image_type}
    if not data_groups and "ALL" not in store_dict:
        # images without a sky or aperture image (a point spread function
        # alone, for example) still belong to a data group
        data_groups["base"] = {}

    return store_dict, data_groups


#: Images opened together share their dimension coordinates. The values of
#: the same axis computed by different readers (CASA and FITS) differ by
#: round-off, so values that agree within this fraction of the axis increment
#: (a pixel or a channel) are taken as equal.
_COORD_MATCH_TOLERANCE = 1e-6
#: Times that agree within this many seconds are taken as equal.
_TIME_MATCH_TOLERANCE_S = 1e-3
#: Seconds per time unit, for the time coordinate units the schema allows.
_SECONDS_PER_TIME_UNIT = {"d": 86400.0, "s": 1.0}


def _seconds_per_time_unit(units) -> float:
    """
    Return the number of seconds in a time unit.

    Parameters
    ----------
    units : str or list of str or None
        Time coordinate units (``None`` means days).

    Returns
    -------
    float
        Seconds per unit.
    """
    if isinstance(units, list | tuple):
        units = units[0] if units else None
    units = units or "d"
    if units in _SECONDS_PER_TIME_UNIT:
        return _SECONDS_PER_TIME_UNIT[units]
    import astropy.units as u

    return float(u.Unit(units).to(u.s))


def _coord_offsets_and_tolerance(
    reference: xr.DataArray, other: xr.DataArray
) -> tuple[np.ndarray, float, str, float | None]:
    """
    Compare the values of a numeric dimension coordinate of two images.

    Parameters
    ----------
    reference : xr.DataArray
        The coordinate of the images opened first.
    other : xr.DataArray
        The same coordinate of the image being added, with the same length.

    Returns
    -------
    offsets : np.ndarray
        ``other - reference``, in the units of ``reference`` (in seconds for
        ``time``).
    tolerance : float
        The largest offset that is taken as round-off, in the same units.
    units : str
        The units of ``offsets`` and ``tolerance``, for messages.
    increment : float or None
        The axis increment the tolerance is based on (None for time and for
        single valued axes).
    """
    name = reference.name
    if name == "time":
        keys = ("units", "format", "scale")
        if all(reference.attrs.get(k) == other.attrs.get(k) for k in keys):
            offsets = (
                np.asarray(other.values, dtype=np.float64)
                - np.asarray(reference.values, dtype=np.float64)
            ) * _seconds_per_time_unit(reference.attrs.get("units"))
        else:
            offsets = np.atleast_1d(
                (
                    time_values_to_astropy(other.values, other.attrs)
                    - time_values_to_astropy(reference.values, reference.attrs)
                ).sec
            )
        return offsets, _TIME_MATCH_TOLERANCE_S, " s", None

    ref_values = np.asarray(reference.values, dtype=np.float64)
    other_values = np.asarray(other.values, dtype=np.float64)
    units = reference.attrs.get("units")
    increment = None
    if name == "frequency":
        # the readers give frequencies in Hz, but compare older datasets
        # in their own units
        ref_units = units or "Hz"
        other_values = other_values * (
            _hz_per_unit(other.attrs.get("units") or "Hz") / _hz_per_unit(ref_units)
        )
        if ref_values.size < 2:
            width = reference.attrs.get("channel_width") or other.attrs.get(
                "channel_width"
            )
            width_hz = (
                abs(float(width["data"])) * _hz_per_unit(width["attrs"]["units"])
                if isinstance(width, dict)
                else SINGLE_CHANNEL_WIDTH_FALLBACK_HZ
            )
            increment = width_hz / _hz_per_unit(ref_units)
    if increment is None and ref_values.size >= 2:
        increment = abs(ref_values[-1] - ref_values[0]) / (ref_values.size - 1)
    if increment:
        tolerance = _COORD_MATCH_TOLERANCE * increment
    else:
        # a single value (or a constant axis): only round-off is tolerated
        increment = None
        tolerance = 1e-9 * max(1.0, float(np.abs(ref_values).max()))
    return (
        other_values - ref_values,
        tolerance,
        f" {units}" if isinstance(units, str) else "",
        increment,
    )


def _has_unknown_observation_date(xds: xr.Dataset) -> bool:
    """
    Tell whether the images of a dataset have no observation date.

    The CASA and FITS readers give an image without an observation date (a
    FITS file without ``DATE-OBS`` and ``MJD-OBS``, or a CASA image with
    casacore's unset date) the time MJD 0, a placeholder, and no ``obsdate``
    attribute.

    Parameters
    ----------
    xds : xr.Dataset
        One or more images, as read by the CASA or FITS reader.

    Returns
    -------
    bool
        True when the dataset has data variables, a single time at MJD 0 and
        no image with an ``obsdate`` attribute.
    """
    if not xds.data_vars or "time" not in xds.coords or xds.sizes.get("time") != 1:
        return False
    time = xds["time"]
    if str(time.attrs.get("format") or "mjd").lower() != "mjd":
        return False
    try:
        value = float(np.asarray(time.values, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError):
        return False
    return value == 0.0 and not any(
        "obsdate" in variable.attrs for variable in xds.data_vars.values()
    )


def _share_observation_date(
    img_xds: xr.Dataset, xds: xr.Dataset, image_type: str, store
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Give images without an observation date the date of the others.

    Images opened together share their time coordinate. An image without an
    observation date has a placeholder time (see
    :func:`_has_unknown_observation_date`), which takes the time of the
    images with a date, whether it is opened before or after them.

    Parameters
    ----------
    img_xds : xr.Dataset
        The images opened so far.
    xds : xr.Dataset
        The image being added.
    image_type : str
        Image type of ``xds``, for messages.
    store : str
        Store of ``xds``, for messages.

    Returns
    -------
    tuple of xr.Dataset
        ``img_xds`` and ``xds``, one of them with the time of the other when
        only that one has an observation date.
    """
    if "time" not in img_xds.coords or "time" not in xds.coords:
        return img_xds, xds
    if img_xds.sizes.get("time") != xds.sizes.get("time"):
        return img_xds, xds
    unknown_new = _has_unknown_observation_date(xds[[image_type]])
    unknown_before = _has_unknown_observation_date(img_xds)
    if unknown_new and not unknown_before:
        xradio_logger().info(
            f"The {image_type} image {store} has no observation date (its time "
            "is the placeholder MJD 0): it takes the time of the images opened "
            "before it"
        )
        return img_xds, xds.assign_coords(time=img_xds["time"])
    if unknown_before and not unknown_new:
        xradio_logger().info(
            "The images opened before the "
            f"{image_type} image {store} have no observation date (their time "
            "is the placeholder MJD 0): they take its time"
        )
        return img_xds.assign_coords(time=xds["time"]), xds
    return img_xds, xds


def _match_shared_dimension_coords(
    img_xds: xr.Dataset, xds: xr.Dataset, image_type: str, store
) -> xr.Dataset:
    """
    Give an image the dimension coordinates of the images opened before it.

    The images opened together share their dimension coordinates, and adding
    an image to the dataset aligns it with those coordinates by value, so an
    image whose coordinates differ only by round-off (as a FITS and a CASA
    image of the same field do) would be reindexed to NaN. Coordinates that
    agree within :data:`_COORD_MATCH_TOLERANCE` of the axis increment (and
    times within :data:`_TIME_MATCH_TOLERANCE_S` seconds) are replaced by the
    shared ones; any other difference raises an error.

    Parameters
    ----------
    img_xds : xr.Dataset
        The images opened so far.
    xds : xr.Dataset
        The image being added (as returned by its reader).
    image_type : str
        Image type of ``xds``, for messages.
    store : str
        Store of ``xds``, for messages.

    Returns
    -------
    xr.Dataset
        ``xds``, with the shared dimension coordinates of ``img_xds``.

    Raises
    ------
    ValueError
        If a shared dimension has a different length, different labels, or
        values that differ by more than the tolerance.
    """
    snapped = {}
    for dim in xds.dims:
        if dim not in img_xds.coords or dim not in xds.coords:
            continue
        reference = img_xds[dim]
        other = xds[dim]
        prefix = (
            f"Cannot open the {image_type} image {store} with the images "
            f"opened before it: its {dim} coordinate"
        )
        if reference.size != other.size:
            raise ValueError(
                f"{prefix} has {other.size} values, theirs has {reference.size}"
            )
        if reference.dtype.kind not in "iuf" or other.dtype.kind not in "iuf":
            # labels, for example polarizations: the image is aligned to the
            # shared labels by value, which needs the same set of labels
            if not np.array_equal(reference.values, other.values) and sorted(
                reference.values.tolist()
            ) != sorted(other.values.tolist()):
                raise ValueError(
                    f"{prefix} has the labels {other.values.tolist()}, theirs "
                    f"has {reference.values.tolist()}"
                )
            continue
        offsets, tolerance, units, increment = _coord_offsets_and_tolerance(
            reference, other
        )
        largest = float(np.abs(offsets).max()) if offsets.size else 0.0
        if largest > tolerance:
            fraction = (
                f" ({largest / increment:.3g} of the {dim} increment)"
                if increment
                else ""
            )
            raise ValueError(
                f"{prefix} differs from theirs by up to {largest:g}{units}"
                f"{fraction}, more than the tolerance of {tolerance:g}{units}. "
                "Images opened together must share their coordinates."
            )
        if largest > 0:
            snapped[dim] = (dim, reference.values, other.attrs)
    if snapped:
        xds = xds.assign_coords(snapped)
    return xds


def create_image_xds_from_store(
    store: list | dict | str,
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
      The images must share their coordinates: values that differ by round-off
      are replaced by those of the first image, other differences raise a
      ValueError (see :func:`_match_shared_dimension_coords`).
    - BEAM_FIT_PARAMS from SKY images take precedence over POINT_SPREAD_FUNCTION.
    - Masks are renamed to MASK_<IMAGE_TYPE> for internal masks.
    - The native casacore image type (or FITS BTYPE) of each image is kept as
      its ``sub_type`` attribute when it is a casacore image type (see
      :func:`xradio.image._util.conventions.normalize_sub_type`).
    """
    store_dict, data_groups = create_store_dict(store)
    if "ALL" in store_dict and len(store_dict) > 1:
        xradio_logger().error(
            "When using a zarr store with multiple data variables, no other stores can be specified."
        )
        raise ValueError(
            "When using a zarr store with multiple data variables, no other stores can be specified."
        )

    if "ALL" in store_dict:
        img_xds = access_store_zarr(store_dict["ALL"]["store"], **zarr_kwargs)
        return img_xds

    img_xds = xr.Dataset()
    # Loop over all the input CASA and Fits images.
    for image_type, store_description in store_dict.items():
        store_type = store_description["store_type"]
        store = store_description["store"]

        fits_kwargs["image_type"] = image_type
        casa_kwargs["image_type"] = image_type

        if store_type == "casa":
            xds = access_store_casa(store, **casa_kwargs)
        elif store_type == "fits":
            if access_store_fits is None:
                xradio_logger().error("FITS not currently supported.")
                raise RuntimeError("FITS not currently supported.")
            xds = access_store_fits(store, **fits_kwargs)
        else:
            xradio_logger().error(
                f"Unrecognized image format for path {store}. Supported types are CASA, FITS, and zarr.\n"
            )
            raise RuntimeError(
                f"Unrecognized image format for path {store}. Supported types are CASA, FITS, and zarr.\n"
            )

        # images without an observation date take the date of the others
        img_xds, xds = _share_observation_date(img_xds, xds, image_type, store)
        # adding the image aligns it with the coordinates of the images
        # opened before it, so they must agree (to round-off)
        xds = _match_shared_dimension_coords(img_xds, xds, image_type, store)
        img_xds.attrs = img_xds.attrs | xds.attrs
        img_xds[image_type] = xds[image_type]
        # The backend stores the native casacore image type (or FITS BTYPE),
        # e.g. "Intensity", in the "type" attribute. Preserve it as the
        # sub image type before overwriting "type" with the data group role.
        # Only casacore image types are kept, in their schema spelling (e.g.
        # "Spectral Index" or "spectral_index" become "SpectralIndex").
        native_image_type = img_xds[image_type].attrs.get("type")
        sub_type = normalize_sub_type(native_image_type)
        if sub_type is None:
            if native_image_type and native_image_type != "Undefined":
                xradio_logger().debug(
                    f"Image type {native_image_type!r} of {store} is not a "
                    f"casacore image type allowed for {image_type}; it is not "
                    "kept as the sub_type"
                )
            img_xds[image_type].attrs.pop("sub_type", None)
        else:
            img_xds[image_type].attrs["sub_type"] = sub_type
        img_xds[image_type].attrs["type"] = image_type.lower()

        expected_flag_name = "FLAG_" + image_type

        def _add_flag_to_output(
            img_xds: xr.Dataset,
            flag_array: xr.DataArray,
            expected_flag_name: str,
            active_group: dict | None = None,
        ):
            img_xds[expected_flag_name] = flag_array
            img_xds[expected_flag_name].attrs["type"] = "flag"
            if active_group is not None:
                active_group["flag"] = expected_flag_name

        active_data_group_name = None
        # If sky image, handle internal masks and beam fit params.
        if "sky" in image_type.lower():
            for data_group_name, data_group in data_groups.items():
                if data_group.get("sky") == image_type:
                    active_data_group_name = data_group_name

            if "BEAM_FIT_PARAMS_" + image_type.upper() in xds:
                img_xds["BEAM_FIT_PARAMS_" + image_type.upper()] = xds[
                    "BEAM_FIT_PARAMS_" + image_type.upper()
                ]
                data_groups[active_data_group_name]["beam_fit_params_sky"] = (
                    "BEAM_FIT_PARAMS_" + image_type.upper()
                )
            img_xds[image_type].attrs["type"] = "sky"

        active_group = (
            data_groups[active_data_group_name]
            if active_data_group_name is not None
            else None
        )
        if expected_flag_name in xds:
            _add_flag_to_output(
                img_xds,
                xds[expected_flag_name],
                expected_flag_name,
                active_group,
            )
        elif "MASK_0" in xds:
            _add_flag_to_output(
                img_xds,
                xds["MASK_0"],
                expected_flag_name,
                active_group,
            )
        elif "MASK" in xds and image_type != "MASK":
            # Do not treat a mask image as its own flag
            _add_flag_to_output(
                img_xds,
                xds["MASK"],
                expected_flag_name,
                active_group,
            )

        # If point spread function, handle beam fit params.
        if "point_spread_function" in image_type.lower():
            if "BEAM_FIT_PARAMS_" + image_type.upper() in xds:
                img_xds["BEAM_FIT_PARAMS_" + image_type.upper()] = xds[
                    "BEAM_FIT_PARAMS_" + image_type.upper()
                ]

        # Figure out data groups.
        # Each sky image gets its own data group and shares all other images between them.
        if "sky" not in image_type.lower():
            for data_group in data_groups.values():
                data_group[image_type.lower()] = image_type

                if "point_spread_function" in image_type.lower():
                    if "BEAM_FIT_PARAMS_" + image_type.upper() in xds:
                        data_group["beam_fit_params_point_spread_function"] = (
                            "BEAM_FIT_PARAMS_" + image_type.upper()
                        )
    if (
        "visibility_normalization" not in image_type.lower()
        or len(img_xds.data_vars) > 1
    ):
        # if beam_param coord not in image type it is not auto assigned to img_xds
        # but it must be present even if unused
        if "beam_params_label" not in img_xds.dims:
            img_xds.expand_dims(beam_params_label=3)

        if "beam_params_label" not in img_xds.coords:
            img_xds = _move_beam_param_dim_coord(img_xds)
    img_xds.attrs["type"] = "image_dataset"
    img_xds.attrs["data_groups"] = data_groups
    return img_xds
