"""
UVW coordinates of the baselines of an ASDM partition.

The UVW of a baseline is the projection of its (geocentric) baseline vector
``POSITION(antenna1) - POSITION(antenna2)`` on a (u, v, w) basis attached to the
phase center, following the convention of casacore/CASA (``Muvw`` with a J2000 or
ICRS phase center, as used by importasdm):

- the ITRS baseline vector is rotated into the (geocentric) GCRS frame with the
  Earth orientation at each time (precession-nutation, Earth rotation angle /
  UT1 and polar motion, from the astropy IERS tables). Rotations do not change the
  baseline length: ``|UVW| == |POSITION(antenna1) - POSITION(antenna2)|``.
- w points towards the geocentric apparent direction of the phase center (the
  catalogue direction corrected for annual aberration and light deflection), so
  that ``w`` gives the geometric delay seen by the correlator.
- u and v are the catalogue (ICRS) east and north axes at the phase center,
  carried onto the apparent direction by the smallest rotation that maps the
  catalogue direction onto the apparent one. Hence the resulting UVW are in the
  frame of the catalogue phase center (ICRS).

No observatory/site position is used (geocentric observer, no diurnal
aberration), so the result does not depend on a hard-coded telescope site.

Phase centers can be given in the celestial frames of
``_ASTROPY_DIRECTION_FRAMES`` (converted to ICRS by astropy). Earth-fixed /
topocentric frames (``_UNSUPPORTED_DIRECTION_FRAMES``) are rejected with
NotImplementedError. :func:`check_uvw_inputs` makes that check (and the other
checks of the inputs) without calculating anything, so that the lazy UVW arrays
can fail when a partition is opened rather than when the UVW are computed.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import xarray as xr
from astropy.time import ScaleValueError, Time

# casacore/MSv4-style phase center frame names -> astropy frame names: the
# celestial frames, whose directions convert to ICRS without an observation time
# or an observatory location. The keys are the supported phase center frames
# (also used by create_field_and_source_xds to reject the other frames when a
# partition is opened).
_ASTROPY_DIRECTION_FRAMES = {
    "icrs": "icrs",
    "fk5": "fk5",
    "j2000": "fk5",
    "fk4": "fk4",
    "b1950": "fk4",
    "galactic": "galactic",
    "supergalactic": "supergalactic",
}

# Frames that a field direction can have (see
# field_source.ASDM_DIRECTION_CODE_TO_FRAME) but that are not supported as phase
# center frames, with the reason. Every frame of field_source must be in one of
# _ASTROPY_DIRECTION_FRAMES or _UNSUPPORTED_DIRECTION_FRAMES (checked by the tests).
_UNSUPPORTED_DIRECTION_FRAMES = {
    "hadec": (
        "an hour angle / declination direction is fixed with respect to the Earth: "
        "it needs an observatory location and the observation time to be converted "
        "to a sky position"
    ),
    "altaz": (
        "an azimuth / elevation direction is topocentric: it needs an observatory "
        "location and the observation time to be converted to a sky position"
    ),
    "itrs": (
        "an ITRS direction is fixed with respect to the Earth: it needs the "
        "observation time to be converted to a sky position"
    ),
}


def calculate_uvw(
    key: tuple[slice, slice, slice] | None,
    time: xr.DataArray,
    baseline_antenna1_name: xr.DataArray,
    baseline_antenna2_name: xr.DataArray,
    antenna_position: xr.DataArray,
    phase_center_direction: xr.DataArray,
) -> np.ndarray:
    """
    Calculate the UVW coordinates of a block of (time, baseline_id, uvw_label).

    Parameters
    ----------
    key : tuple[slice, slice, slice] | None
        Selection along (time, baseline_id, uvw_label), one slice per dimension,
        as produced by the lazy backend arrays (ASDMBackendArray normalises
        integer and strided keys into slices). None selects everything.
    time : xr.DataArray
        Time measure (dim ``time``), with ``units``, ``format`` and ``scale``
        attributes (for example seconds, "unix", "utc"). The values are
        interpreted with these attributes.
    baseline_antenna1_name : xr.DataArray
        Name of the first antenna of every baseline (dim ``baseline_id``).
    baseline_antenna2_name : xr.DataArray
        Name of the second antenna of every baseline (dim ``baseline_id``).
    antenna_position : xr.DataArray
        ITRS (geocentric, cartesian) antenna positions, dims
        (``antenna_name``, <cartesian label>), in the ``units`` attribute
        (default meters).
    phase_center_direction : xr.DataArray
        Phase center (ra, dec), dims (``time``, ``sky_dir_label``) with one
        direction per time, or (``field_name``, ``sky_dir_label``) with a single
        field_name (the same direction for all times). The ``units`` (default
        rad) and ``frame`` (default icrs) attributes are honoured. The frame must
        be a celestial frame (see :func:`phase_center_frame_to_astropy`).

    Returns
    -------
    np.ndarray
        float64 array of shape (num_selected_times, num_selected_baselines,
        num_selected_uvw_labels) with the UVW (meters) of the baselines
        POSITION(antenna1) - POSITION(antenna2).

    Raises
    ------
    NotImplementedError
        If the phase center frame is not supported.
    ValueError
        If the inputs are inconsistent or lack attributes (see
        :func:`check_uvw_inputs`, which checks them without calculating
        anything).
    """
    time_key, baseline_key, label_key = _check_key(key)

    antenna1_idx, antenna2_idx = _baseline_antenna_indices(
        antenna_position, baseline_antenna1_name, baseline_antenna2_name
    )
    positions = _antenna_positions_in_meters(antenna_position)
    baselines_itrs = (
        positions[antenna1_idx[baseline_key]] - positions[antenna2_idx[baseline_key]]
    )

    num_time = time.sizes["time"]
    obstime = _time_from_measure(time[time_key])
    phase_center = _phase_center_per_time(phase_center_direction, num_time, time_key)

    uvw = _calculate_uvw_astropy(obstime, phase_center, baselines_itrs)

    return np.ascontiguousarray(uvw[..., label_key], dtype=np.float64)


def check_uvw_inputs(
    time: xr.DataArray,
    baseline_antenna1_name: xr.DataArray,
    baseline_antenna2_name: xr.DataArray,
    antenna_position: xr.DataArray,
    phase_center_direction: xr.DataArray,
) -> None:
    """
    Check that :func:`calculate_uvw` can calculate the UVW of these inputs,
    without calculating anything (cheap: only the attributes, dimensions, labels
    and names are checked). It raises the errors that :func:`calculate_uvw`
    would raise later for any key, so that a lazily computed UVW array can be
    rejected when it is created (when a partition is opened) rather than when
    its values are computed.

    Parameters
    ----------
    time, baseline_antenna1_name, baseline_antenna2_name, antenna_position, phase_center_direction
        As in :func:`calculate_uvw`.

    Raises
    ------
    NotImplementedError
        If the phase center frame is not supported (see
        :func:`phase_center_frame_to_astropy`).
    ValueError
        If the time measure does not have the single dimension time or lacks
        valid units/format/scale attributes, a baseline antenna is not in the
        antenna_name coordinate, or the antenna positions or the phase center
        direction have unexpected shapes, dimensions, labels or units.
    KeyError
        If the antenna positions have no antenna_name coordinate.
    """
    if tuple(time.dims) != ("time",):
        raise ValueError(
            f"The time measure must have the single dimension 'time', got {time.dims}"
        )
    # Decodes (at most) one value: checks the format/scale/units attributes
    _time_from_measure(time.isel(time=slice(0, 1)))
    _baseline_antenna_indices(
        antenna_position, baseline_antenna1_name, baseline_antenna2_name
    )
    _antenna_positions_in_meters(antenna_position)
    _check_phase_center_direction(phase_center_direction, time.sizes["time"])


def phase_center_frame_to_astropy(frame: str) -> str:
    """
    astropy frame name of a phase center direction ``frame`` attribute.

    Parameters
    ----------
    frame : str
        MSv4 (astropy) or casacore-style frame name, case insensitive (for example
        "fk5", "ICRS", "J2000", "galactic", "supergalactic").

    Returns
    -------
    str
        astropy frame name.

    Raises
    ------
    NotImplementedError
        If UVW cannot be calculated for phase centers in this frame (for example
        the Earth-fixed or topocentric hadec, altaz, itrs frames, see
        ``_UNSUPPORTED_DIRECTION_FRAMES``).
    """
    frame_name = str(frame).strip().lower()
    if frame_name in _ASTROPY_DIRECTION_FRAMES:
        return _ASTROPY_DIRECTION_FRAMES[frame_name]
    reason = _UNSUPPORTED_DIRECTION_FRAMES.get(frame_name, "unknown frame")
    raise NotImplementedError(
        f"UVW calculation for phase centers in frame {frame_name!r} is not "
        f"supported ({reason}). Supported frames: "
        f"{sorted(_ASTROPY_DIRECTION_FRAMES)}"
    )


def _check_key(key: tuple | None) -> tuple[slice, slice, slice]:
    """Validate the (time, baseline_id, uvw_label) key (None -> everything)."""
    if key is None:
        return (slice(None), slice(None), slice(None))
    key = tuple(key)
    if len(key) != 3 or not all(isinstance(dim_key, slice) for dim_key in key):
        raise TypeError(
            "calculate_uvw expects a key with one slice for each of (time, "
            f"baseline_id, uvw_label), got {key=}"
        )
    return key


def _antenna_names_to_indices(
    antenna_name: np.ndarray, baseline_antenna_name: xr.DataArray | np.ndarray
) -> np.ndarray:
    """From baseline antenna names to indices into the antenna_name coordinate."""
    index_by_name = {
        str(name): idx for idx, name in enumerate(np.asarray(antenna_name))
    }
    names = np.asarray(baseline_antenna_name).astype(str)
    unknown = sorted(set(names) - set(index_by_name))
    if unknown:
        raise ValueError(
            f"Baseline antenna names {unknown} not found in the antenna_name "
            f"coordinate {list(index_by_name)}"
        )
    return np.array([index_by_name[name] for name in names], dtype=int)


def _baseline_antenna_indices(
    antenna_position: xr.DataArray,
    baseline_antenna1_name: xr.DataArray,
    baseline_antenna2_name: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices into the antenna positions of the two antennas of every baseline."""
    antenna_names = antenna_position.coords["antenna_name"].values
    antenna1_idx = _antenna_names_to_indices(antenna_names, baseline_antenna1_name)
    antenna2_idx = _antenna_names_to_indices(antenna_names, baseline_antenna2_name)
    if antenna1_idx.shape != antenna2_idx.shape or antenna1_idx.ndim != 1:
        raise ValueError(
            "baseline_antenna1_name and baseline_antenna2_name must be 1D with one "
            f"name per baseline, got shapes {antenna1_idx.shape} and "
            f"{antenna2_idx.shape}"
        )
    return antenna1_idx, antenna2_idx


def _unit_from_attrs(attrs: dict, default: str) -> u.Unit:
    """Single astropy unit from a ``units`` attribute (string or list of strings)."""
    units = attrs.get("units", default)
    if isinstance(units, list | tuple):
        if len(set(units)) != 1:
            raise ValueError(f"Inconsistent units for the components: {units}")
        units = units[0]
    return u.Unit(units)


def _antenna_positions_in_meters(antenna_position: xr.DataArray) -> np.ndarray:
    """(antenna, 3) ITRS cartesian positions in meters."""
    unit = _unit_from_attrs(antenna_position.attrs, "m")
    positions = np.asarray(antenna_position.values, dtype=float) * unit
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "antenna_position must have dims (antenna_name, 3 cartesian "
            f"components), got shape {positions.shape}"
        )
    return positions.to_value(u.m)


def _time_from_measure(time: xr.DataArray) -> Time:
    """astropy Time from a time measure DataArray, using its own attributes."""
    attrs = time.attrs
    missing = [name for name in ("format", "scale") if name not in attrs]
    if missing:
        raise ValueError(
            f"The time measure needs the attributes {missing} to calculate UVW "
            f"(found attrs: {attrs})"
        )
    values = np.atleast_1d(np.asarray(time.values, dtype=float))
    unit = _unit_from_attrs(attrs, "s")
    try:
        return Time(
            values * unit,
            format=str(attrs["format"]).lower(),
            scale=str(attrs["scale"]).lower(),
        )
    except (ValueError, ScaleValueError) as exc:
        raise ValueError(
            f"Cannot interpret the time measure with attributes {attrs}: {exc}"
        ) from exc


def _check_phase_center_direction(
    phase_center_direction: xr.DataArray, num_time: int
) -> tuple[str, u.Unit]:
    """
    Check the dims, sizes, labels, units and frame of the phase center direction.

    Returns
    -------
    tuple[str, u.Unit]
        astropy frame name and angle unit of the direction values.
    """
    dims = phase_center_direction.dims
    if len(dims) != 2 or dims[1] != "sky_dir_label":
        raise ValueError(
            "The phase center direction must have dims (time, sky_dir_label) or "
            f"(field_name, sky_dir_label), got {dims}"
        )
    if dims[0] == "time":
        if phase_center_direction.sizes["time"] != num_time:
            raise ValueError(
                f"The phase center direction has {phase_center_direction.sizes['time']}"
                f" times, but the time coordinate has {num_time}"
            )
    elif dims[0] == "field_name":
        if phase_center_direction.sizes["field_name"] != 1:
            raise ValueError(
                "A phase center direction with a field_name dimension must have "
                "exactly one field (pass one direction per time otherwise), got "
                f"{phase_center_direction.sizes['field_name']}"
            )
    else:
        raise ValueError(
            f"Unexpected first dimension of the phase center direction: {dims[0]}"
        )
    if phase_center_direction.sizes["sky_dir_label"] != 2:
        raise ValueError(
            "The phase center direction must have 2 sky_dir_label values (ra, dec), "
            f"got {phase_center_direction.sizes['sky_dir_label']}"
        )

    if "sky_dir_label" in phase_center_direction.coords:
        labels = [str(label) for label in phase_center_direction.sky_dir_label.values]
        if labels != ["ra", "dec"]:
            raise ValueError(f"Expected sky_dir_label ['ra', 'dec'], got {labels}")

    unit = _unit_from_attrs(phase_center_direction.attrs, "rad")
    if not unit.is_equivalent(u.rad):
        raise ValueError(
            f"The phase center direction units must be angles, got {unit.to_string()}"
        )
    frame = phase_center_frame_to_astropy(
        phase_center_direction.attrs.get("frame", "icrs")
    )
    return frame, unit


def _phase_center_per_time(
    phase_center_direction: xr.DataArray, num_time: int, time_key: slice
) -> coord.SkyCoord:
    """One phase center direction (SkyCoord) per selected time."""
    frame, unit = _check_phase_center_direction(phase_center_direction, num_time)
    if phase_center_direction.dims[0] == "time":
        lon_lat = np.asarray(phase_center_direction.values, dtype=float)[time_key]
    else:
        num_selected = len(range(num_time)[time_key])
        lon_lat = np.broadcast_to(
            np.asarray(phase_center_direction.values, dtype=float)[0],
            (num_selected, 2),
        )
    return coord.SkyCoord(lon_lat[:, 0] * unit, lon_lat[:, 1] * unit, frame=frame)


def _itrs_to_gcrs_matrices(obstime: Time) -> np.ndarray:
    """
    (time, 3, 3) rotation matrices from ITRS to (geocentric) GCRS axes.

    The ITRS->GCRS transformation of geocentric positions is a pure rotation, so
    its columns are the images of the ITRS unit vectors.
    """
    num_time = obstime.size
    identity = np.eye(3)
    unit_vectors = coord.EarthLocation.from_geocentric(
        np.broadcast_to(identity[0], (num_time, 3)),
        np.broadcast_to(identity[1], (num_time, 3)),
        np.broadcast_to(identity[2], (num_time, 3)),
        unit=u.m,
    )
    gcrs_position, _gcrs_velocity = unit_vectors.get_gcrs_posvel(obstime[:, np.newaxis])
    # xyz: (component, time, unit vector) -> (time, component, unit vector)
    return np.moveaxis(gcrs_position.xyz.to_value(u.m), 0, 1)


def _calculate_uvw_astropy(
    obstime: Time,
    phase_center: coord.SkyCoord,
    baselines_itrs: np.ndarray,
) -> np.ndarray:
    """
    UVW per time and baseline.

    Parameters
    ----------
    obstime : Time
        (time,) times of the UVW.
    phase_center : coord.SkyCoord
        (time,) phase center direction for every time (catalogue direction).
    baselines_itrs : np.ndarray
        (baseline, 3) ITRS baseline vectors in meters.

    Returns
    -------
    np.ndarray
        (time, baseline, 3) UVW in meters (see the module docstring for the
        convention).
    """
    num_time = obstime.size
    num_baseline = baselines_itrs.shape[0]
    if num_time == 0 or num_baseline == 0:
        return np.zeros((num_time, num_baseline, 3))

    # Baselines in GCRS axes (pure rotation of the ITRS vectors)
    itrs_to_gcrs = _itrs_to_gcrs_matrices(obstime)
    baselines_gcrs = np.einsum("tij,bj->tbi", itrs_to_gcrs, baselines_itrs)

    # Catalogue (ICRS) direction and its east / north axes
    catalogue = phase_center.icrs
    ra = catalogue.ra.to_value(u.rad)
    dec = catalogue.dec.to_value(u.rad)
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    w_catalogue = np.stack([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], axis=-1)
    u_catalogue = np.stack([-sin_ra, cos_ra, np.zeros_like(ra)], axis=-1)
    v_catalogue = np.stack([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec], axis=-1)

    # Geocentric apparent direction (annual aberration, light deflection), in
    # GCRS axes
    apparent = catalogue.transform_to(coord.GCRS(obstime=obstime))
    w_apparent = apparent.cartesian.xyz.value.T
    w_apparent = w_apparent / np.linalg.norm(w_apparent, axis=-1, keepdims=True)

    # Smallest rotation mapping the catalogue direction onto the apparent one
    # (Rodrigues formula without normalising the axis: well conditioned for the
    # tiny aberration angles)
    cross = np.cross(w_catalogue, w_apparent)
    one_plus_cos = 1.0 + np.sum(w_catalogue * w_apparent, axis=-1, keepdims=True)

    def rotate(vectors: np.ndarray) -> np.ndarray:
        cross_v = np.cross(cross, vectors)
        return vectors + cross_v + np.cross(cross, cross_v) / one_plus_cos

    axes = np.stack(
        [rotate(u_catalogue), rotate(v_catalogue), w_apparent], axis=1
    )  # (time, uvw axis, xyz)

    return np.einsum("tbi,tki->tbk", baselines_gcrs, axes)
